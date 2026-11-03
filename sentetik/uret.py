"""Sentetik veri üreticisi (synthetic data generator).

Gerçek projedeki BigQuery tablolarının ŞEMASINI birebir taklit eder; içerik
tamamen uydurmadır ve hiçbir gerçek müşteriye, şirkete ya da dağılıma dayanmaz.

Ürettiği tablolar (data/*.parquet):
  egitim_a        müşteri × kesim, sipariş/müşteri özellikleri + hedefler (2024-10 → 2025-10, 2026-01)
  egitim_b        aynı satırlar için site/uygulama davranış özellikleri
  egitim_a_ileri  ek eğitim + ileri test + canlı kesimler (canlıda hedefler boş)
  egitim_b_ileri  aynı satırlar için davranış özellikleri
  ilk_alim        2025-10-01 ve 2026-10-01 kesimleri için kesim sonrası ilk sipariş tarihi

Simülasyon mantığı (kısaca):
  * Her müşterinin gizli bir alım hızı ve "bırakma" olasılığı var (BG/NBD tarzı).
    Siparişler mevsimsel bir Poisson süreciyle üretilir.
  * Site/uygulama ziyaretleri iki kaynaktan gelir: (1) müşterinin genel ilgisi
    (alım hızıyla ilişkili ama aynı değil), (2) bir siparişten önceki günlerde
    artan "niyet" ziyaretleri (ürün görüntüleme, sepete ekleme, ödeme başlatma).
  * Ziyaretler müşteriye ancak kimlik bağı kurulduktan sonra (ilk giriş ya da ilk
    sipariş) atanabilir: gerçek projedeki ZAMAN KURALININ karşılığı.
  * Davranış düzeyi mevsimle kayar (ör. Ekim düşük, Ocak yüksek): kesim içi
    yüzdelik dönüşümün neden gerektiğini gösterir.

Kullanım (proje kökünde):
    python -m sentetik.uret                  # varsayılan: 60.000 müşteri
    python -m sentetik.uret --musteri 20000  # daha hızlı, daha gürültülü
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "data"

BASLANGIC = pd.Timestamp("2023-01-01")      # sipariş geçmişinin başladığı gün
BITIS = pd.Timestamp("2026-12-31")          # simülasyonun son günü
CANLI = pd.Timestamp("2026-10-01")
CANLI_BUGUN = pd.Timestamp("2026-11-02")    # canlı kesimde "bugün": yalnızca bundan önceki siparişler biliniyor

KESIM_A = list(pd.date_range("2024-10-01", "2025-10-01", freq="MS")) + [pd.Timestamp("2026-01-01")]
KESIM_ILERI = [pd.Timestamp(x) for x in ["2025-11-01", "2025-12-01", "2026-02-01", "2026-03-01",
                                         "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-10-01"]]

# Ay → çarpan (uydurma). Sipariş mevsimselliği ve davranış düzeyi mevsimselliği ayrı.
SIPARIS_MEVSIM = {1: 1.45, 2: 0.95, 3: 1.0, 4: 0.95, 5: 1.0, 6: 0.95, 7: 1.15, 8: 0.85,
                  9: 0.95, 10: 1.0, 11: 1.4, 12: 1.2}
DAVRANIS_MEVSIM = {1: 2.6, 2: 2.2, 3: 1.6, 4: 1.3, 5: 1.2, 6: 1.1, 7: 1.3, 8: 1.0,
                   9: 0.9, 10: 0.8, 11: 1.3, 12: 1.6}
PLATFORMLAR = np.array(["App", "Mobil", "Web"])
KATEGORILER = np.array(["TEKSTIL", "AKSESUAR", "AYAKKABI"])
ALT_KATEGORI = {"TEKSTIL": 9, "AKSESUAR": 4, "AYAKKABI": 3}


def kume_a(k: pd.Timestamp) -> str:
    if k <= pd.Timestamp("2025-07-01"):
        return "egitim"
    if k == pd.Timestamp("2025-10-01"):
        return "test"
    if k == pd.Timestamp("2026-01-01"):
        return "test2"
    return "tampon"


def kume_ileri(k: pd.Timestamp) -> str:
    if k == CANLI:
        return "canli"
    return "ileri" if k >= pd.Timestamp("2026-04-01") else "ek_egitim"


# --------------------------------------------------------------------------- müşteriler
def musteriler(n: int, rng: np.random.Generator) -> pd.DataFrame:
    gun_araligi = (pd.Timestamp("2026-09-30") - pd.Timestamp("2016-01-01")).days
    # Yeni üyeler daha sık: kayıt tarihini sona doğru çarpıt
    uyelik = pd.Timestamp("2016-01-01") + pd.to_timedelta((rng.beta(2.2, 1.0, n) * gun_araligi).astype(int), "D")
    hiz = rng.gamma(0.55, 2.0, n)                        # yıllık sipariş hızı (gizli)
    birakma = rng.beta(1.3, 4.0, n)                      # her siparişten sonra bırakma olasılığı
    ilgi = np.exp(0.8 * np.log1p(hiz) + rng.normal(0, 0.9, n))   # genel site ilgisi (hızla ilişkili, gürültülü)
    giris = rng.beta(2.0, 2.5, n)                        # oturum açma eğilimi (kimlik bağının ne kadar erken kurulduğu)
    plat = rng.dirichlet([1.4, 1.2, 0.6], n)             # App / Mobil / Web tercihi
    kat = rng.dirichlet([6.0, 1.4, 1.0], n)              # kategori tercihi
    fiyat = np.exp(rng.normal(0, 0.35, n))               # fiyat düzeyi
    indirim_egilimi = rng.beta(2.0, 6.0, n)
    dogum_yili = rng.integers(1955, 2004, n).astype(float)
    dogum_yili[rng.random(n) < 0.12] = np.nan            # yaşı bilinmeyenler
    return pd.DataFrame({"musteri_id": [f"M{i:07d}" for i in range(n)], "uyelik": uyelik, "hiz": hiz,
                         "birakma": birakma, "ilgi": ilgi, "giris": giris,
                         "p_app": plat[:, 0], "p_mobil": plat[:, 1], "k_tekstil": kat[:, 0], "k_aksesuar": kat[:, 1],
                         "fiyat": fiyat, "indirim_egilimi": indirim_egilimi, "dogum_yili": dogum_yili})


# --------------------------------------------------------------------------- siparişler
def siparisler(m: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mevsimsel Poisson + her siparişten sonra bırakma (thinning ile)."""
    toplam_gun = (BITIS - BASLANGIC).days + 1
    mevsim_gun = np.array([SIPARIS_MEVSIM[d.month] for d in pd.date_range(BASLANGIC, BITIS)])
    tepe = max(SIPARIS_MEVSIM.values())
    bas = np.maximum(((m["uyelik"] - BASLANGIC).dt.days.to_numpy()), 0)
    satirlar_m, satirlar_g = [], []
    for i in range(len(m)):
        oran = m["hiz"].iat[i] / 365 * tepe
        if oran <= 0:
            continue
        t = float(bas[i])
        q = m["birakma"].iat[i]
        ilk = True
        while True:
            t += rng.exponential(1 / oran) if not ilk else rng.exponential(min(1 / oran, 60))
            ilk = False
            g = int(t)
            if g >= toplam_gun:
                break
            if rng.random() > mevsim_gun[g] / tepe:
                continue
            satirlar_m.append(i); satirlar_g.append(g)
            if rng.random() < q:
                break
    s = pd.DataFrame({"mi": np.array(satirlar_m, dtype=np.int64), "gun": np.array(satirlar_g, dtype=np.int64)})
    s = s.drop_duplicates(["mi", "gun"]).sort_values(["mi", "gun"]).reset_index(drop=True)
    n = len(s)
    mi = s["mi"].to_numpy()
    s["order_date"] = BASLANGIC + pd.to_timedelta(s["gun"], "D")
    # platform
    u = rng.random(n)
    pa, pm = m["p_app"].to_numpy()[mi], m["p_mobil"].to_numpy()[mi]
    s["platform"] = np.where(u < pa, "App", np.where(u < pa + pm, "Mobil", "Web"))
    # adet, kalem
    s["adet"] = 1 + rng.poisson(0.9, n)
    s["kalem_sayisi"] = np.minimum(s["adet"], 1 + rng.poisson(0.5, n))
    # fiyat: enflasyon (ayda ~%2,5) × müşteri fiyat düzeyi × gürültü
    ay = (s["order_date"].dt.year - 2023) * 12 + s["order_date"].dt.month - 1
    birim = 900 * (1.025 ** ay.to_numpy()) * m["fiyat"].to_numpy()[mi] * np.exp(rng.normal(0, 0.25, n))
    sezon_ind = np.isin(s["order_date"].dt.month, [1, 7, 11]).astype(float)
    derinlik = np.clip(m["indirim_egilimi"].to_numpy()[mi] * (0.6 + 0.8 * sezon_ind) + rng.normal(0, 0.05, n), 0, 0.7)
    liste = birim * s["adet"].to_numpy()
    s["net_tutar"] = liste * (1 - derinlik)
    s["indirim"] = liste * derinlik
    s["indirim_derinligi"] = derinlik
    s["tutar"] = s["net_tutar"] / 1.10                  # "KDV hariç" (uydurma oran)
    eksik = rng.random(n) < 0.03                        # tutarı eksik siparişler (veri kalitesi)
    s.loc[eksik, ["net_tutar", "tutar"]] = np.nan
    s["order_id"] = np.arange(n)
    return s


def kalemler(s: pd.DataFrame, m: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Her sipariş için adet kadar ürün; kategori, alt kategori."""
    tekrar = s["adet"].to_numpy()
    k = s.loc[s.index.repeat(tekrar), ["mi", "order_date", "order_id", "net_tutar", "adet"]].reset_index(drop=True)
    k["birim_tutar"] = (k["net_tutar"] / 1.10) / k["adet"]
    mi = k["mi"].to_numpy()
    u = rng.random(len(k))
    pt, pa = m["k_tekstil"].to_numpy()[mi], m["k_aksesuar"].to_numpy()[mi]
    kat = np.where(u < pt, 0, np.where(u < pt + pa, 1, 2))
    k["category"] = KATEGORILER[kat]
    alt_sayisi = np.array([ALT_KATEGORI[c] for c in KATEGORILER])[kat]
    k["category2"] = k["category"] + "_" + (rng.integers(0, 99, len(k)) % alt_sayisi).astype(str)
    return k[["mi", "order_date", "order_id", "birim_tutar", "category", "category2"]]


# --------------------------------------------------------------------------- ziyaretler
def ziyaretler(m: pd.DataFrame, s: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Gözlenen (kimliği bağlanmış) günlük ziyaretler: genel ilgi + sipariş öncesi niyet + yarım kalan niyet."""
    n = len(m)
    aylar = pd.date_range(BASLANGIC, BITIS, freq="MS")
    parcalar = []
    ilgi = m["ilgi"].to_numpy()
    # 1) Genel ilgi: ay ay Poisson
    for a in aylar:
        gun_sayisi = a.days_in_month
        lam = ilgi * 0.02 * gun_sayisi * DAVRANIS_MEVSIM[a.month] ** 0.3
        say = rng.poisson(lam)
        mi = np.repeat(np.arange(n), say)
        gun = (a - BASLANGIC).days + rng.integers(0, gun_sayisi, len(mi))
        parcalar.append(pd.DataFrame({"mi": mi, "gun": gun, "niyet": 0}))
    # 2) Sipariş öncesi niyet: her siparişten önceki 0–20 günde 1+ ziyaret
    k = 1 + rng.poisson(2.2, len(s))
    mi = np.repeat(s["mi"].to_numpy(), k)
    gun = np.repeat(s["gun"].to_numpy(), k) - rng.geometric(0.18, k.sum()) + 1
    parcalar.append(pd.DataFrame({"mi": mi, "gun": gun, "niyet": 1}))
    # 3) Yarım kalan niyet (sepete ekleyip almayan): gürültü
    say = rng.poisson(ilgi * 0.8 * (BITIS - BASLANGIC).days / 365)
    mi = np.repeat(np.arange(n), say)
    gun = rng.integers(0, (BITIS - BASLANGIC).days, len(mi))
    parcalar.append(pd.DataFrame({"mi": mi, "gun": gun, "niyet": 1}))
    z = pd.concat(parcalar, ignore_index=True)
    z = z[(z["gun"] >= 0) & (z["gun"] <= (BITIS - BASLANGIC).days)]
    z["tarih"] = BASLANGIC + pd.to_timedelta(z["gun"], "D")

    # Kimlik bağı (zaman kuralı): ilk giriş ya da ilk sipariş, hangisi önceyse.
    ilk_sip = s.groupby("mi")["order_date"].min().reindex(range(n))
    giris_gecikme = rng.exponential(30 + 900 * (1 - m["giris"].to_numpy()))
    ilk_giris = m["uyelik"] + pd.to_timedelta(giris_gecikme.astype(int), "D")
    bag = pd.concat([ilk_sip, ilk_giris], axis=1).min(axis=1).to_numpy()
    z = z[z["tarih"].to_numpy() >= bag[z["mi"].to_numpy()]]
    # Her ziyaret, girişli olarak ancak bir olasılıkla gözlenir (ölçüm kapsamı zamanla değişir).
    ay_no = (z["tarih"].dt.year - 2024) * 12 + z["tarih"].dt.month - 6
    kapsam = np.clip(0.30 + 0.05 * ay_no.to_numpy(), 0.30, 0.75)       # 2024 sonu düşük, 2025 baharında artar
    gozlem = rng.random(len(z)) < kapsam * (0.4 + 0.6 * m["giris"].to_numpy()[z["mi"].to_numpy()])
    z = z[gozlem].copy()

    nz = len(z)
    mevsim = z["tarih"].dt.month.map(DAVRANIS_MEVSIM).to_numpy()
    niyet = z["niyet"].to_numpy()
    z["urun_goruntuleme"] = rng.poisson(mevsim * (2.0 + 4.0 * niyet))
    z["liste_goruntuleme"] = rng.poisson(mevsim * (1.5 + 1.0 * niyet))
    z["arama"] = rng.poisson(0.3 + 0.6 * niyet, nz)
    z["sepete_ekleme"] = (rng.random(nz) < 0.05 + 0.5 * niyet) * (1 + rng.poisson(0.4, nz))
    z["favori_ekleme"] = (rng.random(nz) < 0.04 + 0.15 * niyet).astype(int)
    z["odeme_baslatma"] = (rng.random(nz) < 0.01 + 0.25 * niyet).astype(int)
    z["toplam_event"] = (z["urun_goruntuleme"] + z["liste_goruntuleme"] + z["arama"] + z["sepete_ekleme"]
                         + z["favori_ekleme"] + z["odeme_baslatma"] + rng.poisson(3, nz))
    # Kanal: Mayıs 2025'ten sonra uygulama payı artar (ölçüm/kanal karışımı kayması)
    p_web = (1 - m["p_app"].to_numpy()[z["mi"].to_numpy()]) * np.where(z["tarih"] >= pd.Timestamp("2025-05-01"), 0.55, 0.85)
    z["web"] = rng.random(nz) < p_web
    gunluk = (z.groupby(["mi", "tarih"], as_index=False)
                .agg(urun_goruntuleme=("urun_goruntuleme", "sum"), liste_goruntuleme=("liste_goruntuleme", "sum"),
                     arama=("arama", "sum"), sepete_ekleme=("sepete_ekleme", "sum"),
                     favori_ekleme=("favori_ekleme", "sum"), odeme_baslatma=("odeme_baslatma", "sum"),
                     toplam_event=("toplam_event", "sum"), web=("web", "max")))
    return gunluk


# --------------------------------------------------------------------------- tablolar
def tablo_a(kesimler, kume_fn, m, s, k, canli_hedefsiz: bool) -> pd.DataFrame:
    parcalar = []
    for kz in kesimler:
        w = s[(s["order_date"] < kz) & (s["order_date"] >= kz - pd.Timedelta(days=365))].copy()
        if w.empty:
            continue
        gun = (kz - w["order_date"]).dt.days
        w["g30"], w["g90"], w["g180"], w["g270"] = gun <= 30, gun <= 90, gun <= 180, gun <= 270
        w["t270"] = np.where(w["g270"], w["tutar"], np.nan)
        w["t90"] = np.where(w["g90"], w["tutar"], np.nan)
        w["ind270"] = np.where(w["g270"] & w["net_tutar"].notna(), w["indirim"], np.nan)
        w["liste270"] = np.where(w["g270"] & w["net_tutar"].notna(), w["net_tutar"] + w["indirim"], np.nan)
        w["eksik270"] = np.where(w["g270"], w["tutar"].isna().astype(float), np.nan)
        w["is_app"], w["is_mob"], w["is_web"] = (w["platform"] == "App"), (w["platform"] == "Mobil"), (w["platform"] == "Web")
        g = w.groupby("mi")
        o = pd.DataFrame({
            "siparis_12ay": g.size(),
            "siparis_30g": g["g30"].sum(), "siparis_90g": g["g90"].sum(),
            "siparis_180g": g["g180"].sum(), "siparis_9ay": g["g270"].sum(),
            "son": g["order_date"].max(), "ilk": g["order_date"].min(),
            "adet_12ay": g["adet"].sum(), "ort_kalem": g["kalem_sayisi"].mean(),
            "pay_app": g["is_app"].mean(), "pay_mobil_web": g["is_mob"].mean(), "pay_masaustu": g["is_web"].mean(),
            "ort_sepet_9ay": g["t270"].mean(), "ort_sepet_90g": g["t90"].mean(),
            "ind_top": g["ind270"].sum(min_count=1), "liste_top": g["liste270"].sum(min_count=1),
            "tutar_eksik_oran_9ay": g["eksik270"].mean(),
        })
        son = w.sort_values(["order_date", "order_id"]).groupby("mi").tail(1).set_index("mi")
        o["gun_son_siparis"] = (kz - o["son"]).dt.days
        o["gun_ilk_siparis_12ay"] = (kz - o["ilk"]).dt.days
        o["ort_siparis_arasi_gun"] = np.where(o["siparis_12ay"] > 1,
                                              (o["son"] - o["ilk"]).dt.days / (o["siparis_12ay"] - 1), np.nan)
        o["indirim_derinligi_9ay"] = o["ind_top"] / o["liste_top"]
        o["ciro_9ay"] = o["ort_sepet_9ay"] * o["siparis_9ay"]
        o["ciro_90g"] = np.where(o["siparis_90g"] == 0, 0.0, o["ort_sepet_90g"] * o["siparis_90g"])
        o["son_indirim_derinligi"] = son["indirim_derinligi"]
        o["son_platform"] = son["platform"]
        o["son_sepet"] = son["tutar"]
        # kalemler
        kk = k[(k["order_date"] < kz) & (k["order_date"] >= kz - pd.Timedelta(days=365))].copy()
        kk["k270"] = (kz - kk["order_date"]).dt.days <= 270
        for c in KATEGORILER:
            kk[f"is_{c}"] = kk["category"] == c
        gk = kk.groupby("mi")
        o["pay_tekstil"] = gk["is_TEKSTIL"].mean()
        o["pay_aksesuar"] = gk["is_AKSESUAR"].mean()
        o["pay_ayakkabi"] = gk["is_AYAKKABI"].mean()
        o["farkli_kategori2"] = gk["category2"].nunique()
        kk["bt270"] = np.where(kk["k270"], kk["birim_tutar"], np.nan)
        kk["adet270"] = np.where(kk["k270"] & kk["birim_tutar"].notna(), 1.0, np.nan)
        o["ort_birim_fiyat_9ay"] = gk["bt270"].sum(min_count=1) / gk["adet270"].sum(min_count=1)
        # müşteri
        mm = m.iloc[o.index]
        ilk_tum = s.groupby("mi")["order_date"].min().reindex(o.index)
        bas = pd.Series(np.minimum(mm["uyelik"].to_numpy(), ilk_tum.to_numpy()), index=o.index)
        o["uyelik_gun"] = (kz - bas).dt.days
        o["yas"] = kz.year - mm["dogum_yili"].to_numpy()
        # hedefler
        f = s[(s["order_date"] >= kz) & (s["order_date"] < kz + pd.Timedelta(days=90))].groupby("mi")
        y_say = f.size().reindex(o.index).fillna(0).astype(int)
        y_ciro = f["tutar"].sum().reindex(o.index).fillna(0.0)
        o["y_etiket"] = y_say > 0
        o["y_siparis_90g"] = y_say.astype(float)
        o["y_ciro_90g"] = y_ciro
        if canli_hedefsiz and kz == CANLI:
            o["y_etiket"] = pd.NA
            o["y_siparis_90g"] = np.nan
            o["y_ciro_90g"] = np.nan
        o["kesim"] = kz
        o["musteri_id"] = m["musteri_id"].to_numpy()[o.index]
        o["kume"] = kume_fn(kz)
        o["kesim_ay"] = kz.month
        parcalar.append(o.reset_index(drop=True))
    df = pd.concat(parcalar, ignore_index=True)
    sira = ["kesim", "musteri_id", "kume", "kesim_ay", "siparis_12ay", "siparis_30g", "siparis_90g", "siparis_180g",
            "siparis_9ay", "gun_son_siparis", "gun_ilk_siparis_12ay", "ort_siparis_arasi_gun", "adet_12ay", "ort_kalem",
            "pay_app", "pay_mobil_web", "pay_masaustu", "ort_sepet_9ay", "ort_sepet_90g", "indirim_derinligi_9ay",
            "tutar_eksik_oran_9ay", "ciro_9ay", "ciro_90g", "son_indirim_derinligi", "son_platform", "son_sepet",
            "pay_tekstil", "pay_aksesuar", "pay_ayakkabi", "farkli_kategori2", "ort_birim_fiyat_9ay", "uyelik_gun",
            "yas", "y_etiket", "y_siparis_90g", "y_ciro_90g"]
    df = df[sira]
    for c in ["siparis_12ay", "siparis_30g", "siparis_90g", "siparis_180g", "siparis_9ay", "gun_son_siparis",
              "gun_ilk_siparis_12ay", "adet_12ay", "farkli_kategori2", "uyelik_gun", "kesim_ay"]:
        df[c] = df[c].astype("int64")
    df["y_etiket"] = df["y_etiket"].astype("boolean")
    return df


def tablo_b(a: pd.DataFrame, m: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    idx = pd.Series(np.arange(len(m)), index=m["musteri_id"])
    parcalar = []
    for kz, p in a.groupby("kesim"):
        mi = idx.loc[p["musteri_id"]].to_numpy()
        w = z[(z["tarih"] < kz) & (z["tarih"] >= kz - pd.Timedelta(days=30)) & z["mi"].isin(mi)].copy()
        w["son7"] = w["tarih"] >= kz - pd.Timedelta(days=7)
        w["ug7"] = w["urun_goruntuleme"] * w["son7"]
        w["se7"] = w["sepete_ekleme"] * w["son7"]
        g = w.groupby("mi")
        o = pd.DataFrame({
            "aktif_gun_30g": g.size(), "aktif_gun_7g": g["son7"].sum(),
            "gun_son_aktivite": (kz - g["tarih"].max()).dt.days,
            "urun_gor_30g": g["urun_goruntuleme"].sum(),
            "urun_gor_7g": g["ug7"].sum(),
            "liste_gor_30g": g["liste_goruntuleme"].sum(), "arama_30g": g["arama"].sum(),
            "sepete_ekle_30g": g["sepete_ekleme"].sum(),
            "sepete_ekle_7g": g["se7"].sum(),
            "favori_30g": g["favori_ekleme"].sum(), "odeme_baslat_30g": g["odeme_baslatma"].sum(),
            "toplam_event_30g": g["toplam_event"].sum(), "web_gun_payi": g["web"].mean(),
        })
        b = pd.DataFrame({"kesim": kz, "musteri_id": p["musteri_id"].to_numpy(), "mi": mi}).join(o, on="mi")
        b["davranis_var_30g"] = b["aktif_gun_30g"].notna()
        parcalar.append(b.drop(columns="mi"))
    b = pd.concat(parcalar, ignore_index=True)
    sayim = ["aktif_gun_7g", "aktif_gun_30g", "urun_gor_7g", "urun_gor_30g", "liste_gor_30g", "arama_30g",
             "sepete_ekle_7g", "sepete_ekle_30g", "favori_30g", "odeme_baslat_30g", "toplam_event_30g"]
    b[sayim] = b[sayim].fillna(0).astype("int64")
    b["gun_son_aktivite"] = b["gun_son_aktivite"].astype("float64")
    return b[["kesim", "musteri_id", "davranis_var_30g", "aktif_gun_7g", "aktif_gun_30g", "gun_son_aktivite",
              "urun_gor_7g", "urun_gor_30g", "liste_gor_30g", "arama_30g", "sepete_ekle_7g", "sepete_ekle_30g",
              "favori_30g", "odeme_baslat_30g", "toplam_event_30g", "web_gun_payi"]]


def tablo_ilk_alim(a: pd.DataFrame, a_ileri: pd.DataFrame, m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    s = s.assign(musteri_id=m["musteri_id"].to_numpy()[s["mi"].to_numpy()])
    parcalar = []
    for kz, pop, son in [(pd.Timestamp("2025-10-01"), a, None), (CANLI, a_ileri, CANLI_BUGUN)]:
        p = pop.loc[pop["kesim"] == kz, ["kesim", "musteri_id"]]
        ust = kz + pd.Timedelta(days=90) if son is None else min(son, kz + pd.Timedelta(days=90))
        f = s[(s["order_date"] >= kz) & (s["order_date"] < ust)].groupby("musteri_id")["order_date"].min()
        parcalar.append(p.assign(ilk_alim_tarihi=p["musteri_id"].map(f)))
    return pd.concat(parcalar, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sinyal sentetik veri üreticisi")
    ap.add_argument("--musteri", type=int, default=60_000, help="müşteri havuzu büyüklüğü")
    ap.add_argument("--tohum", type=int, default=2026)
    arg = ap.parse_args(argv)
    rng = np.random.default_rng(arg.tohum)
    t0 = time.time()
    VERI.mkdir(exist_ok=True)
    m = musteriler(arg.musteri, rng);                 print(f"müşteri: {len(m):,}")
    s = siparisler(m, rng);                           print(f"sipariş: {len(s):,}  ({time.time() - t0:.0f} sn)")
    k = kalemler(s, m, rng);                          print(f"kalem:   {len(k):,}")
    z = ziyaretler(m, s, rng);                        print(f"gözlenen ziyaret-günü: {len(z):,}  ({time.time() - t0:.0f} sn)")
    a = tablo_a(KESIM_A, kume_a, m, s, k, canli_hedefsiz=False)
    a1 = tablo_a(KESIM_ILERI, kume_ileri, m, s, k, canli_hedefsiz=True)
    print(f"egitim_a: {len(a):,} satır · egitim_a_ileri: {len(a1):,} satır  ({time.time() - t0:.0f} sn)")
    b = tablo_b(a, m, z)
    b1 = tablo_b(a1, m, z)
    ilk = tablo_ilk_alim(a, a1, m, s)
    for ad, t in [("egitim_a", a), ("egitim_b", b), ("egitim_a_ileri", a1), ("egitim_b_ileri", b1), ("ilk_alim", ilk)]:
        t.to_parquet(VERI / f"{ad}.parquet", index=False)
    ozet = (a.groupby("kesim").agg(satir=("musteri_id", "size"), etiket=("y_etiket", "mean"),
                                   tek_siparis=("siparis_12ay", lambda x: (x == 1).mean()))
              .join(b.groupby("kesim")["davranis_var_30g"].mean().rename("kapsam")))
    print(ozet.round(3).to_string())
    print(f"Tamam: {VERI} ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
