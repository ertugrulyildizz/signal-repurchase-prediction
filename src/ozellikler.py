"""Kolon grupları ve özellik dönüşümleri (tek doğruluk kaynağı)."""
from __future__ import annotations

import numpy as np
import pandas as pd

KIMLIK = ["kesim", "musteri_id", "kume"]
HEDEFLER = ["y_etiket", "y_siparis_90g", "y_ciro_90g"]   # ASLA özellik değil
ANALIZ = ["kesim_ay"]                                     # yalnızca analiz için
# Nominal TL kolonları: modele ham halleriyle girmez, endekslenmiş (_e) halleri girer
PARASAL = [
    "ort_sepet_9ay", "ort_sepet_90g", "ciro_9ay", "ciro_90g",
    "son_sepet", "ort_birim_fiyat_9ay",
]
KATEGORIK = ["son_platform"]

# EDA sonucu çıkarılanlar: |Spearman r| > 0,85 ya da yapısal tekrar
GEREKSIZ = [
    "ciro_90g_e",             # siparis_90g ile çok yüksek korelasyon
    "ort_sepet_90g_e",        # ort_sepet_9ay_e ve son_sepet_e ile yüksek korelasyon
    "son_sepet_e",            # ort_sepet_9ay_e ile yüksek korelasyon; erken kesimlerde tutar boşluğuna düşebiliyor
    "son_indirim_derinligi",  # indirim_derinligi_9ay ile yüksek korelasyon
    "pay_masaustu",           # pay_app + pay_mobil_web + pay_masaustu = 1 (referans kategori)
]

# Model B: davranış özellikleri (sql/22_egitim_b.sql → egitim_b)
BAYRAK = ["davranis_var_30g"]
# Sayımlar: düzeyleri kesimden kesime kayıyor (büyük kısmı mevsimsel)
# → modele kesim içi göreli (_y) olarak girer. tanimlar v1.2: yüzdelik sıra YALNIZCA
# davranışı olanlar arasında alınır, davranışsızlar 0 (kapsam kayması özelliğe sızmasın).
DAVRANIS_SAYIM = [
    "aktif_gun_7g", "aktif_gun_30g",
    "urun_gor_7g", "urun_gor_30g", "liste_gor_30g", "arama_30g",
    "sepete_ekle_7g", "sepete_ekle_30g", "favori_30g", "odeme_baslat_30g",
    "toplam_event_30g",
]
DAVRANIS_DIGER = ["gun_son_aktivite", "web_gun_payi"]
# tanimlar v1.2: web_gun_payi'nin düzeyi zamanla kayıyor (kanal karışımı / ölçüm değişikliği)
# → modele web_gun_payi_y (aktifler arasında kesim içi yüzdelik; davranışsızlar NaN) olarak girer.
# Geri dönüş: hazirla(..., web_payi_goreli=False) → ham web_gun_payi kullanılır.
WEB_PAYI_GORELI = True
DAVRANIS = DAVRANIS_SAYIM + DAVRANIS_DIGER


def endeksle(df: pd.DataFrame) -> pd.DataFrame:
    """Parasal kolonları kesim bazında bir fiyat endeksine böler (enflasyon).

    Endeks = o kesimdeki popülasyonun medyan birim fiyatı. Yalnızca kesim
    öncesi veriden hesaplanır, skorlama anında da bilinir; sızıntı yaratmaz.
    Yeni kolonlar: <kolon>_e  (1,0 = o kesimin tipik fiyat seviyesi)
    """
    df = df.copy()
    endeks = df.groupby("kesim")["ort_birim_fiyat_9ay"].transform("median")
    for c in PARASAL:
        df[f"{c}_e"] = df[c] / endeks
    return df


def goreli_davranis(df: pd.DataFrame, yontem: str = "yuzdelik_aktif") -> pd.DataFrame:
    """Davranış sayımlarını kesim içinde göreli hale getirir (<kolon>_y).

    yontem:
      "yuzdelik_aktif" (varsayılan, tanimlar v1.2): yüzdelik sıra yalnızca o kesimde
                 davranışı olanlar (davranis_var_30g) arasında, (0, 1]; davranışsızlar 0.
                 Aynı göreli ilgi, kapsam ne olursa olsun aynı değeri alır.
      "yuzdelik" (tanimlar v1.1): bütün kesim üzerinden yüzdelik. Davranışsızlar (0'lar)
                 sıraya girdiği için değerler kapsama bağlı kayar.
      "zskor"   : log1p sonrası kesim içi z-skoru (bütün kesim; aynı kapsam sorunu).
      "ham"     : dönüşüm yok (karşılaştırma için; kesimler arası düzey kaymasına açık).
    Her yöntem yalnızca o kesimin kendi popülasyonunu kullanır → sızıntı yok.
    Seçim test kesiminde DEĞİL, kayan pencere doğrulamasında yapılır.
    """
    if yontem not in {"yuzdelik_aktif", "yuzdelik", "zskor", "ham"}:
        raise ValueError(f"bilinmeyen yöntem: {yontem}")
    df = df.copy()
    if yontem == "yuzdelik_aktif":
        if "davranis_var_30g" not in df.columns:
            raise KeyError("yuzdelik_aktif için davranis_var_30g kolonu gerekli")
        aktif = df["davranis_var_30g"].fillna(False).astype(bool)
    for c in DAVRANIS_SAYIM:
        if c not in df.columns:
            continue
        if yontem == "yuzdelik_aktif":
            sira = df.loc[aktif, c].groupby(df.loc[aktif, "kesim"]).rank(pct=True, method="average")
            df[f"{c}_y"] = 0.0
            df.loc[aktif, f"{c}_y"] = sira
        elif yontem == "yuzdelik":
            df[f"{c}_y"] = df.groupby("kesim")[c].rank(pct=True, method="average")
        elif yontem == "zskor":
            x = np.log1p(df[c].astype(float))
            g = x.groupby(df["kesim"])
            df[f"{c}_y"] = (x - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        else:
            df[f"{c}_y"] = df[c].astype(float)
    return df


def goreli_web_payi(df: pd.DataFrame) -> pd.DataFrame:
    """web_gun_payi → web_gun_payi_y: davranışı olanlar arasında kesim içi yüzdelik (0, 1].

    Davranışsızlarda pay tanımsızdır → NaN kalır (LightGBM yerel, lojistikte medyan + gösterge).
    Not: payda 0 ve 1'de yığılma var; bağlı gruplar ortalama sıra alır.
    """
    df = df.copy()
    aktif = df["davranis_var_30g"].fillna(False).astype(bool)
    df["web_gun_payi_y"] = np.nan
    df.loc[aktif, "web_gun_payi_y"] = (
        df.loc[aktif, "web_gun_payi"].groupby(df.loc[aktif, "kesim"]).rank(pct=True, method="average"))
    return df


def birlestir(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """egitim_a ile egitim_b'yi (kesim, musteri_id) üzerinden birleştirir.

    Satır sayısı değişmemeli: egitim_b, egitim_a'nın anahtarlarından üretilir.
    """
    birlesik = a.merge(b, on=["kesim", "musteri_id"], how="left", validate="one_to_one")
    assert len(birlesik) == len(a), "birleştirme satır sayısını değiştirdi"
    return birlesik


def hazirla(a: pd.DataFrame, b: pd.DataFrame | None = None,
            davranis_yontemi: str = "yuzdelik_aktif",
            web_payi_goreli: bool = WEB_PAYI_GORELI) -> pd.DataFrame:
    """Model öncesi tek giriş noktası: birleştir → endeksle → davranışı göreli yap."""
    df = birlestir(a, b) if b is not None else a
    df = endeksle(df)
    if b is not None:
        df = goreli_davranis(df, yontem=davranis_yontemi)
        if web_payi_goreli:
            df = goreli_web_payi(df)
    return df


def model_a_ozellikleri(df: pd.DataFrame) -> list[str]:
    """Model A: sipariş + müşteri. Kimlik, hedef, analiz, ham TL, gereksiz ve davranış hariç."""
    davranis_tumu = BAYRAK + DAVRANIS + [f"{c}_y" for c in DAVRANIS_SAYIM] + ["web_gun_payi_y"]
    haric = set(KIMLIK + HEDEFLER + ANALIZ + PARASAL + GEREKSIZ + davranis_tumu)
    return [c for c in df.columns if c not in haric]


def model_a_bayrak_ozellikleri(df: pd.DataFrame) -> list[str]:
    """A + bayrak: A'ya 'son 30 günde (kurallı) davranış verisi var mı' eklenir."""
    return model_a_ozellikleri(df) + BAYRAK


def model_b_ozellikleri(df: pd.DataFrame) -> list[str]:
    """Model B: A + bayrak + göreli davranış sayımları + diğer davranış özellikleri.

    web_gun_payi_y varsa (hazirla varsayılanı) ham web_gun_payi yerine o girer.
    """
    diger = [("web_gun_payi_y" if c == "web_gun_payi" and "web_gun_payi_y" in df.columns else c)
             for c in DAVRANIS_DIGER]
    return model_a_ozellikleri(df) + BAYRAK + [f"{c}_y" for c in DAVRANIS_SAYIM] + diger
