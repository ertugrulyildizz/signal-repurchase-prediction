"""Sinyal | scripts/07_ara_okuma.py  (tanimlar v1.5 ara okuma)

06 defterinin 3. bölümünün tek başına çalışan hali. 06'yı baştan çalıştırma:
ileri test kilidi ve canlı tablo (WRITE_EMPTY) yeniden yazılmaya çalışılır.

Ön koşul: BigQuery'de sql/24_ara_okuma.sql çalıştırılmış olmalı (ilk_alim tablosu).
Çalıştırma (proje kökünde, venv açıkken):  python scripts/07_ara_okuma.py
"""
import hashlib, json, sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))

import pandas as pd
from src.veri import tablo_yukle
from src.degerlendirme import bootstrap_fark, capture_at_k

SONUC, VERI = KOK / "docs" / "sonuclar", KOK / "data"
TOHUM, BOOTSTRAP_N = 42, 1000
KIYAS, CANLI = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-10-01")
pd.set_option("display.float_format", "{:.4f}".format)

# 0) Mühür kontrolü: canlı skorlar 27 Ekim'den beri değişmedi mi?
cs = pd.read_parquet(VERI / "06_canli_skor_20261001.parquet")
metin = "\n".join(f"{r.musteri_id}|{r.skor_B:.8f}" for r in cs.sort_values("musteri_id").itertuples())
muhur = hashlib.sha256(metin.encode("utf-8")).hexdigest()
beklenen = json.loads((SONUC / "06_canli_muhur.json").read_text(encoding="utf-8"))["sha256"]
print("MÜHÜR:", "EŞLEŞİYOR" if muhur == beklenen else "EŞLEŞMİYOR — DUR", muhur[:12], "…")
if muhur != beklenen:
    sys.exit(1)

# 1) İlk alım tarihleri (sql/24)
ilk = tablo_yukle("ilk_alim", yenile=True)
ilk["ilk_alim_tarihi"] = pd.to_datetime(ilk["ilk_alim_tarihi"])
ilk["kesim"] = pd.to_datetime(ilk["kesim"])
st = pd.read_parquet(VERI / "03_skorlar_test.parquet")

ao, YB = [], {}
for ad, skorlar, kesim, kolA, kolB in [("kıyas: Ekim 2025 (test)", st, KIYAS, "skor_lightgbm_A", "skor_lightgbm_B"),
                                      ("canlı: Ekim 2026", cs, CANLI, "skor_A", "skor_B")]:
    x = skorlar.merge(ilk[ilk["kesim"] == kesim][["musteri_id", "ilk_alim_tarihi"]], on="musteri_id", how="left", validate="one_to_one")
    y_ay = (x["ilk_alim_tarihi"] < kesim + pd.DateOffset(months=1)).astype(int).to_numpy()
    YB[ad] = (y_ay, x[kolA].to_numpy(), x[kolB].to_numpy())
    ao.append({"donem": ad, "musteri": len(x), "ilk_ay_alim_orani": y_ay.mean(),
               "capture@10_B": capture_at_k(y_ay, x[kolB]), "capture@10_A": capture_at_k(y_ay, x[kolA]),
               "son_veri": x["ilk_alim_tarihi"].max()})
AO = pd.DataFrame(ao); AO["B-A_puan"] = (AO["capture@10_B"] - AO["capture@10_A"]) * 100
AO.to_csv(SONUC / "06_ara_okuma.csv", index=False)
print(AO.to_string(index=False))

# 2) v1.5 ölçütleri (ara okumadan önce yazıldı)
kiyas, canli = AO.iloc[0], AO.iloc[1]
y_c, a_c, b_c = YB["canlı: Ekim 2026"]
bs = bootstrap_fark(y_c, a_c, b_c, metrik=capture_at_k, n=BOOTSTRAP_N, tohum=TOHUM)
O1 = canli["capture@10_B"] >= kiyas["capture@10_B"] - 0.03
O2 = canli["capture@10_B"] - canli["capture@10_A"] > 0
KA = pd.DataFrame([
    ("Ölçüt 1: B_canlı ≥ B_kıyas − 3 puan", f"{canli['capture@10_B']:.3f} vs {kiyas['capture@10_B']:.3f}", bool(O1)),
    ("Ölçüt 2: B − A > 0 (canlı)", f"{(canli['capture@10_B'] - canli['capture@10_A']) * 100:+.2f} puan "
                                   f"[{bs['alt_%2.5'] * 100:+.2f}, {bs['ust_%97.5'] * 100:+.2f}]", bool(O2)),
], columns=["olcut", "deger", "saglandi"])
KA.to_csv(SONUC / "06_ara_okuma_karar.csv", index=False)
print()
print(KA.to_string(index=False))
print("\nARA OKUMA:", "DESTEKLER" if (O1 and O2) else "DESTEKLEMEZ", "(bilgi amaçlı; kesin okuma 30 Aralık sonrası)")
