"""Duman testi: modeller gerçekten eğitilip skorlanabiliyor mu? (gerçek veri gerekmez)

Neden var: Windows'ta bazı pyarrow sürümleri (ör. 17.x) eski bir msvcp140.dll
getirir; pyarrow, LightGBM'den önce yüklenirse LightGBM eğitimi
"access violation reading 0x0000000000000000" ile çöker. Diğer testler hiçbir
modeli eğitmediği için bu sorun ancak defterde görünüyordu. Bu test, veriyi
parquet üzerinden okuyarak (pyarrow önce yüklenir) her iki modeli eğitir.
"""
import numpy as np
import pandas as pd

from src.modeller import baz_kural, baz_rfm, egit, skorla


def _kucuk_veri(tmp_path, n=3000, tohum=0):
    rng = np.random.default_rng(tohum)
    df = pd.DataFrame({
        "kesim": pd.Timestamp("2025-01-01"),
        "siparis_12ay": rng.integers(1, 8, n),
        "gun_son_siparis": rng.integers(1, 365, n),
        "ciro_9ay_e": np.where(rng.random(n) < 0.1, np.nan, rng.gamma(2.0, 1.5, n)),
        "son_platform": rng.choice(["App", "Mobil", "Web"], n),
    })
    logit = 0.4 * df["siparis_12ay"] - 0.01 * df["gun_son_siparis"] - 1.0
    df["y_etiket"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    yol = tmp_path / "kucuk.parquet"
    df.to_parquet(yol, index=False)          # pyarrow'u LightGBM'den önce yükler
    return pd.read_parquet(yol)


def test_lightgbm_ve_lojistik_egitilir_ve_skorlar(tmp_path):
    df = _kucuk_veri(tmp_path)
    kolonlar = ["siparis_12ay", "gun_son_siparis", "ciro_9ay_e", "son_platform"]
    egitim, hedef = df.iloc[:2000], df.iloc[2000:]
    for model_adi in ["lightgbm", "lojistik"]:
        model = egit(model_adi, egitim, kolonlar)
        p = skorla(model_adi, model, hedef, kolonlar)
        assert p.shape == (len(hedef),)
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        # sinyal var: alıcıların ortalama skoru almayanlarınkinden yüksek
        y = hedef["y_etiket"].to_numpy()
        assert p[y].mean() > p[~y].mean()


def test_bazlar_skor_uretir(tmp_path):
    df = _kucuk_veri(tmp_path)
    for f in (baz_kural, baz_rfm):
        s = f(df)
        assert len(s) == len(df) and s.notna().all()
