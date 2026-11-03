"""Özellik dönüşümlerinin sentetik veriyle doğrulanması (gerçek veri gerekmez)."""
import numpy as np
import pandas as pd
import pytest

from src.degerlendirme import isaret_tutarli
from src.ozellikler import (DAVRANIS_SAYIM, goreli_davranis, goreli_web_payi,
                             model_a_ozellikleri, model_b_ozellikleri)


def _sentetik(kesim, n_aktif, n_pasif, tohum):
    """n_aktif aktif (sayımlar 1..n_aktif), n_pasif davranışsız (sayımlar 0)."""
    sayim = np.concatenate([np.arange(1, n_aktif + 1), np.zeros(n_pasif)])
    d = {"kesim": pd.Timestamp(kesim),
         "davranis_var_30g": [True] * n_aktif + [False] * n_pasif}
    for c in DAVRANIS_SAYIM:
        d[c] = sayim
    return pd.DataFrame(d)


def test_yuzdelik_aktif_kapsamdan_bagimsiz():
    """Aynı aktif dağılımı, farklı kapsam (%30 ve %60) → aktiflerin değerleri aynı."""
    dusuk = _sentetik("2024-10-01", n_aktif=30, n_pasif=70, tohum=0)
    yuksek = _sentetik("2025-03-01", n_aktif=30, n_pasif=20, tohum=1)
    df = goreli_davranis(pd.concat([dusuk, yuksek], ignore_index=True), "yuzdelik_aktif")
    c = f"{DAVRANIS_SAYIM[0]}_y"
    a1 = df.loc[(df["kesim"] == "2024-10-01") & df["davranis_var_30g"], c].to_numpy()
    a2 = df.loc[(df["kesim"] == "2025-03-01") & df["davranis_var_30g"], c].to_numpy()
    np.testing.assert_allclose(a1, a2)
    assert (df.loc[~df["davranis_var_30g"], c] == 0).all()
    assert (df.loc[df["davranis_var_30g"], c] > 0).all()
    assert df[c].between(0, 1).all()


def test_eski_yuzdelik_kapsama_bagli_kayiyor():
    """Belge niteliğinde: v1.1 yüzdeliğinde aynı aktif müşteri kapsama göre farklı değer alır."""
    dusuk = _sentetik("2024-10-01", n_aktif=30, n_pasif=70, tohum=0)
    yuksek = _sentetik("2025-03-01", n_aktif=30, n_pasif=20, tohum=1)
    df = goreli_davranis(pd.concat([dusuk, yuksek], ignore_index=True), "yuzdelik")
    c = f"{DAVRANIS_SAYIM[0]}_y"
    a1 = df.loc[(df["kesim"] == "2024-10-01") & df["davranis_var_30g"], c].to_numpy()
    a2 = df.loc[(df["kesim"] == "2025-03-01") & df["davranis_var_30g"], c].to_numpy()
    assert not np.allclose(a1, a2)


def test_yuzdelik_aktif_bayraksiz_hata_verir():
    df = _sentetik("2024-10-01", 5, 5, 0).drop(columns="davranis_var_30g")
    with pytest.raises(KeyError):
        goreli_davranis(df, "yuzdelik_aktif")


def test_web_payi_goreli_aktiflerde_ve_pasifte_nan():
    df = _sentetik("2024-10-01", 4, 3, 0)
    df["web_gun_payi"] = [0.0, 0.5, 0.5, 1.0, np.nan, np.nan, np.nan]
    out = goreli_web_payi(df)
    aktif = out["davranis_var_30g"]
    assert out.loc[~aktif, "web_gun_payi_y"].isna().all()
    np.testing.assert_allclose(out.loc[aktif, "web_gun_payi_y"], [0.25, 0.625, 0.625, 1.0])


def test_model_a_hicbir_davranis_kolonu_icermez():
    df = _sentetik("2024-10-01", 4, 3, 0)
    df["web_gun_payi"] = 0.5
    df["gun_son_aktivite"] = 3.0
    df["siparis_12ay"] = 1
    df = goreli_web_payi(goreli_davranis(df))
    a = model_a_ozellikleri(df)
    assert a == ["siparis_12ay"]
    b = model_b_ozellikleri(df)
    assert "web_gun_payi_y" in b and "web_gun_payi" not in b


def test_isaret_tutarli_3_kat():
    assert isaret_tutarli([0.012, 0.004, -0.003, 0.008])            # 3/4 pozitif
    assert not isaret_tutarli([0.012, -0.005, -0.001, 0.009])       # 2/4
    assert isaret_tutarli([0.012, 0.004, -0.003, 0.008], test_farki=0.011)
    assert not isaret_tutarli([0.012, 0.004, -0.003, 0.008], test_farki=-0.011)
