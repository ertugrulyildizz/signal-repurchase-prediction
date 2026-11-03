"""Metrik ve kat fonksiyonlarının sentetik veriyle doğrulanması."""
import numpy as np
import pandas as pd

from src.degerlendirme import (ETIKET_UFKU_GUN, TEST_PLANI, alt_kumeler, bootstrap_fark,
                               bootstrap_fark_farki, capture_at_k, hedef_egitim_indeksleri,
                               kayan_katlar, kesinlik_at_k, metrikler)
from src.veri import tablo_yukle


def test_capture_mukemmel_ve_rastgele():
    y = np.array([1] * 10 + [0] * 90)
    mukemmel = np.arange(100, 0, -1)          # alıcılar en üstte
    assert capture_at_k(y, mukemmel, 0.10) == 1.0
    assert kesinlik_at_k(y, mukemmel, 0.10) == 1.0
    ters = -mukemmel                           # alıcılar en altta
    assert capture_at_k(y, ters, 0.10) == 0.0


def test_bootstrap_ayni_skorda_fark_sifir():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.random(500)
    r = bootstrap_fark(y, s, s, n=50)
    assert r["fark"] == 0 and r["alt_%2.5"] == 0 and r["ust_%97.5"] == 0


def test_kayan_katlar_sizdirmaz():
    df = tablo_yukle("egitim_a")
    for v, egt, val in kayan_katlar(df):
        assert (df.loc[egt, "kesim"] + pd.Timedelta(days=ETIKET_UFKU_GUN)).max() <= v
        assert (df.loc[val, "kesim"] == v).all()
        assert set(df.loc[egt, "kume"]) == {"egitim"}
        assert set(df.loc[val, "kume"]) == {"egitim"}


def test_alt_kumeler_tumleyen_ciftler_ve_roc():
    d = pd.DataFrame({"gun_son_siparis": [5, 40, 200, 365], "siparis_12ay": [3, 1, 1, 2],
                      "davranis_var_30g": [True, False, True, False]})
    m = alt_kumeler(d)
    assert set(m) == {"tum", "karar", "kapsam", "davranissiz", "tek_siparis", "tekrar", "uzak_180"}
    assert ((m["kapsam"] ^ m["davranissiz"]) & (m["tek_siparis"] ^ m["tekrar"])).all()
    assert m["uzak_180"].sum() == 2 and m["karar"].sum() == 3
    r = metrikler([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert r["roc_auc"] == 1.0


def test_test_plani_egitim_kesimleri():
    kesimler = pd.date_range("2024-10-01", "2026-01-01", freq="MS")
    kume = {k: ("egitim" if k <= pd.Timestamp("2025-07-01") else "test" if k == pd.Timestamp("2025-10-01")
                else "test2" if k == pd.Timestamp("2026-01-01") else "tampon") for k in kesimler}
    df = pd.DataFrame({"kesim": kesimler, "kume": [kume[k] for k in kesimler]})
    for ad, (hedef, kumeler) in TEST_PLANI.items():
        e = df.loc[hedef_egitim_indeksleri(df, hedef, kumeler)]
        assert (e["kesim"] + pd.Timedelta(days=ETIKET_UFKU_GUN)).max() <= pd.Timestamp(hedef)
        assert not (e["kume"] == ad).any()
    e1 = df.loc[hedef_egitim_indeksleri(df, *TEST_PLANI["test"])]
    assert list(e1["kesim"].dt.strftime("%Y-%m")) == [k.strftime("%Y-%m") for k in kesimler[:10]]
    e2 = df.loc[hedef_egitim_indeksleri(df, *TEST_PLANI["test2"])]
    assert e2["kesim"].max() == pd.Timestamp("2025-10-01") and len(e2) == 13


def test_bootstrap_fark_farki_ayni_skorda_sifir():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400); s = rng.random(400); m = rng.random(400) < 0.5
    r = bootstrap_fark_farki(y, s, s, m, ~m, n=20)
    assert r["fark_farki"] == 0 and r["alt_%2.5"] == 0 and r["ust_%97.5"] == 0
