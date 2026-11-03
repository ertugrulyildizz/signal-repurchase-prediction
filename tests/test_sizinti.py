"""Sızıntı ve tutarlılık testleri. Bunlar geçmeden model kodu yazılmaz.

Çalıştırma (proje kökünde):  pytest -v
Ön koşul: data/egitim_a.parquet (python -m sentetik.uret ya da BigQuery kipi).
"""
import pandas as pd
import pytest

from src.ozellikler import HEDEFLER, KATEGORIK, endeksle, model_a_ozellikleri
from src.veri import tablo_yukle

TEST_KESIMI = pd.Timestamp("2025-10-01")
TEST2_KESIMI = pd.Timestamp("2026-01-01")
ETIKET_UFKU = pd.Timedelta(days=90)


@pytest.fixture(scope="module")
def df():
    return tablo_yukle("egitim_a")


def test_egitim_etiket_pencereleri_test_kesimine_tasmaz(df):
    egitim = df[df["kume"] == "egitim"]
    assert (egitim["kesim"] + ETIKET_UFKU).max() <= TEST_KESIMI


def test_tampon_kesimler_egitimde_yok(df):
    tampon = df[df["kume"] == "tampon"]
    assert ((tampon["kesim"] + ETIKET_UFKU) > TEST_KESIMI).all()


def test_tek_test_kesimi(df):
    assert set(df.loc[df["kume"] == "test", "kesim"].unique()) == {TEST_KESIMI}


def test_ikinci_test_kesimi_var_ve_tek(df):
    # Başarısızsa: önbellek eski olabilir → tablo_yukle("egitim_a", yenile=True)
    assert set(df.loc[df["kume"] == "test2", "kesim"].unique()) == {TEST2_KESIMI}


def test_test2_egitimi_test2ye_tasmaz(df):
    # test2 modeli test2 dışındaki tüm kesimlerle eğitilir; hiçbirinin etiket
    # penceresi 2026-01-01'i geçmemeli.
    digerleri = df[df["kume"] != "test2"]
    assert (digerleri["kesim"] + ETIKET_UFKU).max() <= TEST2_KESIMI


def test_son_siparis_kesimden_once_ve_pencerede(df):
    assert df["gun_son_siparis"].min() >= 1
    assert df["gun_son_siparis"].max() <= 365


def test_uyelik_suresi_negatif_degil(df):
    assert (df["uyelik_gun"] >= 0).all()


def test_musteri_kesim_tekil(df):
    assert not df.duplicated(["kesim", "musteri_id"]).any()


def test_hedefler_ozelliklere_karismaz(df):
    ozellikler = model_a_ozellikleri(endeksle(df))
    assert not any(c.startswith("y_") for c in ozellikler)
    assert not set(HEDEFLER) & set(ozellikler)


def test_ham_tl_kolonlari_ozelliklerde_yok(df):
    ozellikler = model_a_ozellikleri(endeksle(df))
    assert "ort_sepet_9ay" not in ozellikler and "ort_sepet_9ay_e" in ozellikler


def test_kategorik_kolonlar_mevcut(df):
    assert all(c in df.columns for c in KATEGORIK)


def test_gereksiz_ozellikler_modelde_yok(df):
    from src.ozellikler import GEREKSIZ
    ozellikler = model_a_ozellikleri(endeksle(df))
    assert not set(GEREKSIZ) & set(ozellikler)


def test_model_b_davranis_tablosu_ve_goreli_sira():
    from src.ozellikler import DAVRANIS_SAYIM, hazirla, model_b_ozellikleri
    a = tablo_yukle("egitim_a")
    b = tablo_yukle("egitim_b")
    df = hazirla(a, b)
    assert len(df) == len(a)
    oz = model_b_ozellikleri(df)
    # ham sayımlar modele girmez, yalnızca kesim içi yüzdelikleri girer
    assert not set(DAVRANIS_SAYIM) & set(oz)
    for c in DAVRANIS_SAYIM:
        assert df[f"{c}_y"].between(0, 1).all()
    assert not any(c.startswith("y_") for c in oz)
