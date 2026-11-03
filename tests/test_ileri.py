"""v1.3: ileri test ve canlı kesim tablolarının sızdırmazlığı (tablo önbellekte yoksa atlanır)."""
from pathlib import Path

import pandas as pd
import pytest

VERI = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ileri():
    yol = VERI / "egitim_a_ileri.parquet"
    if not yol.exists():
        pytest.skip("data/egitim_a_ileri.parquet yok (python -m sentetik.uret)")
    return pd.read_parquet(yol)


def test_ileri_kesimler_ve_kumeler(ileri):
    beklenen = {"2025-11": "ek_egitim", "2025-12": "ek_egitim", "2026-02": "ek_egitim", "2026-03": "ek_egitim",
                "2026-04": "ileri", "2026-05": "ileri", "2026-06": "ileri", "2026-07": "ileri", "2026-10": "canli"}
    g = ileri.groupby(ileri["kesim"].dt.strftime("%Y-%m"))["kume"].unique()
    assert {k: list(v) for k, v in g.items()} == {k: [v] for k, v in beklenen.items()}


def test_canli_kesimde_hedef_yok(ileri):
    c = ileri[ileri["kume"] == "canli"]
    assert len(c) > 0
    assert c[["y_etiket", "y_siparis_90g", "y_ciro_90g"]].isna().all().all()


def test_ileri_etiketleri_bugunden_once_kapanir(ileri):
    k = ileri.loc[ileri["kume"].isin(["ek_egitim", "ileri"]), "kesim"]
    assert (k + pd.Timedelta(days=90)).max() <= pd.Timestamp("2026-10-01")


def test_eski_tabloyla_cakisma_yok(ileri):
    eski = VERI / "egitim_a.parquet"
    if eski.exists():
        assert not set(pd.read_parquet(eski, columns=["kesim"])["kesim"]) & set(ileri["kesim"])


def test_davranis_verisi_kesimden_bir_gun_once_biter():
    """Canlı dışı her kesimde GA4 verisi kesimden bir gün öncesine kadar gelmeli (eksik gün = bozuk 7/30 günlük özellik).
    Canlı kesim için aynı koruma 06 defterinde (GA4 tazeliği değişebilir)."""
    yol = VERI / "egitim_b_ileri.parquet"
    if not yol.exists():
        pytest.skip("egitim_b_ileri henüz önbellekte değil")
    b = pd.read_parquet(yol)
    a = pd.read_parquet(VERI / "egitim_a_ileri.parquet", columns=["kesim", "musteri_id", "kume"])
    b = b.merge(a, on=["kesim", "musteri_id"])
    akt = b[b["davranis_var_30g"].astype(bool) & (b["kume"] != "canli")]
    assert (akt.groupby("kesim")["gun_son_aktivite"].min() == 1).all()
