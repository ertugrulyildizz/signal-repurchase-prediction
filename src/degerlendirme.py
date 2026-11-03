"""Metrikler, kayan pencere katları ve bootstrap karşılaştırması.

Tanımlar: docs/preregistration.md v1.1 (kilitli).
  Birincil: PR-AUC · İş metriği: capture@10 (en yüksek skorlu %10'da
  yakalanan alıcı payı) ve kesinlik@10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ETIKET_UFKU_GUN = 90
DOGRULAMA_KESIMLERI = ("2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01")
# tanimlar v1.2, madde 2: "kayan pencerelerin en az 3'te 2'si" → 4 katta en az 3 kat,
# işaret test farkının işaretiyle aynı. Geri dönüş: yeni tanım sürümü + bu sabit.
ISARET_MIN_KAT = 3


def isaret_tutarli(kat_farklari, test_farki=None, min_kat: int = ISARET_MIN_KAT) -> bool:
    """Kat farklarının en az min_kat'ı aynı işarette mi?

    test_farki verilirse: test farkıyla aynı işaretteki kat sayısı ≥ min_kat (03_test).
    Verilmezse (02, geliştirme): baskın işaretteki kat sayısı ≥ min_kat.
    """
    f = np.asarray(kat_farklari, dtype=float)
    if test_farki is None:
        return int(max((f > 0).sum(), (f < 0).sum())) >= min_kat
    return int((np.sign(f) == np.sign(test_farki)).sum()) >= min_kat and test_farki != 0


def _ust_k_indeks(skor: np.ndarray, k: float) -> np.ndarray:
    n = max(1, int(round(len(skor) * k)))
    return np.argsort(-skor, kind="mergesort")[:n]


def capture_at_k(y, skor, k: float = 0.10) -> float:
    """Tüm alıcıların ne kadarı en yüksek skorlu %k dilimde?"""
    y = np.asarray(y).astype(int)
    skor = np.asarray(skor, dtype=float)
    toplam = y.sum()
    if toplam == 0:
        return np.nan
    return float(y[_ust_k_indeks(skor, k)].sum() / toplam)


def kesinlik_at_k(y, skor, k: float = 0.10) -> float:
    """En yüksek skorlu %k dilimin ne kadarı alıcı?"""
    y = np.asarray(y).astype(int)
    skor = np.asarray(skor, dtype=float)
    return float(y[_ust_k_indeks(skor, k)].mean())


def metrikler(y, skor, olasilik: bool = True) -> dict:
    y = np.asarray(y).astype(int)
    skor = np.asarray(skor, dtype=float)
    d = {
        "n": int(len(y)),
        "pozitif_oran": float(y.mean()),
        "pr_auc": float(average_precision_score(y, skor)),
        "roc_auc": float(roc_auc_score(y, skor)) if 0 < y.sum() < len(y) else np.nan,  # yalnızca rapor
        "capture@10": capture_at_k(y, skor),
        "kesinlik@10": kesinlik_at_k(y, skor),
    }
    if olasilik:
        d["brier"] = float(brier_score_loss(y, np.clip(skor, 0, 1)))
    return d


def alt_kumeler(d: pd.DataFrame) -> dict:
    """Raporlanan alt kümeler (tanimlar v1.1 madde 6, 7, 9). Tek doğruluk kaynağı (02 ve 03).

    tum · karar (son siparişi > 30 gün) · kapsam / davranissiz (son 30 günde kurallı davranış var/yok)
    · tek_siparis / tekrar (12 ayda 1 / ≥ 2 sipariş) · uzak_180 (son siparişi > 180 gün).
    """
    davranis = d["davranis_var_30g"].fillna(False).astype(bool).values
    return {
        "tum":         np.ones(len(d), dtype=bool),
        "karar":       (d["gun_son_siparis"] > 30).values,
        "kapsam":      davranis,
        "davranissiz": ~davranis,
        "tek_siparis": (d["siparis_12ay"] == 1).values,
        "tekrar":      (d["siparis_12ay"] >= 2).values,
        "uzak_180":    (d["gun_son_siparis"] > 180).values,
    }


def kayan_katlar(df: pd.DataFrame, dogrulama_kesimleri=DOGRULAMA_KESIMLERI):
    """Zaman bazlı doğrulama katları — yalnızca 'egitim' kümesi kullanılır.

    Her doğrulama kesimi v için eğitim = etiket penceresi v'den önce kapanan
    eğitim kesimleri (kesim + 90 gün ≤ v). Test ve test2 hiç kullanılmaz.
    Döndürür: (v, eğitim indeksleri, doğrulama indeksleri)
    """
    egitim = df[df["kume"] == "egitim"]
    for v in map(pd.Timestamp, dogrulama_kesimleri):
        egt = egitim.index[(egitim["kesim"] + pd.Timedelta(days=ETIKET_UFKU_GUN)) <= v]
        val = egitim.index[egitim["kesim"] == v]
        if len(egt) == 0 or len(val) == 0:
            raise ValueError(f"{v.date()} için kat oluşturulamadı")
        yield v, egt, val


def bootstrap_fark(y, skor_a, skor_b, metrik=capture_at_k, n: int = 1000, tohum: int = 42) -> dict:
    """metrik(B) − metrik(A) farkı ve müşteri düzeyinde %95 bootstrap güven aralığı."""
    rng = np.random.default_rng(tohum)
    y = np.asarray(y).astype(int)
    a = np.asarray(skor_a, dtype=float)
    b = np.asarray(skor_b, dtype=float)
    m = len(y)
    farklar = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        farklar[i] = metrik(y[idx], b[idx]) - metrik(y[idx], a[idx])
    return {
        "fark": float(metrik(y, b) - metrik(y, a)),
        "alt_%2.5": float(np.percentile(farklar, 2.5)),
        "ust_%97.5": float(np.percentile(farklar, 97.5)),
    }


# ----------------------------------------------------------------- 03: test planı
# tanimlar v1.1 "Kesimler": test modeli eğitim kesimleriyle; test2 modeli 2024-10 → 2025-10
# kesimleriyle (eğitim + tampon + test) eğitilir. Her iki durumda etiket penceresi hedef
# kesimden önce kapanmalı (kesim + 90 ≤ hedef).
TEST_PLANI = {
    "test":  ("2025-10-01", ("egitim",)),
    "test2": ("2026-01-01", ("egitim", "tampon", "test")),
}


def hedef_egitim_indeksleri(df: pd.DataFrame, hedef, kumeler) -> pd.Index:
    """Hedef kesim için eğitim satırları: izinli kümelerde ve etiket penceresi hedeften önce kapanan."""
    hedef = pd.Timestamp(hedef)
    m = df["kume"].isin(kumeler) & ((df["kesim"] + pd.Timedelta(days=ETIKET_UFKU_GUN)) <= hedef)
    return df.index[m]


def bootstrap_fark_farki(y, skor_a, skor_b, maske1, maske2, metrik=capture_at_k,
                         n: int = 1000, tohum: int = 42) -> dict:
    """İkincil hipotez: [metrik(B) − metrik(A)] alt küme 1'de eksi aynısı alt küme 2'de.

    Müşteriler bütün kesimden birlikte yeniden örneklenir (iki alt küme aynı örneklemden).
    """
    rng = np.random.default_rng(tohum)
    y = np.asarray(y).astype(int)
    a = np.asarray(skor_a, dtype=float)
    b = np.asarray(skor_b, dtype=float)
    m1 = np.asarray(maske1, dtype=bool)
    m2 = np.asarray(maske2, dtype=bool)

    def _farklar(idx):
        yy, aa, bb, k1, k2 = y[idx], a[idx], b[idx], m1[idx], m2[idx]
        f1 = metrik(yy[k1], bb[k1]) - metrik(yy[k1], aa[k1])
        f2 = metrik(yy[k2], bb[k2]) - metrik(yy[k2], aa[k2])
        return f1, f2

    f1, f2 = _farklar(np.arange(len(y)))
    dagilim = np.empty(n)
    for i in range(n):
        g1, g2 = _farklar(rng.integers(0, len(y), len(y)))
        dagilim[i] = g1 - g2
    return {
        "fark_alt1": float(f1), "fark_alt2": float(f2), "fark_farki": float(f1 - f2),
        "alt_%2.5": float(np.percentile(dagilim, 2.5)),
        "ust_%97.5": float(np.percentile(dagilim, 97.5)),
    }

