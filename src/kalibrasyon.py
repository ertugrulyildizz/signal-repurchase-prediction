"""Kalibrasyon yardımcıları. Sıralamayı değil, olasılık düzeyini düzeltir.

Yöntem (sonuçlar görülmeden yazıldı):
  M1 izotonik: skor → olasılık eşlemesi, hedef kesimden ÖNCE etiketi kapanmış,
     zaman dışı (out-of-time) skorlarla öğrenilir.
  M2 düzey: logit kaydırma; ortalama olasılık = geçen yılın aynı ayının etiket oranı
     (skorlama anında bilinen bir sayı). Mevsimsel düzey kaybını giderir.
  Ana kalibrasyon = M1 + M2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression

EPS = 1e-4


def izotonik_uydur(skor, y) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    iso.fit(np.asarray(skor, dtype=float), np.asarray(y, dtype=float))
    return iso


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def duzey_kaydir(p, hedef_oran: float) -> np.ndarray:
    """Logit'e sabit ekleyerek ortalama olasılığı hedef_oran'a eşitler (sıralama değişmez)."""
    z = _logit(p)
    f = lambda d: (1 / (1 + np.exp(-(z + d)))).mean() - hedef_oran
    d = brentq(f, -10, 10)
    return 1 / (1 + np.exp(-(z + d)))


def guvenilirlik_tablosu(y, p, dilim: int = 10) -> pd.DataFrame:
    """Olasılık dilimlerinde ortalama tahmin ve gerçek oran (eşit büyüklükte dilimler)."""
    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    d["dilim"] = pd.qcut(d["p"].rank(method="first"), dilim, labels=False)
    return d.groupby("dilim").agg(tahmin=("p", "mean"), gercek=("y", "mean")).reset_index()


def kalibrasyon_ozeti(y, p, dilim: int = 10) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    t = guvenilirlik_tablosu(y, p, dilim)
    ref = y.mean() * (1 - y.mean())
    brier = float(((p - y) ** 2).mean())
    return {
        "ort_tahmin": float(p.mean()), "gercek_oran": float(y.mean()),
        "brier": brier, "brier_beceri": float(1 - brier / ref),
        "ece": float((t["tahmin"] - t["gercek"]).abs().mean()),   # dilimler eşit büyüklükte
    }
