"""Kalibrasyon yardımcıları — sentetik veri."""
import numpy as np

from src.kalibrasyon import duzey_kaydir, izotonik_uydur, kalibrasyon_ozeti


def test_duzey_kaydir_ortalamayi_tutturur_siralamayi_korur():
    rng = np.random.default_rng(0)
    p = rng.beta(2, 10, 5000)
    q = duzey_kaydir(p, 0.25)
    assert abs(q.mean() - 0.25) < 1e-6
    assert (np.argsort(p, kind="mergesort") == np.argsort(q, kind="mergesort")).all()


def test_izotonik_kalibrasyonu_iyilestirir():
    rng = np.random.default_rng(1)
    gercek = rng.uniform(0.02, 0.6, 20000)
    y = rng.random(20000) < gercek
    skor = gercek ** 3          # sıralama doğru, düzey yanlış
    iso = izotonik_uydur(skor[:10000], y[:10000])
    once = kalibrasyon_ozeti(y[10000:], skor[10000:])
    sonra = kalibrasyon_ozeti(y[10000:], iso.predict(skor[10000:]))
    assert sonra["brier"] < once["brier"] and sonra["ece"] < once["ece"]
