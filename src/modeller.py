"""Bazlar ve modeller. Tüm skorlar 'yüksek = alma olasılığı yüksek' yönündedir."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ozellikler import KATEGORIK

PLATFORMLAR = ["App", "Mobil", "Web"]


# ----------------------------------------------------------------- bazlar
def baz_kural(df: pd.DataFrame) -> pd.Series:
    """'Son 90 günde aldıysa alır' kuralı; kural içinde eşitlik yakınlıkla bozulur.

    Not: kural popülasyonun %10'undan fazlasını işaretlediğinde capture@10 için bu,
    saf yakınlık sıralamasıyla aynı sonucu verir.
    """
    g = df["gun_son_siparis"].astype(float)
    return (g <= 90).astype(float) + (1 - g / 1000.0) * 0.5


def baz_rfm(df: pd.DataFrame) -> pd.Series:
    """RFM puanı: kesim içi yüzdelik R (yakın = yüksek) + F + M toplamı."""
    k = df["kesim"]
    r = df["gun_son_siparis"].astype(float).groupby(k).rank(pct=True, ascending=False)
    f = df["siparis_12ay"].astype(float).groupby(k).rank(pct=True)
    m = df["ciro_9ay_e"].astype(float).fillna(0).groupby(k).rank(pct=True)
    return r + f + m


def baz_bgnbd(df: pd.DataFrame, ufuk_gun: int = 90, ceza: float = 0.001) -> pd.Series:
    """BG/NBD (lifetimes): önümüzdeki 90 günde beklenen sipariş sayısı.

    Her kesim kendi popülasyonunun YALNIZCA kesim öncesi verisiyle uydurulur
    (etiket kullanılmaz) → sızıntı yok. Birimler gün; pencere 12 ay.
    """
    from lifetimes import BetaGeoFitter

    skor = pd.Series(np.nan, index=df.index, dtype=float)
    for _, p in df.groupby("kesim"):
        frekans = (p["siparis_12ay"].astype(float) - 1).clip(lower=0)
        T = p["gun_ilk_siparis_12ay"].astype(float)
        yakinlik = (p["gun_ilk_siparis_12ay"] - p["gun_son_siparis"]).astype(float)
        # Uydurma bazen sayısal olarak ıraksar (NaN); ceza katsayısını büyüterek yeniden dene.
        for c in (ceza, ceza * 10, ceza * 100, ceza * 1000):
            bgf = BetaGeoFitter(penalizer_coef=c)
            bgf.fit(frekans, yakinlik, T)
            tahmin = bgf.conditional_expected_number_of_purchases_up_to_time(ufuk_gun, frekans, yakinlik, T)
            if np.isfinite(np.asarray(tahmin, dtype=float)).all():
                break
        skor.loc[p.index] = np.asarray(tahmin, dtype=float)
    if skor.isna().any():
        raise ValueError("BG/NBD uydurulamadı (NaN skor)")
    return skor


# ----------------------------------------------------------------- matris
def matris(df: pd.DataFrame, kolonlar: list[str], lgbm: bool = False) -> pd.DataFrame:
    """Özellik matrisi: sayısallar float64 (Int64/bool → float), kategorikler sabit kategorili."""
    X = pd.DataFrame(index=df.index)
    for c in kolonlar:
        if c in KATEGORIK:
            if lgbm:
                X[c] = pd.Categorical(df[c], categories=PLATFORMLAR)
            else:
                X[c] = df[c].astype(object).where(df[c].notna(), "yok")
        else:
            X[c] = df[c].astype("float64")
    return X


# ----------------------------------------------------------------- modeller
def lojistik(kolonlar: list[str], C: float = 1.0):
    """Cezalı lojistik regresyon: medyan doldurma + eksiklik göstergesi + ölçekleme + one-hot."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    sayisal = [c for c in kolonlar if c not in KATEGORIK]
    kategorik = [c for c in kolonlar if c in KATEGORIK]
    on_isleme = ColumnTransformer([
        ("sayisal", Pipeline([
            ("doldur", SimpleImputer(strategy="median", add_indicator=True)),
            ("olcek", StandardScaler()),
        ]), sayisal),
        ("kategorik", OneHotEncoder(handle_unknown="ignore"), kategorik),
    ])
    return Pipeline([("on_isleme", on_isleme),
                     ("model", LogisticRegression(C=C, max_iter=2000))])


def lightgbm(**parametreler):
    """Ana model. Doğrulama kümesi erken durdurmada KULLANILMAZ (sabit ağaç sayısı)."""
    from lightgbm import LGBMClassifier

    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=42, n_jobs=-1, verbose=-1)
    p.update(parametreler)
    return LGBMClassifier(**p)


def egit(model_adi: str, egitim: pd.DataFrame, kolonlar: list[str]):
    """Modeli verilen eğitim satırlarında eğitir ve döndürür (03: SHAP/kalibrasyon için saklanır)."""
    lgbm = model_adi == "lightgbm"
    model = lightgbm() if lgbm else lojistik(kolonlar)
    model.fit(matris(egitim, kolonlar, lgbm), egitim["y_etiket"].astype(int))
    return model


def skorla(model_adi: str, model, d: pd.DataFrame, kolonlar: list[str]) -> np.ndarray:
    """Eğitilmiş modelle olasılık skoru."""
    return model.predict_proba(matris(d, kolonlar, model_adi == "lightgbm"))[:, 1]


def egit_ve_skorla(model_adi: str, egitim: pd.DataFrame, dogrulama: pd.DataFrame,
                   kolonlar: list[str]) -> np.ndarray:
    """Modeli eğitim kesimlerinde eğitir, doğrulama kesiminde olasılık döndürür."""
    return skorla(model_adi, egit(model_adi, egitim, kolonlar), dogrulama, kolonlar)
