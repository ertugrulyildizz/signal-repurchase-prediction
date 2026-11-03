"""Veri katmanı: proje tablolarını yükler.

İki kaynak var (ortam değişkeni SINYAL_VERI_KAYNAGI):
  "sentetik"  (varsayılan) data/<tablo>.parquet okunur. Önce üret:
              python -m sentetik.uret
  "bigquery"  Kendi BigQuery tablolarınız (sql/ şablonlarıyla üretilmiş).
              SINYAL_BQ_PROJE ve SINYAL_BQ_VERI_SETI tanımlanmalı; tablo yerel
              Parquet önbelleğine (data/) yazılır. Kimlik doğrulama için bir kez:
                  gcloud auth application-default login

data/ dizini .gitignore ile depodan hariç tutulur.
"""
from __future__ import annotations

import decimal
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KAYNAK = os.environ.get("SINYAL_VERI_KAYNAGI", "sentetik")
PROJE = os.environ.get("SINYAL_BQ_PROJE", "PROJE_ID")
VERI_SETI = os.environ.get("SINYAL_BQ_VERI_SETI", "sinyal")
KOK = Path(__file__).resolve().parents[1]
VERI_DIZINI = KOK / "data"


def _istemci():
    from google.cloud import bigquery

    return bigquery.Client(project=PROJE)


def _tipleri_duzelt(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery NUMERIC -> float, DATE (dbdate) -> datetime64."""
    for c in df.columns:
        if str(df[c].dtype) == "dbdate":
            df[c] = pd.to_datetime(df[c])
        elif df[c].dtype == object:
            ornek = df[c].dropna()
            if len(ornek) and isinstance(ornek.iloc[0], decimal.Decimal):
                df[c] = df[c].astype(float)
    return df


def _cache_stale(yol: Path, tablo: str) -> bool:
    """Önbellek BigQuery tablosundan eskiyse True. BigQuery'e erişilemezse False."""
    if not yol.exists():
        return True
    try:
        table = _istemci().get_table(f"{PROJE}.{VERI_SETI}.{tablo}")
        modified = getattr(table, "modified", None)
        if modified is None:
            return False
        local_mtime = datetime.fromtimestamp(yol.stat().st_mtime, tz=timezone.utc)
        return local_mtime < modified.astimezone(timezone.utc)
    except Exception:
        return False


def tablo_yukle(tablo: str, yenile: bool = False) -> pd.DataFrame:
    """<tablo> tablosunu döndürür.

    sentetik kip: data/<tablo>.parquet (yenile yok sayılır).
    bigquery kip: önbellekte varsa oradan; yenile=True ise BigQuery'den yeniden çeker.
    """
    VERI_DIZINI.mkdir(exist_ok=True)
    yol = VERI_DIZINI / f"{tablo}.parquet"
    if KAYNAK == "sentetik":
        if not yol.exists():
            raise FileNotFoundError(f"{yol} yok. Önce sentetik veriyi üretin: python -m sentetik.uret")
        return pd.read_parquet(yol)
    if yol.exists() and not yenile:
        try:
            df = pd.read_parquet(yol)
            if {"kesim", "musteri_id"}.issubset(df.columns) and not _cache_stale(yol, tablo):
                return df
        except Exception:
            pass
    sorgu = f"SELECT * FROM `{PROJE}.{VERI_SETI}.{tablo}`"
    df = _istemci().query(sorgu).to_dataframe(create_bqstorage_client=True)
    df = _tipleri_duzelt(df)
    df.to_parquet(yol, index=False)
    return df
