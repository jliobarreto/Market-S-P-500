# utils/cache_io.py
from __future__ import annotations
import os, json
from typing import Dict
import pandas as pd
from datetime import datetime, timedelta, timezone

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def path_for(cache_dir: str, ticker: str) -> str:
    _ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"{ticker.upper()}.parquet")

def manifest_path(cache_dir: str) -> str:
    _ensure_dir(cache_dir)
    return os.path.join(cache_dir, "manifest.json")

def load_manifest(cache_dir: str) -> Dict:
    p = manifest_path(cache_dir)
    if not os.path.exists(p): return {"created_at": datetime.now(timezone.utc).isoformat(), "tickers": {}}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(cache_dir: str, data: Dict):
    with open(manifest_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def mark_status(cache_dir: str, ticker: str, status: str, note: str = ""):
    m = load_manifest(cache_dir)
    t = ticker.upper()
    m["tickers"][t] = {"status": status, "note": note, "ts": datetime.now(timezone.utc).isoformat()}
    save_manifest(cache_dir, m)

def is_fresh(cache_dir: str, ticker: str, freshness_days: int) -> bool:
    p = path_for(cache_dir, ticker)
    if not os.path.exists(p): return False
    mtime = datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) < timedelta(days=freshness_days)

def save_df(cache_dir: str, ticker: str, df: pd.DataFrame):
    p = path_for(cache_dir, ticker)
    df.to_parquet(p, index=False)

def load_df(cache_dir: str, ticker: str) -> pd.DataFrame:
    p = path_for(cache_dir, ticker)
    if not os.path.exists(p): return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()
