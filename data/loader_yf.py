# data/loader_yf.py
"""
Descarga OHLCV con proveedor 'yahoo' (yfinance) y fallback 'stooq',
con caché parquet en storage/cache_ohlcv.

Expone:
- download_ohlcv(ticker, period='1y', interval='1d', provider='auto')
- download_ohlcv_cached_resume(tickers, period='1y', interval='1d', provider='auto', cache_dir='storage/cache_ohlcv', resume=True)

Compatibilidad:
- Columnas devueltas en minúsculas: date, open, high, low, close, volume
- Índice reseteado (columna 'date' nativa)
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

# ====== Opcional: usa pandas-datareader si está instalado ======
try:
    from pandas_datareader.stooq import StooqDailyReader  # type: ignore
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

try:
    import yfinance as yf  # type: ignore
    _HAS_YF = True
except Exception:
    _HAS_YF = False


# Directorio de caché por defecto (lo usa tu healthcheck)
DEFAULT_CACHE_DIR = Path("storage") / "cache_ohlcv"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(cache_dir: Path, ticker: str, period: str, interval: str, provider: str) -> Path:
    t = ticker.upper().replace("/", "_").replace("\\", "_")
    return cache_dir / f"{t}__{provider}__{period}_{interval}.parquet"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out = out.reset_index()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    # yfinance -> 'index' puede ser datetime si no trajo 'Date' explícita
    if "date" not in out.columns:
        if "index" in out.columns:
            out = out.rename(columns={"index": "date"})
        else:
            # último recurso: intenta detectar la primer col datetime
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out = out.rename(columns={c: "date"})
                    break
    # Mantener sólo columnas clave si existen
    keep = ["date", "open", "high", "low", "close", "volume"]
    for k in keep:
        if k not in out.columns:
            out[k] = pd.NA
    out = out[keep]
    # Asegurar tipos
    out["date"] = pd.to_datetime(out["date"])
    for k in ["open", "high", "low", "close", "volume"]:
        out[k] = pd.to_numeric(out[k], errors="coerce")
    # Orden temporal
    out = out.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    # Elimina duplicados de fecha
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


# ----------------------------
# Yahoo (yfinance)
# ----------------------------
def _download_yahoo(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if not _HAS_YF:
        raise RuntimeError("yfinance no está instalado.")
    # Para reducir rate-limit usa 1:1 y sin threading (opcional)
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("Yahoo devolvió vacío")
            return _normalize_cols(df)
        except Exception as e:
            # 429 / red: backoff
            time.sleep(1.5 * (attempt + 1))
            last_err = e
    raise RuntimeError(f"Yahoo fallo para {ticker}: {last_err}")


# ----------------------------
# Stooq (CSV público o pandas-datareader)
# ----------------------------
def _to_stooq_symbol(ticker: str) -> str:
    """
    En Stooq, tickers USA se piden como 'TICKER.US'.
    """
    t = ticker.upper().strip()
    if not t.endswith(".US"):
        t = f"{t}.US"
    return t

def _download_stooq_csv(ticker: str) -> pd.DataFrame:
    """
    Descarga CSV diario desde Stooq.
    URL: https://stooq.com/q/d/l/?s={SYMBOL}&i=d
    """
    symbol = _to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    # Stooq trae columnas: Date, Open, High, Low, Close, Volume
    return _normalize_cols(df)

def _download_stooq_pdr(ticker: str) -> pd.DataFrame:
    if not _HAS_PDR:
        raise RuntimeError("pandas-datareader no instalado.")
    symbol = _to_stooq_symbol(ticker)
    df = StooqDailyReader(symbols=symbol).read()
    return _normalize_cols(df)

def _download_stooq(ticker: str) -> pd.DataFrame:
    # Preferir pandas-datareader si está disponible (más estable a veces)
    try:
        if _HAS_PDR:
            return _download_stooq_pdr(ticker)
    except Exception:
        pass
    # Fallback a CSV directo
    return _download_stooq_csv(ticker)


# ----------------------------
# API PÚBLICA (usada por healthcheck)
# ----------------------------
def download_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    provider: str = "auto",  # 'auto' | 'yahoo' | 'stooq'
    cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Descarga OHLCV (con caché parquet).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prov = provider.lower().strip()

    def _load_from_cache() -> Optional[pd.DataFrame]:
        p = _cache_path(cache_dir, ticker, period, interval, prov if prov != "auto" else "yahoo")
        if p.exists() and use_cache:
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        return None

    def _save_to_cache(df: pd.DataFrame, used_provider: str):
        p = _cache_path(cache_dir, ticker, period, interval, used_provider)
        try:
            df.to_parquet(p, index=False)
        except Exception:
            # pyarrow no instalado: fallback a CSV
            p_csv = p.with_suffix(".csv")
            df.to_csv(p_csv, index=False)

    # 1) Si hay caché, úsala
    cached = _load_from_cache()
    if cached is not None and not cached.empty:
        return _normalize_cols(cached)

    # 2) Elegir proveedor
    last_err = None
    if prov in ("yahoo", "auto"):
        try:
            df = _download_yahoo(ticker, period, interval)
            if df is not None and not df.empty:
                _save_to_cache(df, "yahoo")
                return df
        except Exception as e:
            last_err = e
            # si 'auto', caemos a stooq
            if prov == "yahoo":
                raise

    # 3) Fallback a Stooq
    df = _download_stooq(ticker)
    if df is None or df.empty:
        raise RuntimeError(f"No se obtuvo data para {ticker}. Último error Yahoo: {last_err}")
    _save_to_cache(df, "stooq")
    return df


def download_ohlcv_cached_resume(
    tickers: Iterable[str],
    period: str = "1y",
    interval: str = "1d",
    provider: str = "auto",
    cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR,
    resume: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Descarga múltiple con caché y reanudación.
    - resume=True: salta tickers con parquet existente.
    - provider='auto': intenta Yahoo y cae a Stooq si hay 429/errores.
    """
    cache_dir = Path(cache_dir)
    out: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        t = t.strip().upper()
        prov = provider
        try:
            # Si resume, intenta leer primero de caché (independiente del proveedor)
            if resume:
                # probar con ambos sufijos (yahoo/stooq) por si hay uno guardado
                for prov_try in ("yahoo", "stooq"):
                    p = _cache_path(cache_dir, t, period, interval, prov_try)
                    if p.exists():
                        try:
                            df = pd.read_parquet(p)
                            if df is not None and not df.empty:
                                out[t] = _normalize_cols(df)
                                prov = prov_try
                                raise StopIteration  # saltar a siguiente ticker
                        except Exception:
                            pass
            # si no hubo caché válida, descarga normal
            df = download_ohlcv(t, period=period, interval=interval, provider=provider, cache_dir=cache_dir, use_cache=False)
            out[t] = df
        except StopIteration:
            continue
        except Exception as e:
            # no cortar el proceso; registra vacío
            out[t] = pd.DataFrame()
            print(f"Failed to get ticker '{t}' reason: {e}")

    return out
