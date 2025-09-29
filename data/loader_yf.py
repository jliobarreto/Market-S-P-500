# data/loader_yf.py  (descarga robusta + fallback Stooq + caché y reanudación)
from __future__ import annotations
import time
from typing import List
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf
from pandas_datareader import data as pdr  # Fallback: Stooq

# === NUEVO: caché y detección de rate-limit ===
from utils.cache_io import (
    is_fresh, save_df, load_df, mark_status, load_manifest, save_manifest
)
from utils.pause import looks_like_rate_limit
from config.config import load_config

CFG = load_config()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Fallbacks para índices si Yahoo falla
INDEX_FALLBACKS = {
    "^VIX": ["VIXY", "VIXM"],
    "SPY":  ["VOO", "IVV"],
}

def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return s

SESSION = _make_session()

# -------------------------------------------------
# Utilidades base (Yahoo + fallbacks + Stooq)
# -------------------------------------------------

def _sanitize_universe(tickers: List[str]) -> List[str]:
    bad = {"", None, "NAN", "NULL"}
    t = [str(x).strip().upper() for x in tickers if x and str(x).upper() not in bad]
    t = list(dict.fromkeys(t))
    # Algunos delisted conocidos (ajusta a gusto)
    delisted_guess = {"SNDK", "COR", "GTM", "INGM", "RAL", "VIK"}
    return [x for x in t if x not in delisted_guess]

def _single_history_yf(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    try:
        tk = yf.Ticker(ticker, session=SESSION)
        df = tk.history(
            start=start, end=end, interval=interval,
            auto_adjust=False, actions=False, prepost=False
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns=str.capitalize)
            df["Ticker"] = ticker
            df = df.reset_index().rename(columns={"Date": "Date"})
            return df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]
    except Exception:
        pass
    return pd.DataFrame()

def _yf_with_fallbacks(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = _single_history_yf(ticker, start, end, interval)
    if not df.empty:
        return df
    for alt in INDEX_FALLBACKS.get(ticker, []):
        df = _single_history_yf(alt, start, end, interval)
        if not df.empty:
            df["Ticker"] = ticker  # conserva etiqueta original
            return df
    return pd.DataFrame()

def _stooq_symbol(ticker: str) -> str:
    # Stooq usa sufijo .US para acciones/ETFs de USA
    return f"{ticker}.US"

def _single_history_stooq(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    try:
        sym = _stooq_symbol(ticker)
        df = pdr.DataReader(sym, "stooq", start=start_ts, end=end_ts)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns=str.capitalize).sort_index()
            if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
                return pd.DataFrame()
            if "Volume" not in df.columns:
                df["Volume"] = 0
            out = df.reset_index().rename(columns={"Date": "Date"})
            out["Ticker"] = ticker
            return out[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]
    except Exception:
        pass
    return pd.DataFrame()

# -------------------------------------------------
# Descarga en bloque (Yahoo + fallback Stooq)
# -------------------------------------------------

def download_ohlcv(
    tickers: List[str],
    lookback_years: int = 3,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Descarga OHLCV para múltiples tickers (Yahoo como principal, Stooq como último recurso).
    Devuelve columnas: ['Date','Open','High','Low','Close','Volume','Ticker']
    """
    tickers = _sanitize_universe(tickers)
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.DateOffset(years=lookback_years)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    all_frames: List[pd.DataFrame] = []
    block_size = 20

    for i in range(0, len(tickers), block_size):
        block = tickers[i : i + block_size]
        try:
            df = yf.download(
                tickers=block,
                start=s_str, end=e_str, interval=interval,
                group_by="ticker", auto_adjust=False, threads=False,
                progress=False, prepost=False, session=SESSION
            )
        except Exception:
            df = pd.DataFrame()

        if isinstance(df, pd.DataFrame) and not df.empty:
            # Caso 1 ticker: columnas planas
            if set(["Open","High","Low","Close","Adj Close","Volume"]).issubset(df.columns):
                one = df.copy().rename(columns={"Adj Close":"AdjClose"})
                one["Ticker"] = block[0]
                one = one.reset_index().rename(columns={"Date":"Date"})
                one = one[["Date","Open","High","Low","Close","Volume","Ticker"]]
                all_frames.append(one)
            else:
                # Múltiples: columnas multinivel
                for tkr in block:
                    if hasattr(df.columns, "get_level_values") and tkr in df.columns.get_level_values(0):
                        sub = df[tkr].copy()
                        if sub.empty:
                            continue
                        sub = sub.rename(columns={"Adj Close":"AdjClose"})
                        sub["Ticker"] = tkr
                        sub = sub.reset_index().rename(columns={"Date":"Date"})
                        sub = sub[["Date","Open","High","Low","Close","Volume","Ticker"]]
                        all_frames.append(sub)

        # Completar faltantes con fallback (Yahoo por ticker, luego Stooq)
        got = set(f["Ticker"].iloc[0] for f in all_frames if not f.empty)
        missing = [t for t in block if t not in got]
        for m in missing:
            one = _yf_with_fallbacks(m, s_str, e_str, interval)
            if one.empty:
                one = _single_history_stooq(m, start, end)  # Fallback final
            if not one.empty:
                all_frames.append(one)
            time.sleep(0.25)
        time.sleep(0.5)

    if not all_frames:
        # Último recurso: todo por Stooq
        for t in tickers:
            one = _single_history_stooq(t, start, end)
            if not one.empty:
                all_frames.append(one)
            time.sleep(0.15)

    if not all_frames:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume","Ticker"])

    out = pd.concat(all_frames, ignore_index=True)
    out = out.dropna(subset=["Open","High","Low","Close"])
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"].fillna(0)
    out["Date"] = pd.to_datetime(out["Date"], utc=True)
    out = out.sort_values(["Ticker","Date"]).reset_index(drop=True)
    return out

# -------------------------------------------------
# NUEVO: Descarga por ticker con CACHÉ y REANUDACIÓN
# -------------------------------------------------

def download_ohlcv_cached_resume(
    tickers: List[str],
    lookback_years: int = 3,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Descarga OHLCV por ticker, guardando inmediatamente cada éxito en parquet y
    permitiendo REANUDAR tras un bloqueo/rate-limit.
    - Usa caché si el archivo del ticker es "fresco" (< CFG.cache.freshness_days).
    - Si detecta rate-limit o N errores seguidos, guarda manifest y SALE (SystemExit 0).
    """
    tickers = _sanitize_universe(tickers)

    cache_dir = CFG.cache.dir
    fresh_days = int(CFG.cache.freshness_days)
    max_errs = int(CFG.cache.max_consecutive_errors_to_pause)

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.DateOffset(years=lookback_years)

    done_frames: List[pd.DataFrame] = []
    pending: List[str] = []
    consecutive_errors = 0

    # 1) Cargar de caché lo fresco
    for t in tickers:
        if is_fresh(cache_dir, t, fresh_days):
            dfc = load_df(cache_dir, t)
            if not dfc.empty:
                # Asegurar columnas estándar
                base_cols = ["Date","Open","High","Low","Close","Volume","Ticker"]
                miss = [c for c in base_cols if c not in dfc.columns]
                if not miss:
                    done_frames.append(dfc.assign(Ticker=t.upper()))
                    mark_status(cache_dir, t, "cached")
                    continue
        pending.append(t)

    # 2) Descargar pendientes de a 1 (para guardar inmediatamente)
    for t in pending:
        try:
            one = download_ohlcv([t], lookback_years=lookback_years, interval=interval)
            if not one.empty:
                sub = one[one["Ticker"].str.upper() == t.upper()].copy()
                if not sub.empty:
                    save_df(cache_dir, t, sub)
                    mark_status(cache_dir, t, "ok")
                    done_frames.append(sub)
                    consecutive_errors = 0
                    continue
            # si llegó aquí, no hubo datos para el ticker
            consecutive_errors += 1
            mark_status(cache_dir, t, "fail", "no_data")
        except Exception as e:
            consecutive_errors += 1
            mark_status(cache_dir, t, "fail", str(e))
            # Pausa si huele a rate-limit o demasiados errores seguidos
            if looks_like_rate_limit(str(e)) or consecutive_errors >= max_errs:
                save_manifest(cache_dir, load_manifest(cache_dir))
                print("[INFO] Posible rate-limit/bloqueo detectado. Progreso guardado. "
                      "Cierra ahora y reintenta en unas horas. Al relanzar, se reanudará.")
                raise SystemExit(0)

        # Pequeño backoff por cortesía
        time.sleep(0.25)

    if not done_frames:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume","Ticker"])

    out = pd.concat(done_frames, ignore_index=True)
    # Normalización final
    out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
    out = out.dropna(subset=["Date","Open","High","Low","Close"])
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"].fillna(0)
    out = out.sort_values(["Ticker","Date"]).reset_index(drop=True)
    return out
