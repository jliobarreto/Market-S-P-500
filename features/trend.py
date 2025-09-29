from __future__ import annotations
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()


def add_emas(df: pd.DataFrame, fast: int = 20, slow: int = 50, long: int = 100) -> pd.DataFrame:
    out = df.copy()
    out[f"ema{fast}"] = ema(out["Close"], fast)
    out[f"ema{slow}"] = ema(out["Close"], slow)
    out[f"ema{long}"] = ema(out["Close"], long)
    return out


def rs_ratio(asset_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    """Línea de fuerza relativa: asset / benchmark (normalizada por el primer valor válido)."""
    r = asset_close.astype(float) / bench_close.astype(float)
    first = r[r.notna()].iloc[0] if r.notna().any() else None
    return (r / first) if first and first != 0 else r


def rs_score(asset_close: pd.Series, bench_close: pd.Series, lookback: int = 60) -> pd.Series:
    """Score de RS basado en retorno relativo de 'lookback' barras.
    >1 implica que el activo superó al benchmark en el periodo.
    """
    pa = asset_close.astype(float)
    pb = bench_close.astype(float)
    ra = pa / pa.shift(lookback)
    rb = pb / pb.shift(lookback)
    score = (ra / rb)
    return score
