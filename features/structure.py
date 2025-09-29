from __future__ import annotations
import pandas as pd
import numpy as np

from .volatility import bollinger_bands, atr, is_atr_falling


def local_minima(low: pd.Series, order: int = 2) -> pd.Series:
    """Marca mínimos locales simples: Low[i] < Low[i±k] para k=1..order."""
    low = low.astype(float)
    cond = pd.Series(True, index=low.index)
    for k in range(1, order + 1):
        cond &= (low < low.shift(k)) & (low < low.shift(-k))
    return cond.fillna(False)


def higher_lows(df: pd.DataFrame, lookback: int = 60, pivots_needed: int = 2) -> bool:
    lows = df["Low"].tail(lookback)
    piv = local_minima(lows, order=2)
    pts = lows[piv]
    if len(pts) < pivots_needed:
        return False
    # Tomar los últimos 'pivots_needed' mínimos y verificar creciente
    last = pts.tail(pivots_needed)
    return bool(last.is_monotonic_increasing)


def support_resistance(df: pd.DataFrame, window: int = 60) -> tuple[float, float]:
    sub = df.tail(window)
    sup = float(sub["Low"].min())
    res = float(sub["High"].max())
    return sup, res


def compression_flags(close: pd.Series, bb_period: int = 20, bb_std: float = 2.0, width_pct_max: float = 0.10) -> pd.DataFrame:
    bb = bollinger_bands(close, period=bb_period, num_std=bb_std)
    bb["bb_is_compressed"] = bb["bb_width_pct"] <= float(width_pct_max)
    return bb


def base_diagnostics(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    width_pct_max: float = 0.10,
    atr_lookback: int = 14,
    atr_falling_min_bars: int = 10,
    lookback_sr: int = 60,
    require_higher_lows: bool = True,
) -> dict:
    """Diagnóstico de base de consolidación.
    Retorna dict con campos: support, resistance, bb_width_pct, bb_is_compressed,
    atr_falling, higher_lows, is_base.
    """
    df = df.copy()
    bb = compression_flags(df["Close"], bb_period, bb_std, width_pct_max)
    atr_series = atr(df, lookback=atr_lookback)
    atr_fall = is_atr_falling(atr_series, min_bars=atr_falling_min_bars)

    support, resistance = support_resistance(df, window=lookback_sr)
    hl = higher_lows(df, lookback=lookback_sr, pivots_needed=2) if require_higher_lows else True

    last = df.index[-1]
    out = {
        "support": support,
        "resistance": resistance,
        "bb_width_pct": float(bb.loc[last, "bb_width_pct"]) if last in bb.index else np.nan,
        "bb_is_compressed": bool(bb.loc[last, "bb_is_compressed"]) if last in bb.index else False,
        "atr_falling": bool(atr_fall.loc[last]) if last in atr_fall.index else False,
        "higher_lows": bool(hl),
    }
    out["is_base"] = bool(out["bb_is_compressed"] and out["atr_falling"] and (out["higher_lows"] or not require_higher_lows))
    return out


def breakout_trigger(
    df: pd.DataFrame,
    resistance: float,
    min_close_above: float = 0.0,
) -> pd.Series:
    """Serie booleana: True cuando Close cierra por encima de resistencia + margen."""
    close = df["Close"].astype(float)
    level = float(resistance) * (1.0 + float(min_close_above))
    return close >= level
