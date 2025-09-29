from __future__ import annotations
import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Average True Range clásico (no anualizado).
    Requiere columnas: ['High','Low','Close'].
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(lookback, min_periods=lookback).mean()


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bandas de Bollinger y ancho relativo (% del precio).
    Devuelve columnas: 'bb_mid','bb_up','bb_low','bb_width_pct'.
    """
    close = close.astype(float)
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    up = mid + num_std * std
    low = mid - num_std * std
    width_pct = (up - low) / close.replace(0, np.nan)
    out = pd.DataFrame({
        "bb_mid": mid,
        "bb_up": up,
        "bb_low": low,
        "bb_width_pct": width_pct,
    })
    return out


def is_atr_falling(atr_series: pd.Series, min_bars: int = 10) -> pd.Series:
    """Retorna una Serie booleana donde True indica que el ATR viene cayendo
    durante al menos 'min_bars' barras consecutivas.
    """
    s = atr_series.fillna(method="ffill")
    # True si el valor actual <= valor previo
    non_increasing = s <= s.shift(1)
    # Contar rachas consecutivas
    streak = (~non_increasing).cumsum()
    run_length = non_increasing.groupby(streak).cumcount() + 1
    # Donde non_increasing es False, la racha se invalida (poner 0)
    run_length = run_length.where(non_increasing, 0)
    return run_length >= int(min_bars)
