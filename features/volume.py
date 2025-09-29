from __future__ import annotations
import pandas as pd


def sma_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.astype(float).rolling(window, min_periods=window).mean()


def volume_breakout(
    volume: pd.Series,
    window: int = 20,
    multiple: float = 1.5,
) -> pd.Series:
    """True cuando el volumen actual >= multiple * SMA(volume, window)."""
    vma = sma_volume(volume, window)
    return volume.astype(float) >= (multiple * vma)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume clásico."""
    close = close.astype(float)
    volume = volume.astype(float)
    direction = close.diff().fillna(0.0)
    sign = direction.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    return (sign * volume).cumsum()
