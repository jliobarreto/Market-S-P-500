# src/signals/breakout.py
"""
Señal de Breakout (Donchian High) con confirmación de volumen y niveles operativos.

Características:
- Sin look-ahead: usa el MÁXIMO de los últimos N días EXCLUYENDO la barra actual.
- Confirmación opcional por volumen: z-score sobre ventana móvil (default 20).
- Cálculo de ATR para stop y R-multiples de TP.
- Función de escaneo sobre universo: devuelve tabla con señales listas para filtrar/ordenar.

Requisitos de columnas del DataFrame: date, open, high, low, close, volume (minúsculas).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Utilidades de indicadores
# =========================

def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Presentes: {list(df.columns)}")


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range clásico (Wilder). Retorna Serie alineada al índice del df."""
    _require_cols(df, ["high", "low", "close"])
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing: EMA con alpha = 1/window
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return atr


def _volume_zscore(df: pd.DataFrame, window: int = 20) -> float:
    """Z-score del volumen en la última barra."""
    _require_cols(df, ["volume"])
    vol = df["volume"].astype(float)
    mean = vol.rolling(window, min_periods=window).mean()
    std = vol.rolling(window, min_periods=window).std(ddof=0)
    if pd.isna(mean.iloc[-1]) or pd.isna(std.iloc[-1]) or std.iloc[-1] == 0:
        return float("nan")
    return float((vol.iloc[-1] - mean.iloc[-1]) / std.iloc[-1])


def _donchian_high_prev(df: pd.DataFrame, lookback: int) -> float:
    """
    Máximo de los últimos N días excluyendo la barra actual (evita look-ahead).
    """
    _require_cols(df, ["high"])
    highs = df["high"].astype(float)
    if len(highs) < lookback + 1:  # necesitamos al menos N previos + barra actual
        return float("nan")
    return float(highs.iloc[-(lookback + 1):-1].max())


# =========================
# Resultado de la señal
# =========================

@dataclass(frozen=True)
class BreakoutSignal:
    triggered: bool                 # ¿hubo señal hoy?
    entry: Optional[float]          # nivel de entrada (close actual)
    stop: Optional[float]           # stop por ATR (si se pasa atr_mult)
    tp1: Optional[float]            # TP por múltiplos de R
    tp2: Optional[float]
    r_multiple: Optional[float]     # (entry - stop)
    lookback_high: Optional[float]  # máximo N previo
    vol_zscore: Optional[float]     # z-score de volumen
    notes: str                      # texto de soporte (motivo/validaciones)


# =========================
# Señal principal
# =========================

def breakout_signal(
    df: pd.DataFrame,
    lookback: int = 55,
    vol_confirm: bool = True,
    vol_window: int = 20,
    min_close_buffer: float = 0.0,
    # niveles operativos
    atr_window: int = 14,
    atr_mult: float = 1.8,
    tp1_r: float = 1.5,
    tp2_r: float = 2.5,
) -> BreakoutSignal:
    """
    Detecta un breakout de N-máximos (Donchian) con confirmación opcional por volumen.

    Args:
        df: precios OHLCV diarios (columnas minúsculas)
        lookback: N días para el máximo previo (excluyendo la barra actual)
        vol_confirm: si True, exige vol_zscore > 1.0 (ajustable editando abajo)
        vol_window: ventana para z-score de volumen
        min_close_buffer: margen mínimo por encima del máximo previo (p.ej. 0.001 = 0.1%)
        atr_window: ventana ATR
        atr_mult: múltiplo de ATR para stop
        tp1_r, tp2_r: R-múltiplos para take-profit

    Returns:
        BreakoutSignal
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    _require_cols(df, ["date", "open", "high", "low", "close", "volume"])

    # Validación de historial mínimo
    min_bars = max(lookback + 1, vol_window + 1, atr_window + 1, 100)
    if len(df) < min_bars:
        return BreakoutSignal(
            triggered=False,
            entry=None,
            stop=None,
            tp1=None,
            tp2=None,
            r_multiple=None,
            lookback_high=None,
            vol_zscore=None,
            notes=f"Historial insuficiente (<{min_bars} barras).",
        )

    close = df["close"].astype(float)
    last_close = float(close.iloc[-1])

    # 1) Máximo N previo excluyendo la barra actual
    high_n_prev = _donchian_high_prev(df, lookback=lookback)
    if np.isnan(high_n_prev):
        return BreakoutSignal(
            triggered=False,
            entry=None,
            stop=None,
            tp1=None,
            tp2=None,
            r_multiple=None,
            lookback_high=None,
            vol_zscore=None,
            notes="No es posible calcular high_n_prev.",
        )

    # 2) Condición de breakout con buffer mínimo
    buffer_level = high_n_prev * (1 + min_close_buffer)
    is_breakout = bool(last_close > buffer_level)

    # 3) Confirmación por volumen (opcional)
    vz = _volume_zscore(df, window=vol_window)
    vol_ok = True
    vol_threshold = 1.0  # puedes parametrizarlo si quieres
    if vol_confirm:
        vol_ok = (not np.isnan(vz)) and (vz > vol_threshold)

    # 4) ATR y niveles operativos
    atr = _atr(df, window=atr_window).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        atr = float("nan")

    stop = None
    tp1 = None
    tp2 = None
    r_mult = None

    if is_breakout and vol_ok and not np.isnan(atr):
        stop = round(last_close - atr_mult * atr, 4)
        r_mult = round(last_close - stop, 4)
        tp1 = round(last_close + tp1_r * (last_close - stop), 4)
        tp2 = round(last_close + tp2_r * (last_close - stop), 4)

    notes = []
    notes.append(f"N={lookback}")
    notes.append(f"breakout={'YES' if is_breakout else 'NO'} (close>{buffer_level:.4f})")
    if vol_confirm:
        notes.append(f"vol_z={vz:.2f} ({'OK' if vol_ok else 'LOW'})")
    if np.isnan(atr):
        notes.append("ATR n/d")
    else:
        notes.append(f"ATR{atr_window}={atr:.4f}")

    return BreakoutSignal(
        triggered=bool(is_breakout and vol_ok and not np.isnan(atr)),
        entry=round(last_close, 4) if is_breakout else None,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        r_multiple=r_mult,
        lookback_high=round(high_n_prev, 4),
        vol_zscore=None if np.isnan(vz) else round(vz, 3),
        notes=" | ".join(notes),
    )


# ====================================
# Escaneo de universo y tabla de salida
# ====================================

def scan_breakout(
    prices: Dict[str, pd.DataFrame],
    lookback: int = 55,
    vol_confirm: bool = True,
    vol_window: int = 20,
    min_close_buffer: float = 0.0,
    atr_window: int = 14,
    atr_mult: float = 1.8,
    tp1_r: float = 1.5,
    tp2_r: float = 2.5,
    min_price: float = 5.0,
    min_avg_dollar_vol: float = 1_000_000.0,  # filtro de liquidez (precio*vol promedio 20d)
) -> pd.DataFrame:
    """
    Escanea un diccionario {ticker: df} y devuelve un DataFrame con señales de breakout.

    Filtros de calidad:
    - Precio actual >= min_price
    - Dollar-volume promedio 20d >= min_avg_dollar_vol

    Columns del resultado:
        ticker, date, close, breakout, entry, stop, tp1, tp2, r,
        lookback_high, vol_z, notes
    """
    rows = []
    for ticker, df in prices.items():
        try:
            if df is None or df.empty:
                continue
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_values("date").reset_index(drop=True)

            _require_cols(df, ["date", "close", "volume"])

            last_row = df.iloc[-1]
            last_date = pd.to_datetime(last_row["date"])
            last_close = float(last_row["close"])

            # Liquidez básica
            dv20 = (df["close"].astype(float) * df["volume"].astype(float)).rolling(20).mean().iloc[-1]
            if pd.isna(dv20):
                dv20 = 0.0

            if last_close < min_price or dv20 < min_avg_dollar_vol:
                # Salta por filtros de liquidez/precio
                continue

            sig = breakout_signal(
                df,
                lookback=lookback,
                vol_confirm=vol_confirm,
                vol_window=vol_window,
                min_close_buffer=min_close_buffer,
                atr_window=atr_window,
                atr_mult=atr_mult,
                tp1_r=tp1_r,
                tp2_r=tp2_r,
            )

            rows.append(
                {
                    "ticker": ticker,
                    "date": last_date,
                    "close": round(last_close, 4),
                    "breakout": sig.triggered,
                    "entry": sig.entry,
                    "stop": sig.stop,
                    "tp1": sig.tp1,
                    "tp2": sig.tp2,
                    "r": sig.r_multiple,
                    "lookback_high": sig.lookback_high,
                    "vol_z": sig.vol_zscore,
                    "notes": sig.notes,
                }
            )
        except Exception as e:
            # No detener el escaneo por errores puntuales
            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.NaT,
                    "close": np.nan,
                    "breakout": False,
                    "entry": None,
                    "stop": None,
                    "tp1": None,
                    "tp2": None,
                    "r": None,
                    "lookback_high": None,
                    "vol_z": None,
                    "notes": f"ERROR: {e}",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["breakout", "vol_z", "close"], ascending=[False, False, False]).reset_index(drop=True)
    return out


# ======================
# Ejecución de ejemplo
# ======================

if __name__ == "__main__":
    # Demo mínima (requiere loader y universo en el PYTHONPATH del proyecto)
    try:
        from data.loader import fetch_history
        from universe.sp500 import load_sp500

        tickers = load_sp500()[:10]  # muestra
        prices = {t: fetch_history(t, period="2y") for t in tickers}
        table = scan_breakout(prices, lookback=55, min_close_buffer=0.001)
        print(table.head(20))
    except Exception as e:
        print("Ejecuta este módulo dentro del proyecto con loader/universe disponibles.")
        print("Detalle:", e)
