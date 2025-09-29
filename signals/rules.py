from __future__ import annotations
import pandas as pd

from features.volatility import atr, bollinger_bands, is_atr_falling
from features.volume import volume_breakout, obv
from features.trend import add_emas, rs_score
from features.structure import base_diagnostics, breakout_trigger

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None


def check_base_and_breakout(
    df: pd.DataFrame,
    bench_df: pd.DataFrame,
) -> dict:
    """Evalúa si un activo está en base y si tiene disparo de ruptura.
    Requiere DataFrame diario con columnas OHLCV y DataFrame benchmark con Close.
    """
    if df.empty or bench_df.empty:
        return {"is_base": False, "trigger": False}

    rules = CFG.rules if CFG else None
    if rules is None:
        raise ValueError("Config no cargada para reglas.")

    # Diagnóstico de base
    base_info = base_diagnostics(
        df,
        bb_period=rules.bb_period,
        bb_std=rules.bb_std,
        width_pct_max=rules.bb_width_pct_max,
        atr_lookback=rules.atr_lookback,
        atr_falling_min_bars=rules.atr_falling_min_bars,
        lookback_sr=60,
        require_higher_lows=rules.need_higher_lows,
    )

    # Momentum y fuerza relativa
    rs = rs_score(df["Close"], bench_df["Close"], lookback=60)
    base_info["rs_score"] = float(rs.iloc[-1]) if not rs.empty else 0.0
    base_info["rsi_ok"] = df["Close"].rolling(14).apply(lambda x: ((x.pct_change()+1).prod()-1)*100).iloc[-1] > rules.rsi_min

    # Volumen
    base_info["vol_breakout"] = bool(volume_breakout(df["Volume"], window=rules.vol_sma, multiple=rules.vol_breakout_mult).iloc[-1])
    base_info["obv"] = float(obv(df["Close"], df["Volume"]).iloc[-1])

    # Disparo de ruptura
    base_info["trigger"] = bool(breakout_trigger(df, base_info["resistance"], min_close_above=0.0).iloc[-1])

    return base_info
