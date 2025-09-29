# src/regime/detector.py
"""
Detector de régimen de mercado (Market Regime).

- Entrada: precios diarios de SPY y QQQ (DataFrames con columnas: date, open, high, low, close, volume)
- Salida: RegimeState con label ("BULL" | "NEUTRAL" | "BEAR"), confidence [0..1],
         multiplicador de riesgo y componentes del score para auditoría.

Fórmula del score compuesto (0..1):
    score = 0.6 * score_spy + 0.4 * score_qqq

Cada score índice (SPY/QQQ) se compone de:
    - Tendencia (50%): cierres por encima de SMA50/100/200
    - Momentum (35%): ROC(3m/6m/12m) > 0
    - Volatilidad (15%): penalización por volatilidad realizada elevada

Umbrales:
    score >= 0.67  -> BULL
    0.40 <= score < 0.67 -> NEUTRAL
    score < 0.40   -> BEAR
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


# =========================
# Dataclass de resultado
# =========================

@dataclass(frozen=True)
class RegimeState:
    label: str                 # "BULL" | "NEUTRAL" | "BEAR"
    confidence: float          # 0..1
    asof: pd.Timestamp         # fecha de cálculo (última barra válida)
    risk_multiplier: float     # sugerencia para dimensionar posición
    components: Dict[str, float]  # desglose de score (tendencia, momentum, vol, spy, qqq)


# =========================
# Utilidades de indicadores
# =========================

def _safe_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def _roc(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods)

def _realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    # Volatilidad realizada (desv. estándar de rendimientos diarios) anualizada
    rets = close.pct_change()
    vol = rets.rolling(window, min_periods=window).std() * np.sqrt(252)
    return vol

def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    # Asegura nombres en minúsculas (compatibilidad con loader.py)
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def _last_common_date(spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.Timestamp:
    d1 = pd.to_datetime(spy["date"])
    d2 = pd.to_datetime(qqq["date"])
    last = min(d1.iloc[-1], d2.iloc[-1])
    return last


# ====================================
# Cálculo de score por índice (SPY/QQQ)
# ====================================

def _index_regime_score(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Devuelve el score [0..1] para un índice y el desglose por componente.
    """
    df = _normalize_colnames(df).sort_values("date")
    close = df["close"].astype(float)

    # --- Tendencia (50%) ---
    sma50 = _safe_sma(close, 50)
    sma100 = _safe_sma(close, 100)
    sma200 = _safe_sma(close, 200)

    cond_50 = float(close.iloc[-1] > sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else 0.0
    cond_100 = float(close.iloc[-1] > sma100.iloc[-1]) if not np.isnan(sma100.iloc[-1]) else 0.0
    cond_200 = float(close.iloc[-1] > sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else 0.0

    trend_raw = (cond_50 + cond_100 + cond_200) / 3.0  # [0..1]

    # --- Momentum (35%) ---
    roc_3m = _roc(close, 63)   # ~3 meses
    roc_6m = _roc(close, 126)  # ~6 meses
    roc_12m = _roc(close, 252) # ~12 meses

    m3 = float(roc_3m.iloc[-1] > 0) if not np.isnan(roc_3m.iloc[-1]) else 0.0
    m6 = float(roc_6m.iloc[-1] > 0) if not np.isnan(roc_6m.iloc[-1]) else 0.0
    m12 = float(roc_12m.iloc[-1] > 0) if not np.isnan(roc_12m.iloc[-1]) else 0.0

    mom_raw = (m3 + m6 + m12) / 3.0  # [0..1]

    # --- Volatilidad (15%) ---
    # Penalización suave cuando la vol anualizada 20d supera umbrales típicos.
    vol20 = _realized_vol(close, 20).iloc[-1]
    # Puntos de referencia (ajustables): ~15% cómodo, 25% elevado, 35% muy alto
    if np.isnan(vol20):
        vol_penalty = 0.0
    else:
        if vol20 <= 0.15:
            vol_penalty = 0.0
        elif vol20 <= 0.25:
            vol_penalty = 0.10     # -0.10 sobre el total de 1.0
        elif vol20 <= 0.35:
            vol_penalty = 0.20
        else:
            vol_penalty = 0.30

    # Ponderaciones
    w_trend, w_mom, w_vol = 0.50, 0.35, 0.15

    base = w_trend * trend_raw + w_mom * mom_raw
    score = max(0.0, min(1.0, base - w_vol * (vol_penalty / 0.30)))  # normaliza penalización a [0..w_vol]

    components = {
        "trend": round(trend_raw, 4),
        "momentum": round(mom_raw, 4),
        "vol_penalty": round(vol_penalty, 4),
        "vol20": round(float(vol20) if not np.isnan(vol20) else 0.0, 4),
    }
    return score, components


# ============================
# Regla de ensamblaje SPY/QQQ
# ============================

def compute_regime(spy: pd.DataFrame, qqq: pd.DataFrame) -> RegimeState:
    """
    Calcula el régimen combinando SPY y QQQ.

    Args:
        spy: DataFrame de SPY (daily)
        qqq: DataFrame de QQQ (daily)

    Returns:
        RegimeState
    """
    spy = _normalize_colnames(spy).sort_values("date")
    qqq = _normalize_colnames(qqq).sort_values("date")

    # Asegurar alineación temporal mínima
    asof = _last_common_date(spy, qqq)

    score_spy, comp_spy = _index_regime_score(spy)
    score_qqq, comp_qqq = _index_regime_score(qqq)

    score = 0.60 * score_spy + 0.40 * score_qqq
    score = float(max(0.0, min(1.0, score)))

    if score >= 0.67:
        label = "BULL"
        risk_multiplier = 1.00   # tamaño normal
    elif score >= 0.40:
        label = "NEUTRAL"
        risk_multiplier = 0.60   # reduce riesgo
    else:
        label = "BEAR"
        risk_multiplier = 0.30   # riesgo mínimo

    components = {
        "score_spy": round(score_spy, 4),
        "score_qqq": round(score_qqq, 4),
        "trend_spy": comp_spy["trend"],
        "momentum_spy": comp_spy["momentum"],
        "vol_penalty_spy": comp_spy["vol_penalty"],
        "vol20_spy": comp_spy["vol20"],
        "trend_qqq": comp_qqq["trend"],
        "momentum_qqq": comp_qqq["momentum"],
        "vol_penalty_qqq": comp_qqq["vol_penalty"],
        "vol20_qqq": comp_qqq["vol20"],
    }

    return RegimeState(
        label=label,
        confidence=round(score, 4),
        asof=pd.to_datetime(asof),
        risk_multiplier=risk_multiplier,
        components=components,
    )


# ==================================================
# Helpers opcionales para integrarse fácil al flujo
# ==================================================

def compute_regime_from_symbols(
    fetch_func,  # e.g., data.loader.fetch_history
    spy_symbol: str = "SPY",
    qqq_symbol: str = "QQQ",
    period: str = "max",
    interval: str = "1d",
    force_refresh: bool = False,
) -> RegimeState:
    """
    Descarga SPY/QQQ usando `fetch_func` (firma compatible con loader.fetch_history)
    y calcula el régimen.

    Ejemplo:
        from data.loader import fetch_history
        state = compute_regime_from_symbols(fetch_history)

    """
    spy = fetch_func(spy_symbol, period=period, interval=interval, force_refresh=force_refresh)
    qqq = fetch_func(qqq_symbol, period=period, interval=interval, force_refresh=force_refresh)
    return compute_regime(spy, qqq)


def regime_summary(state: RegimeState) -> str:
    """
    Cadena corta y legible, útil para logs o notificaciones.
    """
    return (
        f"Régimen: {state.label} | Confianza: {state.confidence:.2f} | "
        f"Riesgo x{state.risk_multiplier:.2f} | Fecha: {state.asof.date()}"
    )


# ======================
# Uso directo de consola
# ======================

if __name__ == "__main__":
    # Ejemplo de uso independiente (requiere loader en el PYTHONPATH)
    try:
        from data.loader import fetch_history  # noqa: WPS433
        state = compute_regime_from_symbols(fetch_history)
        print(regime_summary(state))
        print("Componentes:", state.components)
    except Exception as e:
        print("[INFO] Ejecuta este módulo dentro del proyecto con loader disponible.")
        print("Detalle:", e)
