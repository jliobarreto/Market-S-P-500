# src/scoring/ranker.py
"""
Ranker multi-factor para universo S&P 500.

Produce una tabla con score 0..100 por ticker combinando:
- Técnica (breakout, tendencia, relative strength vs SPY)
- Momentum (3/6/12 meses)
- Calidad de entrada (distancia a stop/ATR)
- Liquidez (dollar-volume)
- Penalización por volatilidad reciente
- Ajuste por régimen de mercado (risk_multiplier)

Requiere:
- DataFrames con columnas: date, open, high, low, close, volume (minúsculas)
- signals.breakout.breakout_signal
- regime.detector.RegimeState (opcional pero recomendado)

Salida:
pd.DataFrame con columnas:
[ticker, date, close, score, bucket, w_tech, w_mom, w_quality, w_liq, w_vol_pen,
 tech_score, mom_score, quality_score, liq_score, vol_penalty, rs_63d, trend_score,
 entry, stop, tp1, tp2, atr14, dollar_vol20, notes]

Nota: los pesos por defecto suman 100, pero puedes ajustarlos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple

import numpy as np
import pandas as pd

from signals.breakout import breakout_signal
from regime.detector import RegimeState


# ======================
# Configuración de pesos
# ======================

@dataclass(frozen=True)
class ScoringWeights:
    technical: float = 40.0   # breakout + tendencia + RS
    momentum: float = 25.0    # 3/6/12m
    quality: float = 15.0     # distancia a stop/ATR (mejor cercanía)
    liquidity: float = 10.0   # dollar-volume
    vol_penalty: float = 10.0 # penalización por vol (se resta del total)

    def normalized(self) -> "ScoringWeights":
        # Normaliza para que (technical + momentum + quality + liquidity) = 100 y vol_penalty se maneja aparte
        pos_sum = self.technical + self.momentum + self.quality + self.liquidity
        if pos_sum <= 0:
            return ScoringWeights(40, 25, 15, 10, self.vol_penalty)
        k = 100.0 / pos_sum
        return ScoringWeights(
            technical=self.technical * k,
            momentum=self.momentum * k,
            quality=self.quality * k,
            liquidity=self.liquidity * k,
            vol_penalty=self.vol_penalty,  # se usa como resta final escalada a [0..vol_penalty]
        )


# ======================
# Utilidades de cálculo
# ======================

def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Presentes: {list(df.columns)}")


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    return out.sort_values("date").reset_index(drop=True)


def _pct_change_period(close: pd.Series, periods: int) -> float:
    if len(close) <= periods or periods <= 0:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[-periods] - 1.0))


def _rolling_mean(series: pd.Series, window: int) -> float:
    s = series.rolling(window, min_periods=window).mean()
    v = s.iloc[-1]
    return float(v) if not pd.isna(v) else float("nan")


def _atr(df: pd.DataFrame, window: int = 14) -> float:
    _require_cols(df, ["high", "low", "close"])
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else float("nan")


def _realized_vol(close: pd.Series, window: int = 20) -> float:
    rets = close.pct_change()
    vol = rets.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252)
    v = vol.iloc[-1]
    return float(v) if not pd.isna(v) else float("nan")


def _relative_strength(
    stock_close: pd.Series,
    spy_close: pd.Series,
    lookback: int = 63,  # ~3 meses
) -> float:
    """
    RS simple: ratio(stock/spy) y su retorno en lookback.
    """
    if len(stock_close) <= lookback or len(spy_close) <= lookback:
        return float("nan")
    ratio_now = float(stock_close.iloc[-1] / spy_close.iloc[-1])
    ratio_prev = float(stock_close.iloc[-lookback - 1] / spy_close.iloc[-lookback - 1])
    return (ratio_now / ratio_prev) - 1.0


def _trend_score(close: pd.Series) -> float:
    """
    Score [0..1] por tendencia usando SMA50/100/200.
    """
    sma50 = close.rolling(50, min_periods=50).mean().iloc[-1]
    sma100 = close.rolling(100, min_periods=100).mean().iloc[-1]
    sma200 = close.rolling(200, min_periods=200).mean().iloc[-1]
    c = float(close.iloc[-1])
    bits = 0.0
    total = 0
    for sma in [sma50, sma100, sma200]:
        if not pd.isna(sma):
            total += 1
            if c > float(sma):
                bits += 1
    if total == 0:
        return 0.0
    return bits / total


def _vol_penalty(vol20: float) -> float:
    """
    Penalización heurística por volatilidad anualizada 20d.
    Devuelve [0..1] (1 = máxima penalización).
    """
    if np.isnan(vol20):
        return 0.0
    # tramos suaves
    if vol20 <= 0.15:
        return 0.0
    if vol20 <= 0.25:
        return 0.33
    if vol20 <= 0.35:
        return 0.66
    return 1.0


def _bucket_by_score(score: float) -> str:
    if score >= 85:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 65:
        return "B"
    if score >= 55:
        return "C"
    return "D"


# ===================================
# Scoring por ticker (núcleo)
# ===================================

def _score_ticker(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    regime: Optional[RegimeState],
    weights: ScoringWeights,
    *,
    breakout_lookback: int = 55,
    vol_confirm: bool = True,
    vol_window: int = 20,
    min_close_buffer: float = 0.0,
    atr_window: int = 14,
    atr_mult: float = 1.8,
    min_price: float = 5.0,
    min_avg_dollar_vol: float = 1_000_000.0,
) -> Optional[Dict]:
    """
    Calcula el score y metadatos para un ticker. Devuelve None si no cumple filtros básicos.
    """
    try:
        df = _norm_cols(df)
        spy_df = _norm_cols(spy_df)
        _require_cols(df, ["date", "open", "high", "low", "close", "volume"])
        _require_cols(spy_df, ["date", "close"])

        last = df.iloc[-1]
        last_date = pd.to_datetime(last["date"])
        close = float(last["close"])

        # Filtros básicos
        if close < min_price or len(df) < 252:  # al menos ~1 año de datos
            return None

        dv20 = _rolling_mean(df["close"].astype(float) * df["volume"].astype(float), 20)
        if np.isnan(dv20) or dv20 < min_avg_dollar_vol:
            return None

        # Señal de breakout (proporciona entry/stop/tps y notas)
        sig = breakout_signal(
            df,
            lookback=breakout_lookback,
            vol_confirm=vol_confirm,
            vol_window=vol_window,
            min_close_buffer=min_close_buffer,
            atr_window=atr_window,
            atr_mult=atr_mult,
        )

        # Técnica: tendencia + RS + (bonus si hay breakout válido)
        trend = _trend_score(df["close"].astype(float))  # [0..1]
        rs_63 = _relative_strength(df["close"].astype(float), spy_df["close"].astype(float), 63)
        rs_flag = 1.0 if (not np.isnan(rs_63) and rs_63 > 0) else 0.0
        breakout_bonus = 0.25 if sig.triggered else 0.0  # pequeño empuje si hay señal válida
        tech_score = np.clip(0.60 * trend + 0.40 * rs_flag + breakout_bonus, 0.0, 1.0)

        # Momentum multi-horizonte: 3/6/12m (>0 suma, <0 no)
        m3 = _pct_change_period(df["close"].astype(float), 63)
        m6 = _pct_change_period(df["close"].astype(float), 126)
        m12 = _pct_change_period(df["close"].astype(float), 252)
        mom_bits = sum(int(x > 0) for x in [m3, m6, m12] if not np.isnan(x))
        mom_total = sum(1 for x in [m3, m6, m12] if not np.isnan(x))
        mom_score = (mom_bits / mom_total) if mom_total > 0 else 0.0  # [0..1]

        # Calidad de entrada: mejor cuando el stop está "cerca" (entry-stop)/ATR moderado
        atr14 = _atr(df, window=atr_window)
        quality_score = 0.0
        r_now = None
        if sig.triggered and sig.entry is not None and sig.stop is not None and not np.isnan(atr14) and atr14 > 0:
            r_now = float(sig.entry - sig.stop)  # distancia riesgo en $
            # Convertir a múltiplos de ATR (más pequeño = mejor calidad)
            r_atr = r_now / atr14 if atr14 > 0 else np.inf
            # mapa heurístico: <=1 ATR -> 1.0 ; 1.5 -> 0.7 ; 2.0 -> 0.4 ; >=3 -> 0.0
            if r_atr <= 1.0:
                quality_score = 1.0
            elif r_atr <= 1.5:
                quality_score = 0.7
            elif r_atr <= 2.0:
                quality_score = 0.4
            elif r_atr <= 3.0:
                quality_score = 0.1
            else:
                quality_score = 0.0
        else:
            # Si no hay señal, usamos una aproximación: última barra vs ATR (rango/ATR)
            if not np.isnan(atr14) and atr14 > 0:
                # distancia del close al mínimo de 20 días en ATRs (más cerca a soporte moderado = mejor)
                low20 = df["low"].astype(float).rolling(20, min_periods=20).min().iloc[-1]
                if not np.isnan(low20):
                    dist = (close - low20) / atr14
                    # mejor calidad si dist ~1-2 ATR (evita demasiado extendido)
                    if dist <= 0.5:
                        quality_score = 0.3
                    elif dist <= 1.0:
                        quality_score = 1.0
                    elif dist <= 2.0:
                        quality_score = 0.7
                    elif dist <= 3.0:
                        quality_score = 0.4
                    else:
                        quality_score = 0.1

        # Liquidez: normalizamos dollar-volume 20d en percentil vs universo (se ajustará fuera)
        # Aquí devolvemos el valor bruto; la normalización se hará globalmente
        liq_raw = dv20  # se transformará a [0..1] por percentil

        # Penalización por volatilidad (se resta al final)
        vol20 = _realized_vol(df["close"].astype(float), 20)
        vol_pen = _vol_penalty(vol20)  # [0..1]

        # Combine factores con pesos (0..100) y restamos penalización
        w = weights.normalized()
        partial = (
            w.technical * tech_score +
            w.momentum * mom_score +
            w.quality * quality_score
            # liquidity se completará tras normalización global
        )
        # guardamos intermedios; la parte de liquidez y la resta vol_penalty se aplican después
        return {
            "ticker": ticker,
            "date": last_date,
            "close": round(close, 4),
            "tech_score": float(np.round(tech_score, 4)),
            "mom_score": float(np.round(mom_score, 4)),
            "quality_score": float(np.round(quality_score, 4)),
            "liq_raw": float(liq_raw),
            "vol_penalty": float(np.round(vol_pen, 4)),
            "rs_63d": float(np.round(rs_63, 4)) if not np.isnan(rs_63) else np.nan,
            "trend_score": float(np.round(trend, 4)),
            "entry": sig.entry,
            "stop": sig.stop,
            "tp1": sig.tp1,
            "tp2": sig.tp2,
            "atr14": float(np.round(atr14, 4)) if not np.isnan(atr14) else np.nan,
            "w_tech": w.technical,
            "w_mom": w.momentum,
            "w_quality": w.quality,
            "w_liq": w.liquidity,
            "w_vol_pen": w.vol_penalty,
            "partial_no_liq": float(np.round(partial, 4)),
            "dollar_vol20": float(np.round(dv20, 2)),
            "notes": f"breakout={'YES' if sig.triggered else 'NO'} | {sig.notes}",
        }
    except Exception:
        return None


# ===================================
# Rank del universo
# ===================================

def rank(
    prices: Dict[str, pd.DataFrame],
    spy_prices: pd.DataFrame,
    regime: Optional[RegimeState] = None,
    weights: ScoringWeights = ScoringWeights(),
    *,
    breakout_lookback: int = 55,
    vol_confirm: bool = True,
    vol_window: int = 20,
    min_close_buffer: float = 0.0,
    atr_window: int = 14,
    atr_mult: float = 1.8,
    min_price: float = 5.0,
    min_avg_dollar_vol: float = 1_000_000.0,
    sector_map: Optional[pd.DataFrame] = None,  # opcional: df con columnas [ticker, sector]
) -> pd.DataFrame:
    """
    Calcula el ranking del universo con score 0..100.

    - Normaliza la liquidez por percentiles (0..1) en el universo
    - Resta una penalización por volatilidad (escala vol_penalty)
    - Ajusta por régimen (multiplica el score final por regime.risk_multiplier)

    Returns:
        DataFrame ordenado por score desc.
    """
    spy_df = _norm_cols(spy_prices)
    rows = []
    for ticker, df in prices.items():
        row = _score_ticker(
            ticker, df, spy_df, regime, weights,
            breakout_lookback=breakout_lookback,
            vol_confirm=vol_confirm,
            vol_window=vol_window,
            min_close_buffer=min_close_buffer,
            atr_window=atr_window,
            atr_mult=atr_mult,
            min_price=min_price,
            min_avg_dollar_vol=min_avg_dollar_vol,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "date", "close", "score", "bucket", "w_tech", "w_mom", "w_quality", "w_liq", "w_vol_pen",
            "tech_score", "mom_score", "quality_score", "liq_score", "vol_penalty", "rs_63d", "trend_score",
            "entry", "stop", "tp1", "tp2", "atr14", "dollar_vol20", "notes"
        ])

    df_out = pd.DataFrame(rows)

    # Normalización de liquidez: percentil dentro del universo (0..1)
    liq = df_out["liq_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if liq.max() == liq.min():
        liq_score = pd.Series(0.5, index=df_out.index)  # si todo igual, valor medio
    else:
        ranks = liq.rank(method="average", pct=True)  # 0..1
        liq_score = ranks

    df_out["liq_score"] = liq_score.round(4)

    # Combinación final
    w = weights.normalized()
    base_score = (
        df_out["partial_no_liq"]
        + w.liquidity * df_out["liq_score"]
    )

    # Resta por volatilidad
    vol_pen_scaled = (df_out["vol_penalty"].clip(0, 1) * weights.vol_penalty)
    score = base_score - vol_pen_scaled

    # Ajuste por régimen (si se pasa)
    if regime is not None:
        score = score * float(regime.risk_multiplier)

    # Clampeo y bucket
    score = score.clip(0, 100)
    df_out["score"] = score.round(2)
    df_out["bucket"] = df_out["score"].apply(_bucket_by_score)

    # Orden final
    keep_cols = [
        "ticker", "date", "close", "score", "bucket",
        "w_tech", "w_mom", "w_quality", "w_liq", "w_vol_pen",
        "tech_score", "mom_score", "quality_score", "liq_score",
        "vol_penalty", "rs_63d", "trend_score",
        "entry", "stop", "tp1", "tp2", "atr14", "dollar_vol20", "notes",
    ]
    out = df_out[keep_cols].sort_values(["score", "liq_score", "close"], ascending=[False, False, False]).reset_index(drop=True)

    # Enriquecimiento opcional con sector
    if sector_map is not None and "ticker" in sector_map.columns and "sector" in sector_map.columns:
        sm = sector_map.copy()
        sm["ticker"] = sm["ticker"].astype(str).str.upper().str.strip()
        out = out.merge(sm[["ticker", "sector"]].drop_duplicates(), on="ticker", how="left")
        # reordenar con sector al lado
        cols = out.columns.tolist()
        cols.insert(1, cols.pop(cols.index("sector")))
        out = out[cols]

    return out


# ======================
# Ejecución de ejemplo
# ======================

if __name__ == "__main__":
    # Demo mínima (requiere loader/universe en el PYTHONPATH del proyecto)
    try:
        from data.loader import fetch_universe, fetch_history
        from universe.sp500 import load_sp500, load_sp500_with_metadata
        from regime.detector import compute_regime

        uni = load_sp500()[:25]  # muestra
        prices = fetch_universe(uni, period="2y")
        spy = fetch_history("SPY", period="2y")
        qqq = fetch_history("QQQ", period="2y")
        state = compute_regime(spy, qqq)

        meta = load_sp500_with_metadata()[["ticker", "sector"]]
        table = rank(prices, spy, state, sector_map=meta)
        print(table.head(20))
    except Exception as e:
        print("Ejecutar dentro del proyecto con módulos disponibles. Detalle:", e)
