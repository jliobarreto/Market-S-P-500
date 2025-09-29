# reports/weekly_ranker.py
from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass
import math
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr

from data.universe import build_universe
from data.loader_yf import download_ohlcv_cached_resume  # ← usa caché + reanudación
from signals.rules import check_base_and_breakout
from signals.planner import plan_trades
from risk.sizing import plan_position_from_levels

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None

# ---------------------------
# Fuentes de benchmark / VIX
# ---------------------------

def _stooq(sym: str, period_years: int) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.DateOffset(years=period_years)
    try:
        df = pdr.DataReader(f"{sym}.US", "stooq", start=start, end=end)
    except Exception:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.capitalize).sort_index()
        return df.dropna(subset=["Close"])
    return pd.DataFrame()

def _yf_single(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(
            period=period, interval="1d",
            auto_adjust=False, actions=False, prepost=False
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns=str.capitalize)
            return df.dropna(subset=["Close"])
    except Exception:
        pass
    return pd.DataFrame()

def _get_benchmark_df() -> pd.DataFrame:
    # SPY/VOO/IVV por Yahoo; si falla todo, Stooq SPY.US
    for tkr in ["SPY", "VOO", "IVV"]:
        df = _yf_single(tkr, period=f"{CFG.data.lookback_years}y")
        if not df.empty:
            return df
    df = _stooq("SPY", CFG.data.lookback_years)
    if not df.empty:
        return df
    return pd.DataFrame(columns=["Close"])

def _get_vix_df() -> Tuple[pd.DataFrame, bool]:
    """
    Devuelve (df, is_real_vix).
    is_real_vix=True solo si los datos provienen de ^VIX real.
    Si usamos VIXY/VIXM o Stooq, devolvemos False para no comparar contra vix_max.
    """
    df = _yf_single("^VIX", period="3mo")
    if not df.empty:
        return df, True
    for proxy in ["VIXY", "VIXM"]:
        d = _yf_single(proxy, period="3mo")
        if not d.empty:
            return d, False
    # stooq: usa VIXY.US como aproximación
    d = _stooq("VIXY", 1)
    return d, False if not d.empty else (pd.DataFrame(), False)

# ---------------------------
# Utilidades de cálculo
# ---------------------------

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    out["Open"] = df["Open"].resample("W-FRI").first()
    out["High"] = df["High"].resample("W-FRI").max()
    out["Low"]  = df["Low"].resample("W-FRI").min()
    out["Close"]= df["Close"].resample("W-FRI").last()
    out["Volume"] = df["Volume"].resample("W-FRI").sum(min_count=1) if "Volume" in df.columns else 0
    return out.dropna(subset=["Close"])

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.astype(float).ewm(span=span, adjust=False).mean()

def _macro_filter_ok() -> bool:
    if CFG is None:
        return True

    # Condición SPY > EMA50 semanal (tolerante si faltan datos)
    spy = _get_benchmark_df()
    if spy.empty:
        cond_spy = True  # no bloqueamos si no hay datos
    else:
        w = _to_weekly(spy)
        if w.empty or len(w) < 60:
            cond_spy = True
        else:
            cond_spy = bool(w["Close"].iloc[-1] > _ema(w["Close"], 50).iloc[-1])

    # Condición VIX < umbral (solo si es ^VIX real)
    vix_df, is_real_vix = _get_vix_df()
    if not is_real_vix or vix_df.empty:
        cond_vix = True
    else:
        cond_vix = float(vix_df["Close"].dropna().iloc[-1]) < float(CFG.macro.vix_max)

    return bool(cond_spy and cond_vix) if CFG.macro.require_spy_above_ema50w else True

def _normalize_potential(tp: float, ref: float, target_pct: float = 0.20) -> float:
    if ref <= 0:
        return 0.0
    pct = (tp / ref) - 1.0
    return float(np.clip(pct / target_pct, 0.0, 1.0))

# ---------------------------
# Modelo de salida
# ---------------------------

@dataclass
class ScoredIdea:
    ticker: str
    score_total: float
    score_rs: float
    score_structure: float
    score_volume: float
    score_potential: float
    resistance: float
    entry1: float
    entry2: float
    entry3: float
    tp1: float
    tp2: float
    sl: float
    position_size_usd: float
    shares_e1: float
    shares_e2: float
    shares_e3: float

# ---------------------------
# Pipeline principal
# ---------------------------

def rank_universe(source: str = "combined") -> pd.DataFrame:
    if CFG is None:
        raise ValueError("Config no cargada.")

    # Filtro macro (tolerante)
    if not _macro_filter_ok():
        print("[INFO] Macro filtro no cumple. No se generan nuevas señales esta semana.")
        cols = [
            "ticker","score_total","score_rs","score_structure","score_volume","score_potential",
            "entry1","entry2","entry3","tp1","tp2","sl",
            "position_size_usd","shares_e1","shares_e2","shares_e3"
        ]
        return pd.DataFrame(columns=cols)

    # Universo
    uni = build_universe(source=source)
    if not uni:
        return pd.DataFrame()

    # Descarga robusta con CACHÉ y REANUDACIÓN
    raw = download_ohlcv_cached_resume(
        uni,
        lookback_years=CFG.data.lookback_years,
        interval="1d"
    )
    if raw.empty:
        return pd.DataFrame()

    # Benchmark
    bench = _get_benchmark_df()
    if bench.empty:
        return pd.DataFrame()
    bench = bench.dropna(subset=["Close"])
    bench.index = pd.to_datetime(bench.index)
    bench = bench.loc[~bench.index.duplicated(keep="last")]

    # Scoring
    rows: List[dict] = []
    for tkr, sub in raw.groupby("Ticker"):
        df = sub.copy().rename(columns=str.capitalize)
        df = df.dropna(subset=["Close"])
        df = df.set_index(pd.to_datetime(df["Date"])).sort_index()
        df = df[["Open","High","Low","Close","Volume"]] if "Volume" in df.columns else df[["Open","High","Low","Close"]]

        common_idx = df.index.intersection(bench.index)
        if len(common_idx) < 120:
            continue
        dfa = df.loc[common_idx]
        ben = bench.loc[common_idx]

        info = check_base_and_breakout(dfa, ben)
        rs_comp = 1.0 if float(info.get("rs_score", 0.0)) >= float(CFG.rules.rs_vs_spy_min) else 0.0
        structure_comp = 1.0 if info.get("is_base", False) else 0.0
        volume_comp = 1.0 if info.get("vol_breakout", False) else 0.0

        resistance = float(info.get("resistance", np.nan))
        if not math.isfinite(resistance) or resistance <= 0:
            continue

        plan = plan_trades(resistance)
        ref_price = float(dfa["Close"].iloc[-1])
        pot_comp = _normalize_potential(
            plan["tp1"], ref_price,
            target_pct=float(CFG.exits.tp1_pct_from_avg_entry)
        )

        w = CFG.ranking
        score_total = (
            (w.rs * rs_comp) +
            (w.structure * structure_comp) +
            (w.volume * volume_comp) +
            (w.potential * pot_comp)
        )

        sizing = plan_position_from_levels(
            capital=float(CFG.backtest.initial_capital),
            entry1=plan["entry1"], entry2=plan["entry2"], entry3=plan["entry3"],
            sl=plan["sl"], tranches_pct=CFG.entries.tranches_pct,
            risk_per_trade=CFG.risk.risk_per_trade,
        )

        rows.append({
            "ticker": tkr,
            "score_total": float(score_total),
            "score_rs": float(rs_comp),
            "score_structure": float(structure_comp),
            "score_volume": float(volume_comp),
            "score_potential": float(pot_comp),
            "resistance": resistance,
            "entry1": plan["entry1"], "entry2": plan["entry2"], "entry3": plan["entry3"],
            "tp1": plan["tp1"], "tp2": plan["tp2"], "sl": plan["sl"],
            "position_size_usd": sizing["capital_required"],
            "shares_e1": sizing["shares_e1"], "shares_e2": sizing["shares_e2"], "shares_e3": sizing["shares_e3"],
        })

    df_rank = pd.DataFrame(rows)
    if df_rank.empty:
        return df_rank
    df_rank = df_rank.sort_values("score_total", ascending=False).head(int(CFG.backtest.topn_max))
    return df_rank.reset_index(drop=True)

def make_telegram_messages(df_rank: pd.DataFrame) -> List[str]:
    if CFG is None or not CFG.telegram.enabled:
        return []
    msgs: List[str] = []
    for _, r in df_rank.iterrows():
        msgs.append(CFG.telegram.template.format(
            ticker=r["ticker"], e1=r["entry1"], e2=r["entry2"], e3=r["entry3"],
            tp1=r["tp1"], tp2=r["tp2"], sl=r["sl"],
        ))
    return msgs
