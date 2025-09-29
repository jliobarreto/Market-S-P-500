from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min())


def equity_curve_from_trades(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.Series:
    if trades.empty:
        return pd.Series([initial_capital])
    # Ordenar por fecha de salida; si falta, usar fecha de señal
    t = trades.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"]).fillna(pd.to_datetime(t["date_signal"]))
    t = t.sort_values("exit_date")

    eq = [initial_capital]
    cur = initial_capital
    for _, r in t.iterrows():
        pnl = (float(r.get("result_pct", 0.0)) / 100.0) * float(r.get("position_size_usd", 0.0))
        cur += pnl
        eq.append(cur)
    return pd.Series(eq)


def compute_portfolio_metrics(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> Dict[str, float]:
    if trades.empty:
        return {"CAGR": 0.0, "MaxDD": 0.0, "WinRate": 0.0, "Expectancy_R": 0.0, "ProfitFactor": 0.0, "Sharpe": 0.0}

    equity = equity_curve_from_trades(trades, initial_capital)
    # CAGR aproximado asumiendo duración ≈ (última salida - primera señal)
    try:
        start = pd.to_datetime(trades["date_signal"]).min()
        end = pd.to_datetime(trades["exit_date"]).max()
        years = max(1e-6, (end - start).days / 365.25)
    except Exception:
        years = 1.0
    CAGR = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1.0

    MaxDD = max_drawdown(equity)

    wins = trades[trades["result_pct"] > 0]
    losses = trades[trades["result_pct"] <= 0]
    WinRate = len(wins) / max(1, len(trades))
    avg_R = trades["result_R"].mean() if "result_R" in trades.columns else 0.0

    gross_profit = wins["result_pct"].sum()
    gross_loss = -losses["result_pct"].sum()
    ProfitFactor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # Sharpe simple con retornos por trade (supone rf=0 y 252 trades ~ días no aplica; es indicativo)
    rets = trades["result_pct"].astype(float) / 100.0
    Sharpe = (rets.mean() / (rets.std(ddof=1) + 1e-9)) * np.sqrt(12)  # anualiza aprox. por meses

    return {
        "CAGR": float(CAGR),
        "MaxDD": float(MaxDD),
        "WinRate": float(WinRate),
        "Expectancy_R": float(avg_R),
        "ProfitFactor": float(ProfitFactor),
        "Sharpe": float(Sharpe),
    }


def per_ticker_metrics(trades: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for tkr, grp in trades.groupby("ticker"):
        m = compute_portfolio_metrics(grp, initial_capital)
        m["ticker"] = tkr
        rows.append(m)
    return pd.DataFrame(rows).set_index("ticker")
