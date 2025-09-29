from __future__ import annotations
from pathlib import Path
import pandas as pd

from .metrics import compute_portfolio_metrics, per_ticker_metrics


def save_trades_csv(trades: pd.DataFrame, path: str = "storage/history/trades_backtest.csv") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path, index=False)
    return path


def save_metrics_csv(trades: pd.DataFrame, initial_capital: float = 10_000.0, basepath: str = "storage/history/") -> tuple[str, str]:
    Path(basepath).mkdir(parents=True, exist_ok=True)
    port = compute_portfolio_metrics(trades, initial_capital)
    df_port = pd.DataFrame([port])
    path_port = str(Path(basepath) / "metrics_portfolio.csv")
    df_port.to_csv(path_port, index=False)

    df_tkr = per_ticker_metrics(trades, initial_capital)
    path_tkr = str(Path(basepath) / "metrics_by_ticker.csv")
    df_tkr.to_csv(path_tkr)
    return path_port, path_tkr
