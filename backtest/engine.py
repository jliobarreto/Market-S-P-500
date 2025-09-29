from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import pandas as pd

from signals.rules import check_base_and_breakout
from signals.planner import plan_trades
from risk.sizing import plan_position_from_levels

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None


@dataclass
class Trade:
    ticker: str
    date_signal: pd.Timestamp
    entry1: float
    entry2: float
    entry3: float
    sl_initial: float
    tp1: float
    tp2: float
    # Resultados
    exit_date: Optional[pd.Timestamp] = None
    result_R: Optional[float] = None
    result_pct: Optional[float] = None
    holding_days: Optional[int] = None
    # Tamaños
    shares_e1: float = 0.0
    shares_e2: float = 0.0
    shares_e3: float = 0.0
    entry_avg: float = 0.0
    position_size_usd: float = 0.0


def _first_hit_after(prices: pd.Series, level: float, after_idx: int, direction: str) -> Optional[int]:
    """Devuelve el índice del primer día en que se toca 'level' después de 'after_idx'.
    direction='up' usa High>=level; direction='down' usa Low<=level.
    """
    if direction == "up":
        cond = prices >= level
    else:
        cond = prices <= level
    hit = cond.iloc[after_idx+1:]
    if not hit.any():
        return None
    return hit.idxmax()  # primer True


def simulate_signal(
    tkr: str,
    df: pd.DataFrame,
    bench_df: pd.DataFrame,
    initial_capital: float,
    risk_per_trade: float,
    tranches_pct: List[float],
) -> List[Trade]:
    """Simula entradas/TP/SL por cada señal de ruptura anticipada encontrada en df.
    Asume timeframe diario; evalúa señal en cierre del día y ejecuta niveles a partir del día siguiente.
    """
    trades: List[Trade] = []

    # Iterar por días buscando triggers
    for i in range(max(61, df.index.get_loc(df.index[0]) if len(df) else 0), len(df)-2):
        window = df.iloc[: i+1]
        chk = check_base_and_breakout(window, bench_df)
        if not (chk.get("is_base") and chk.get("trigger")):
            continue

        resistance = chk.get("resistance")
        plan = plan_trades(resistance)

        # Tamaño por riesgo usando entradas planificadas y SL
        sizing = plan_position_from_levels(
            capital=initial_capital,
            entry1=plan["entry1"], entry2=plan["entry2"], entry3=plan["entry3"],
            sl=plan["sl"], tranches_pct=tranches_pct, risk_per_trade=risk_per_trade,
        )

        tr = Trade(
            ticker=tkr,
            date_signal=window.index[-1],
            entry1=float(plan["entry1"]), entry2=float(plan["entry2"]), entry3=float(plan["entry3"]),
            sl_initial=float(plan["sl"]), tp1=float(plan["tp1"]), tp2=float(plan["tp2"]),
            shares_e1=float(sizing["shares_e1"]), shares_e2=float(sizing["shares_e2"]), shares_e3=float(sizing["shares_e3"]),
            entry_avg=float(sizing["entry_avg"]), position_size_usd=float(sizing["capital_required"]),
        )

        # Simulación posterior a la señal: a partir del día siguiente
        after = i
        highs = df["High"].astype(float)
        lows = df["Low"].astype(float)

        # Orden de chequeo: SL primero (protección), luego TP1, luego TP2
        sl_idx = _first_hit_after(lows, tr.sl_initial, after, direction="down")
        tp1_idx = _first_hit_after(highs, tr.tp1, after, direction="up")
        tp2_idx = _first_hit_after(highs, tr.tp2, after, direction="up")

        # Resolver qué ocurre primero en el tiempo
        candidates: List[Tuple[str, Optional[pd.Timestamp]]] = []
        if sl_idx is not None:
            candidates.append(("SL", sl_idx))
        if tp1_idx is not None:
            candidates.append(("TP1", tp1_idx))
        if tp2_idx is not None:
            candidates.append(("TP2", tp2_idx))

        if not candidates:
            # No se alcanzó ni SL ni TP en el horizonte; cerrar al último precio
            last_close = float(df["Close"].iloc[-1])
            pct = (last_close / tr.entry_avg) - 1.0
            R = pct / ( (tr.entry_avg - tr.sl_initial) / tr.entry_avg ) if tr.entry_avg>0 else 0.0
            tr.exit_date = df.index[-1]
            tr.result_pct = pct * 100.0
            tr.result_R = R
            tr.holding_days = int((tr.exit_date - tr.date_signal).days)
            trades.append(tr)
            continue

        # Elegir el primer evento en el tiempo
        kind, when = min(candidates, key=lambda kv: df.index.get_loc(kv[1]))

        if kind == "SL":
            exit_price = tr.sl_initial
            pct = (exit_price / tr.entry_avg) - 1.0
            R = -1.0 * (tr.entry_avg - tr.sl_initial) / (tr.entry_avg - tr.sl_initial)  # -1R
            tr.exit_date = when
            tr.result_pct = pct * 100.0
            tr.result_R = -1.0
            tr.holding_days = int((tr.exit_date - tr.date_signal).days)
            trades.append(tr)
            continue

        # Si toca TP1 primero
        remaining_weight = 1.0 - 0.30  # vendimos 30%
        # ¿Luego toca TP2 o SL?
        after_tp1_idx = df.index.get_loc(when)
        sl_after_tp1 = _first_hit_after(lows, tr.sl_initial, after_tp1_idx, direction="down")
        tp2_after_tp1 = _first_hit_after(highs, tr.tp2, after_tp1_idx, direction="up")

        if tp2_after_tp1 is not None and (sl_after_tp1 is None or df.index.get_loc(tp2_after_tp1) < df.index.get_loc(sl_after_tp1)):
            # Toca TP2 después de TP1 → cerramos 40% adicional; el resto queda para trailing (no simulado, cerramos a TP2 para simplificar)
            avg_exit = (0.30 * tr.tp1) + (0.40 * tr.tp2) + (0.30 * tr.tp2)
            pct = (avg_exit / tr.entry_avg) - 1.0
            R = pct / ((tr.entry_avg - tr.sl_initial) / tr.entry_avg)
            tr.exit_date = tp2_after_tp1
            tr.result_pct = pct * 100.0
            tr.result_R = R
            tr.holding_days = int((tr.exit_date - tr.date_signal).days)
            trades.append(tr)
        else:
            # Toca SL después de TP1 → ganancias parciales pero cierra por SL
            avg_exit = (0.30 * tr.tp1) + (0.70 * tr.sl_initial)
            pct = (avg_exit / tr.entry_avg) - 1.0
            R = pct / ((tr.entry_avg - tr.sl_initial) / tr.entry_avg)
            tr.exit_date = sl_after_tp1 if sl_after_tp1 is not None else df.index[-1]
            tr.result_pct = pct * 100.0
            tr.result_R = R
            tr.holding_days = int((tr.exit_date - tr.date_signal).days)
            trades.append(tr)

        # Saltar unas barras para evitar señales duplicadas muy cercanas
        # (opcional) i = max(i, df.index.get_loc(tr.exit_date))

    return trades


def run_backtest(
    prices: Dict[str, pd.DataFrame],
    bench: pd.DataFrame,
    initial_capital: Optional[float] = None,
) -> pd.DataFrame:
    """Ejecuta backtest simple sobre varios tickers. Retorna DataFrame de trades."""
    if CFG is None:
        raise ValueError("Config no cargada.")

    capital = initial_capital if initial_capital is not None else CFG.backtest.initial_capital
    risk = CFG.risk.risk_per_trade
    tranches = CFG.entries.tranches_pct

    rows: List[dict] = []
    for tkr, df in prices.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        trades = simulate_signal(
            tkr,
            df=df,
            bench_df=bench,
            initial_capital=capital,
            risk_per_trade=risk,
            tranches_pct=tranches,
        )
        for tr in trades:
            rows.append({
                "ticker": tr.ticker,
                "date_signal": tr.date_signal.date(),
                "entry1": tr.entry1,
                "entry2": tr.entry2,
                "entry3": tr.entry3,
                "sl_initial": tr.sl_initial,
                "tp1": tr.tp1,
                "tp2": tr.tp2,
                "entry_avg": tr.entry_avg,
                "position_size_usd": tr.position_size_usd,
                "result_R": tr.result_R,
                "result_pct": tr.result_pct,
                "exit_date": tr.exit_date.date() if tr.exit_date else None,
                "holding_days": tr.holding_days,
            })

    return pd.DataFrame(rows)
