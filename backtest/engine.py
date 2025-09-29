# src/backtest/engine.py
"""
Motor de backtesting (long-only) para universo S&P 500.

Características:
- Entradas por breakout Donchian (N-máximos, sin look-ahead) + confirmación opcional por volumen.
- Stop ATR inicial + trailing stop ATR opcional.
- Take-profits por múltiplos de R (con opción de salida escalonada).
- Dimensionamiento por riesgo (% del equity por trade).
- Filtro de régimen diario (BULL/NEUTRAL/BEAR) a partir de SPY (SMA200 + ROC 6m).

Entradas esperadas:
- prices: dict[str, pd.DataFrame] con columnas en minúsculas: date, open, high, low, close, volume
- spy: pd.DataFrame con columnas: date, open, high, low, close, volume (para régimen)

Salidas:
- dict con:
    - "metrics": pd.Series
    - "equity_curve": pd.DataFrame [date, equity, cash, exposure]
    - "trades": pd.DataFrame (log de operaciones)
    - "daily_positions": pd.DataFrame (opcional, peso por ticker)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Configuración del backtest
# =========================

@dataclass(frozen=True)
class BacktestConfig:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_equity: float = 100_000.0

    # Señal de entrada (breakout)
    breakout_lookback: int = 55
    vol_confirm: bool = True
    vol_window: int = 20
    min_close_buffer: float = 0.0  # p.ej. 0.001 = 0.1%

    # Stops / TPs
    atr_window: int = 14
    stop_atr_mult: float = 1.8
    trailing_atr_mult: Optional[float] = 2.5  # None = sin trailing
    take_profit_multiples: Tuple[float, ...] = (1.5, 2.5)
    scale_out: bool = True  # True: salida escalonada igual por TP; False: salida total al primer TP

    # Sizing y cartera
    risk_per_trade: float = 0.01  # 1% del equity
    max_positions: int = 20       # número máximo de posiciones simultáneas
    capital_usage_limit: float = 0.95  # % máximo de equity invertido

    # Costos de transacción
    slippage_bps: float = 5.0          # 5 bps = 0.05%
    commission_per_share: float = 0.0  # comisión plana por acción

    # Filtros básicos
    min_price: float = 5.0
    min_avg_dollar_vol: float = 1_000_000.0

    # Régimen
    use_regime_filter: bool = True
    regime_bull_min_score: float = 0.67  # BULL si score >= 0.67; NEUTRAL si >= 0.40; BEAR si < 0.40


# =========================
# Utilidades de indicadores
# =========================

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    return out.sort_values("date").reset_index(drop=True)

def _require(df: pd.DataFrame, cols: Iterable[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Faltan columnas {miss}. Presentes: {list(df.columns)}")

def _atr_series(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return atr

def _donchian_high_prev(high: pd.Series, lookback: int) -> pd.Series:
    """
    Serie del máx N excluyendo la barra actual: max(high[t-N:t]) para cada t.
    """
    # rolling(max) incluye la barra actual, así que desplazamos 1.
    return high.rolling(lookback + 1, min_periods=lookback + 1).max().shift(1)

def _volume_z_series(vol: pd.Series, window: int) -> pd.Series:
    mean = vol.rolling(window, min_periods=window).mean()
    std = vol.rolling(window, min_periods=window).std(ddof=0)
    z = (vol - mean) / std
    return z

def _realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    rets = close.pct_change()
    return rets.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252)

def _regime_score_spy(spy: pd.DataFrame) -> pd.Series:
    """
    Score [0..1] diario a partir de SPY:
    - Tendencia: close > SMA50/100/200 (0..1)
    - Momentum: ROC 3/6/12m > 0 (0..1)
    - Penalización por vol (suave)
    """
    c = spy["close"].astype(float)
    sma50 = c.rolling(50, min_periods=50).mean()
    sma100 = c.rolling(100, min_periods=100).mean()
    sma200 = c.rolling(200, min_periods=200).mean()
    trend = ((c > sma50).astype(float) + (c > sma100).astype(float) + (c > sma200).astype(float)) / 3.0

    roc3 = c.pct_change(63)
    roc6 = c.pct_change(126)
    roc12 = c.pct_change(252)
    mom = ((roc3 > 0).astype(float) + (roc6 > 0).astype(float) + (roc12 > 0).astype(float)) / 3.0

    vol20 = _realized_vol(c, 20)
    vol_pen = pd.Series(0.0, index=c.index)
    vol_pen = vol_pen.mask(vol20.between(0.15, 0.25), 0.10)
    vol_pen = vol_pen.mask(vol20.between(0.25, 0.35), 0.20)
    vol_pen = vol_pen.mask(vol20 > 0.35, 0.30)

    score = 0.5 * trend + 0.35 * mom - 0.15 * (vol_pen / 0.30)  # normaliza penalización
    return score.clip(0, 1)

def _label_regime(score: float) -> str:
    if score >= 0.67:
        return "BULL"
    if score >= 0.40:
        return "NEUTRAL"
    return "BEAR"


# =========================
# Preparación de datos
# =========================

def _prepare_prices(prices: Dict[str, pd.DataFrame], cfg: BacktestConfig) -> Dict[str, pd.DataFrame]:
    out = {}
    for t, df in prices.items():
        if df is None or df.empty:
            continue
        df = _norm(df)
        _require(df, ["date", "open", "high", "low", "close", "volume"])
        if cfg.start_date:
            df = df[df["date"] >= cfg.start_date]
        if cfg.end_date:
            df = df[df["date"] <= cfg.end_date]
        if df.empty:
            continue
        out[t] = df.reset_index(drop=True)
    return out

def _common_calendar(prepped: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    # Unimos todas las fechas y usamos sólo las únicas ordenadas
    all_dates: List[pd.Timestamp] = []
    for df in prepped.values():
        all_dates.append(pd.to_datetime(df["date"]))
    if not all_dates:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(sorted(pd.unique(pd.concat(all_dates))))
    return idx


# =========================
# Señal vectorizada (por ticker)
# =========================

def _compute_signal_table(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas:
    date, open, high, low, close, volume, atr, high_n_prev, vol_z, is_breakout, entry_price, stop_init
    """
    d = df.copy()
    c = d["close"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)
    v = d["volume"].astype(float)
    o = d["open"].astype(float)

    atr = _atr_series(d, cfg.atr_window)
    high_prev = _donchian_high_prev(h, cfg.breakout_lookback)
    vol_z = _volume_z_series(v, cfg.vol_window)
    buffer_level = high_prev * (1 + cfg.min_close_buffer)
    is_breakout = c > buffer_level

    entry_price = c  # entramos a cierre (se ejecutará al open del día siguiente)
    stop_init = entry_price - cfg.stop_atr_mult * atr

    out = pd.DataFrame({
        "date": d["date"],
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "atr": atr,
        "high_n_prev": high_prev,
        "vol_z": vol_z,
        "is_breakout": is_breakout,
        "entry_price": entry_price,
        "stop_init": stop_init,
    })
    # Confirmación de volumen
    if cfg.vol_confirm:
        out["vol_ok"] = (out["vol_z"] > 1.0)
    else:
        out["vol_ok"] = True
    return out


# =========================
# Simulación por fechas
# =========================

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_px: float
    shares: int
    stop: float
    atr: float
    risk_R: float  # $ por acción = entry_px - stop_init
    scale_left: float  # fracción restante de la posición (1.0 al inicio)

def _apply_costs(px: float, cfg: BacktestConfig, side: str = "buy") -> float:
    # Slippage en bps: sube precio si compras, baja si vendes
    slip = px * (cfg.slippage_bps / 10_000.0)
    px_adj = px + slip if side == "buy" else px - slip
    return px_adj

def _size_shares(equity: float, entry_px: float, stop_px: float, cfg: BacktestConfig) -> int:
    risk_dollars = equity * cfg.risk_per_trade
    risk_per_share = max(entry_px - stop_px, 1e-6)
    raw_shares = risk_dollars / risk_per_share
    shares = int(np.floor(raw_shares))
    return max(shares, 0)

def _tp_levels(entry_px: float, stop_px: float, multiples: Tuple[float, ...]) -> List[float]:
    R = entry_px - stop_px
    return [entry_px + m * R for m in multiples]

def _exit_reason_name(code: str) -> str:
    mapping = {
        "STOP": "Stop",
        "TP": "Take-Profit",
        "TRAIL": "Trailing Stop",
        "EOT": "Fin backtest"
    }
    return mapping.get(code, code)


# =========================
# Métricas
# =========================

def _metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    eq = equity_curve["equity"].astype(float)
    rets = eq.pct_change().fillna(0.0)
    days = len(eq)
    if days <= 1:
        days = 1
    years = days / 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0

    # Max drawdown
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1.0)
    maxdd = dd.min()

    # Sharpe/Sortino (diario -> anualizado)
    sharpe = (rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(252) if rets.std(ddof=0) > 0 else 0.0
    downside = rets.copy()
    downside[downside > 0] = 0
    sortino = (rets.mean() / (downside.std(ddof=0) + 1e-12)) * np.sqrt(252) if downside.std(ddof=0) > 0 else 0.0

    # Trade stats
    if trades.empty:
        hit_rate = 0.0
        pf = 0.0
        avg_win = 0.0
        avg_loss = 0.0
    else:
        wins = trades[trades["pnl"] > 0]["pnl"]
        losses = trades[trades["pnl"] < 0]["pnl"]
        hit_rate = len(wins) / max(len(trades), 1)
        pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0

    return pd.Series({
        "CAGR": round(float(cagr), 4),
        "MaxDD": round(float(maxdd), 4),
        "Sharpe": round(float(sharpe), 3),
        "Sortino": round(float(sortino), 3),
        "HitRate": round(float(hit_rate), 3),
        "ProfitFactor": round(float(pf), 3),
        "AvgWin": round(float(avg_win), 2),
        "AvgLoss": round(float(avg_loss), 2),
        "FinalEquity": round(float(eq.iloc[-1]), 2),
        "Trades": int(len(trades)),
    })


# =========================
# Motor principal
# =========================

def run(
    prices: Dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Ejecuta el backtest y devuelve métricas, curva de equity y trades.
    """
    # Preparar datos
    prepped = _prepare_prices(prices, cfg)
    if not prepped:
        return {"metrics": pd.Series(dtype=float), "equity_curve": pd.DataFrame(), "trades": pd.DataFrame()}

    spy = _norm(spy)
    _require(spy, ["date", "open", "high", "low", "close", "volume"])
    if cfg.start_date:
        spy = spy[spy["date"] >= cfg.start_date]
    if cfg.end_date:
        spy = spy[spy["date"] <= cfg.end_date]
    spy = spy.reset_index(drop=True)
    if spy.empty:
        raise ValueError("SPY vacío tras filtros de fecha.")

    # Calendario maestro
    calendar = _common_calendar(prepped)
    if len(calendar) < 200:
        raise ValueError("Calendario muy corto para un backtest útil.")

    # Régimen diario
    regime_score = _regime_score_spy(spy).reindex(calendar).ffill()
    regime_label = regime_score.apply(_label_regime) if cfg.use_regime_filter else pd.Series("BULL", index=calendar)

    # Precalcular señales por ticker
    signals: Dict[str, pd.DataFrame] = {}
    for t, df in prepped.items():
        tab = _compute_signal_table(df, cfg)
        tab.index = pd.to_datetime(tab["date"])
        signals[t] = tab.reindex(calendar)

    # Estado de la simulación
    equity = cfg.initial_equity
    cash = cfg.initial_equity
    exposure = 0.0
    positions: Dict[str, Position] = {}
    equity_rows = []
    trade_rows = []

    # Simulación día a día
    for i, date in enumerate(calendar):
        if i == 0:
            equity_rows.append({"date": date, "equity": equity, "cash": cash, "exposure": exposure})
            continue

        today = date
        yday = calendar[i - 1]

        # 1) Actualizar stops (trailing)
        to_close: List[Tuple[str, str, float]] = []  # (ticker, reason, exit_price)
        for t, pos in list(positions.items()):
            row_prev = signals[t].loc[yday]
            row = signals[t].loc[today]
            if row.isna().any() or row_prev.isna().any():
                continue

            # trailing stop
            new_stop = pos.stop
            if cfg.trailing_atr_mult is not None and not np.isnan(row["atr"]):
                trail_stop = row["close"] - cfg.trailing_atr_mult * row["atr"]
                new_stop = max(pos.stop, trail_stop)

            positions[t].stop = float(new_stop)

            # Evaluar salidas por stop/TP con datos de hoy (se ejecuta a close de hoy, usando high/low de hoy)
            # Orden de prioridad: STOP (riesgo), luego TPs
            # Precio de ejecución con slippage
            exit_px = None
            reason = None

            low_t = float(row["low"])
            high_t = float(row["high"])

            # STOP
            if low_t <= positions[t].stop:
                # Ejecuta al stop (con slippage a la baja por venta)
                exit_px = _apply_costs(positions[t].stop, cfg, side="sell")
                reason = "STOP"

            # TPs (si no hubo stop)
            if exit_px is None and cfg.take_profit_multiples:
                levels = _tp_levels(pos.entry_px, pos.entry_px - (pos.entry_px - pos.stop), cfg.take_profit_multiples)
                # vender por escalones o todo al primer TP tocado
                for j, lvl in enumerate(levels):
                    if high_t >= lvl and pos.scale_left > 0:
                        sell_frac = (1.0 / len(levels)) if cfg.scale_out else 1.0
                        sell_shares = int(np.floor(pos.shares * sell_frac))
                        sell_shares = max(sell_shares, 0)
                        if sell_shares > 0:
                            px_exec = _apply_costs(lvl, cfg, side="sell")
                            pnl = (px_exec - pos.entry_px) * sell_shares - cfg.commission_per_share * sell_shares
                            cash += pnl + px_exec * sell_shares
                            pos.shares -= sell_shares
                            pos.scale_left = max(0.0, pos.scale_left - sell_frac)
                    # si no hay scale-out y se ejecutó el primer TP, cerramos de golpe
                    if not cfg.scale_out and high_t >= lvl:
                        exit_px = _apply_costs(lvl, cfg, side="sell")
                        reason = "TP"
                        break

            # Cierre por trailing explícito (si nos quedamos sin shares por scale-out, se considera trade cerrado)
            if pos.shares <= 0 and reason is None:
                reason = "TP"  # todo vendido por TPs
                exit_px = None  # ya registrado arriba por escalones

            if reason in ("STOP", "TP"):
                if pos.shares > 0 and exit_px is not None:
                    # venta del remanente
                    pnl = (exit_px - pos.entry_px) * pos.shares - cfg.commission_per_share * pos.shares
                    cash += pnl + exit_px * pos.shares
                    trade_rows.append({
                        "ticker": t,
                        "entry_date": pos.entry_date,
                        "exit_date": today,
                        "entry": pos.entry_px,
                        "exit": exit_px,
                        "shares": pos.shares,
                        "pnl": round(float(pnl), 2),
                        "exit_reason": _exit_reason_name(reason),
                    })
                elif pos.shares == 0:
                    # ya registramos PnL por escalones; agregamos fila resumen (opcional)
                    pass
                positions.pop(t, None)

        # 2) Generar nuevas entradas (al open de hoy) si hay capacidad y régimen lo permite
        allow_entries = True
        if cfg.use_regime_filter:
            reg = regime_label.loc[today]
            allow_entries = reg in ("BULL", "NEUTRAL")  # evitar entrar en BEAR

        # Libertad de capital y slots
        slots_left = cfg.max_positions - len(positions)
        capital_in_use = sum(signals[t].loc[today]["close"] * p.shares for t, p in positions.items() if not np.isnan(signals[t].loc[today]["close"]))
        capital_room = max(cfg.capital_usage_limit * equity - capital_in_use, 0.0)

        if allow_entries and slots_left > 0 and capital_room > 0:
            # Candidatos: tickers con breakout ayer (se ejecuta hoy al open)
            cands: List[Tuple[str, float]] = []
            for t, tab in signals.items():
                row_y = tab.loc[yday]
                if row_y.isna().any():
                    continue
                cond = bool(row_y["is_breakout"]) and bool(row_y["vol_ok"]) and (row_y["close"] >= cfg.min_price)
                # Liquidez
                dv20 = (tab["close"] * tab["volume"]).rolling(20, min_periods=20).mean().loc[yday]
                if pd.isna(dv20) or dv20 < cfg.min_avg_dollar_vol:
                    cond = False
                if cond and not np.isnan(row_y["atr"]) and row_y["atr"] > 0:
                    cands.append((t, float(row_y["entry_price"])))

            # Orden simple por dollar-volume descendente (priorizar liquidez)
            if cands:
                dv_map = {t: float((signals[t]["close"] * signals[t]["volume"]).rolling(20, min_periods=20).mean().loc[yday]) for t, _ in cands}
                cands.sort(key=lambda kv: dv_map.get(kv[0], 0.0), reverse=True)

            for t, entry_ref in cands[: max(slots_left, 0)]:
                row_y = signals[t].loc[yday]
                row_t = signals[t].loc[today]
                if row_t.isna().any() or row_y.isna().any():
                    continue

                entry_px = _apply_costs(float(row_t["open"]), cfg, side="buy")  # ejecución al open de hoy
                stop_init = float(row_y["stop_init"])
                if np.isnan(stop_init) or entry_px <= 0 or entry_px <= stop_init:
                    continue

                # Dimensionamiento
                shares = _size_shares(equity, entry_px, stop_init, cfg)
                if shares <= 0:
                    continue

                # Respetar límite de capital
                needed_cap = entry_px * shares
                if needed_cap > capital_room:
                    # reduce shares para encajar
                    shares = int(np.floor(capital_room / max(entry_px, 1e-6)))
                    if shares <= 0:
                        continue
                    needed_cap = entry_px * shares

                # Abrir posición
                cash -= needed_cap + cfg.commission_per_share * shares
                positions[t] = Position(
                    ticker=t,
                    entry_date=today,
                    entry_px=float(entry_px),
                    shares=int(shares),
                    stop=float(stop_init),
                    atr=float(row_y["atr"]),
                    risk_R=float(entry_px - stop_init),
                    scale_left=1.0,
                )

        # 3) Recalcular equity (mark-to-market a close de hoy)
        mtm = sum(float(signals[t].loc[today]["close"]) * p.shares for t, p in positions.items() if not np.isnan(signals[t].loc[today]["close"]))
        equity = cash + mtm
        exposure = mtm / equity if equity > 0 else 0.0
        equity_rows.append({"date": today, "equity": equity, "cash": cash, "exposure": exposure})

    # Cerrar posiciones remanentes al final
    last_date = calendar[-1]
    for t, p in positions.items():
        last_close = float(signals[t].loc[last_date]["close"])
        exit_px = _apply_costs(last_close, cfg, side="sell")
        pnl = (exit_px - p.entry_px) * p.shares - cfg.commission_per_share * p.shares
        cash += pnl + exit_px * p.shares
        trade_rows.append({
            "ticker": t,
            "entry_date": p.entry_date,
            "exit_date": last_date,
            "entry": p.entry_px,
            "exit": exit_px,
            "shares": p.shares,
            "pnl": round(float(pnl), 2),
            "exit_reason": _exit_reason_name("EOT"),
        })

    # Equity final
    equity = cash
    equity_rows[-1]["equity"] = equity
    equity_rows[-1]["cash"] = cash

    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = _metrics(equity_curve, trades, cfg)

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        # "daily_positions": ... (opcional si luego quieres registrar pesos diarios por ticker)
    }


# ======================
# Ejecución de ejemplo
# ======================

if __name__ == "__main__":
    # Demo rápida: requiere tus módulos loader/universe
    try:
        from data.loader import fetch_universe, fetch_history
        from universe.sp500 import load_sp500

        uni = load_sp500()[:50]  # muestra para demo
        prices = fetch_universe(uni, period="10y")
        spy = fetch_history("SPY", period="10y")

        cfg = BacktestConfig(
            start_date=None,
            end_date=None,
            initial_equity=100_000.0,
            breakout_lookback=55,
            vol_confirm=True,
            vol_window=20,
            min_close_buffer=0.001,
            atr_window=14,
            stop_atr_mult=1.8,
            trailing_atr_mult=2.5,
            take_profit_multiples=(1.5, 2.5),
            scale_out=True,
            risk_per_trade=0.01,
            max_positions=20,
            capital_usage_limit=0.95,
            slippage_bps=5.0,
            commission_per_share=0.0,
            min_price=5.0,
            min_avg_dollar_vol=1_000_000.0,
            use_regime_filter=True,
        )

        result = run(prices, spy, cfg)
        print("=== Métricas ===")
        print(result["metrics"])
        print("\n=== Trades (head) ===")
        print(result["trades"].head())
        print("\n=== Equity (tail) ===")
        print(result["equity_curve"].tail())
    except Exception as e:
        print("Ejecuta este módulo dentro del proyecto con loader/universe disponibles.")
        print("Detalle:", e)
