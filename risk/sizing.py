from __future__ import annotations
from math import floor
from typing import Dict, Optional

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None


def position_size_by_risk(
    capital: float,
    entry_avg: float,
    stop_price: float,
    risk_per_trade: Optional[float] = None,
    min_shares: int = 1,
) -> Dict[str, float]:
    """Calcula tamaño de posición (número de acciones) para que la pérdida máxima si toca el SL
    sea igual a capital * risk_per_trade.
    Retorna dict con: shares, risk_dollars, per_share_risk, notional.
    """
    assert capital > 0 and entry_avg > 0 and stop_price > 0
    rpt = risk_per_trade if risk_per_trade is not None else (CFG.risk.risk_per_trade if CFG else 0.005)
    per_share_risk = max(1e-6, entry_avg - stop_price)
    risk_dollars = capital * float(rpt)
    shares = floor(risk_dollars / per_share_risk)
    shares = max(min_shares, shares)
    notional = shares * entry_avg
    return {
        "shares": float(shares),
        "risk_dollars": float(risk_dollars),
        "per_share_risk": float(per_share_risk),
        "notional": float(notional),
    }


def distribute_tranches_shares(total_shares: float, tranches_pct: list[float]) -> Dict[str, float]:
    """Distribuye el total de acciones entre tramos según porcentajes.
    Ajusta para que la suma entera coincida con el total usando redondeo conservador.
    """
    ints = [floor(total_shares * p) for p in tranches_pct]
    diff = int(total_shares) - sum(ints)
    # Repartir las acciones faltantes comenzando por el mayor porcentaje
    order = sorted(range(len(tranches_pct)), key=lambda i: tranches_pct[i], reverse=True)
    j = 0
    while diff > 0 and j < len(order):
        ints[order[j]] += 1
        diff -= 1
        j = (j + 1) % len(order)
    return {f"shares_tranche_{i+1}": float(n) for i, n in enumerate(ints)}


def plan_position_from_levels(
    capital: float,
    entry1: float,
    entry2: float,
    entry3: float,
    sl: float,
    tranches_pct: Optional[list[float]] = None,
    risk_per_trade: Optional[float] = None,
) -> Dict[str, float]:
    """Calcula tamaño total por riesgo y distribuye en 3 tramos.
    Usa el promedio ponderado de entradas para el cálculo del riesgo.
    """
    tpc = tranches_pct or (CFG.entries.tranches_pct if CFG else [0.4, 0.35, 0.25])
    assert abs(sum(tpc) - 1.0) < 1e-6 and len(tpc) == 3

    entry_avg = (entry1 * tpc[0]) + (entry2 * tpc[1]) + (entry3 * tpc[2])
    base = position_size_by_risk(
        capital=capital,
        entry_avg=entry_avg,
        stop_price=sl,
        risk_per_trade=risk_per_trade,
    )
    dist = distribute_tranches_shares(base["shares"], tpc)

    out = {
        **base,
        "entry_avg": float(entry_avg),
        "shares_e1": dist["shares_tranche_1"],
        "shares_e2": dist["shares_tranche_2"],
        "shares_e3": dist["shares_tranche_3"],
        "capital_required": float(base["shares"] * entry_avg),
    }
    return out
