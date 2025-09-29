from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None


@dataclass
class Position:
    ticker: str
    shares: float
    entry_avg: float
    stop: float
    notional: float


@dataclass
class PortfolioState:
    initial_capital: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def equity(self) -> float:
        # Para selección previa a ejecución asumimos notional = entry_avg * shares
        invested = sum(p.notional for p in self.positions.values())
        return self.cash + invested

    def exposure(self) -> float:
        invested = sum(p.notional for p in self.positions.values())
        return invested / max(1e-6, self.initial_capital)

    def can_add(self, notional_new: float, max_exposure: float, max_positions: int) -> bool:
        if len(self.positions) >= max_positions:
            return False
        future_invested = sum(p.notional for p in self.positions.values()) + notional_new
        return (future_invested / max(1e-6, self.initial_capital)) <= max_exposure


def select_candidates(
    state: PortfolioState,
    candidates: List[Tuple[str, float, float]],
    # lista de tuplas: (ticker, score, capital_required)
    max_exposure: float,
    max_positions: int,
) -> List[Tuple[str, float, float]]:
    """Ordena por score (desc) y toma solo los que caben por exposición y cupo de posiciones."""
    out: List[Tuple[str, float, float]] = []
    for tkr, score, cap_req in sorted(candidates, key=lambda x: x[1], reverse=True):
        if state.can_add(notional_new=cap_req, max_exposure=max_exposure, max_positions=max_positions):
            out.append((tkr, score, cap_req))
            # Simular adición para las siguientes decisiones
            state.positions[tkr] = Position(ticker=tkr, shares=0.0, entry_avg=0.0, stop=0.0, notional=cap_req)
        if len(out) >= max_positions:
            break
    return out
