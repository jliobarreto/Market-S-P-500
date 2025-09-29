from __future__ import annotations
from typing import Dict

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None


def plan_trades(resistance: float) -> Dict[str, float]:
    """Calcula entradas y TPs escalonados desde la resistencia dada."""
    if CFG is None:
        raise ValueError("Config no cargada.")

    entries = []
    for tranche_pct, spacing in zip(CFG.entries.tranches_pct, CFG.entries.spacing_atr):
        price = resistance * (1.0 + spacing * 0.01)  # Espaciado ATR simulado como %
        entries.append(price)

    tp1 = entries[0] * (1.0 + CFG.exits.tp1_pct_from_avg_entry)
    tp2 = entries[0] * (1.0 + CFG.exits.tp2_pct_from_avg_entry)

    return {
        "entry1": entries[0],
        "entry2": entries[1],
        "entry3": entries[2],
        "tp1": tp1,
        "tp2": tp2,
        "sl": entries[0] * (1.0 - CFG.exits.initial_sl_pct_or_struct_max),
    }
