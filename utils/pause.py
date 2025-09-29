# utils/pause.py
from __future__ import annotations

RATE_HINTS = (
    "JSONDecodeError", "Too Many Requests", "429",
    "Read timed out", "ReadTimeout", "Max retries exceeded", "timed out",
    "Failed to get ticker", "possibly delisted; No price data found"
)

def looks_like_rate_limit(msg: str) -> bool:
    m = (msg or "").lower()
    return any(h.lower() in m for h in RATE_HINTS)
