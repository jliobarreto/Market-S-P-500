# src/config/config.py
"""
Configuración central del proyecto (Settings) y utilidades de logging.

- Lee variables desde entorno y .env (si existe).
- Define rutas por defecto para universo y caché.
- Expone helper get_settings() con caché interna.
- Incluye get_logger() listo para producción/desarrollo.

Requisitos:
    pip install pydantic pydantic-settings python-dotenv (opcional)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    # Fallback mínimo si no está pydantic-settings instalado
    class BaseSettings:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def SettingsConfigDict(**kwargs):  # type: ignore
        return dict(**kwargs)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UNIVERSE_DIR = DATA_DIR / "universe"
CACHE_DIR = DATA_DIR / "cache" / "equities"


class Settings(BaseSettings):
    """
    Ajustes generales del pipeline. Puedes sobreescribirlos con variables
    de entorno o un archivo .env en la raíz del proyecto.
    """

    # --- Universo / Datos
    universe_source: str = str(UNIVERSE_DIR / "sp500.csv")
    timeframe: str = "1d"
    min_history_days: int = 252 * 3
    min_price: float = 5.0
    min_avg_dollar_vol: float = 1_000_000.0

    # --- Descargas / Caché
    cache_dir: str = str(CACHE_DIR)
    yfinance_period: str = "max"
    yfinance_interval: str = "1d"
    force_refresh: bool = False

    # --- Régimen
    use_regime_filter: bool = True

    # --- Scoring
    top_n: int = 25
    breakout_lookback: int = 55
    vol_confirm: bool = True
    vol_window: int = 20
    min_close_buffer: float = 0.0
    atr_window: int = 14
    atr_mult: float = 1.8

    # --- Backtest (valores por defecto si luego los usas)
    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.01
    max_positions: int = 20

    # --- Salidas
    output_dir: str = str(PROJECT_ROOT / "output")
    output_rank_csv: str = "rank_latest.csv"

    # --- Telegram (opcional)
    telegram_enabled: bool = False
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_dry_run: bool = True  # por defecto no envía

    # --- Misc
    random_seed: Optional[int] = 42
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve Settings singleton (caché en memoria)."""
    s = Settings()
    # Crear directorios clave si no existen
    Path(s.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(s.output_dir).mkdir(parents=True, exist_ok=True)
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    return s


def get_logger(name: str = "sp500bot") -> logging.Logger:
    """Logger con formateador legible y nivel desde settings."""
    s = get_settings()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # ya configurado

    level = getattr(logging, s.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
