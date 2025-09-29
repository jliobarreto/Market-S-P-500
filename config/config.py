# config/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import os

# Carga variables de entorno (.env) si está disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ========================
# CONFIGURACIONES PRINCIPALES
# ========================

@dataclass
class ScheduleConfig:
    # Lunes 2 horas antes de la apertura de NY (solo para programar tarea)
    market_tz: str = "America/New_York"
    market_open_hhmm: str = "09:30"   # Hora de apertura del mercado
    run_hhmm: str = "07:30"           # 2 horas antes de la apertura
    day_of_week: int = 1              # 1=Lunes (para cron/Task Scheduler)

@dataclass
class DataConfig:
    provider: str = "yfinance"
    lookback_years: int = 3
    ohlcv: bool = True

# -------- Baseline de liquidez / universo --------
@dataclass
class LiquidityConfig:
    # Filtros mínimos de liquidez para evitar penny/ilíquidas
    min_price: float = 10.0                   # Precio mínimo para universo inicial
    min_avg_volume: float = 1_500_000         # Acciones/día (media 60d)

@dataclass
class UniverseConfig:
    exchanges: List[str] = field(default_factory=lambda: ["NYSE", "NASDAQ", "AMEX"])  # referencial
    min_market_cap: float = 2_000_000_000     # USD
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)

# -------- Gestión de riesgo / capital --------
@dataclass
class RiskConfig:
    risk_per_trade: float = 0.005      # 0.5% del capital
    max_exposure: float = 0.65         # 65% del capital
    max_open_positions: int = 12

# -------- Plan de entradas y salidas --------
@dataclass
class EntryPlan:
    tranches_pct: List[float] = field(default_factory=lambda: [0.40, 0.35, 0.25])
    spacing_atr: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])

@dataclass
class ExitPlan:
    tp1_pct_from_avg_entry: float = 0.20  # +20%
    tp1_qty_pct: float = 0.30             # 30% de la posición
    tp2_pct_from_avg_entry: float = 0.50  # +50% (o resistencia mayor)
    tp2_qty_pct: float = 0.40
    tp3_trailing: str = "ema50w"          # trailing por EMA50 semanal
    tp3_qty_pct: float = 0.30
    initial_sl_pct_or_struct_max: float = 0.10  # -10% o soporte mayor semanal

# -------- Reglas de señal --------
@dataclass
class SignalRules:
    # Compresión y volumen
    bb_period: int = 20
    bb_std: float = 2.0
    bb_width_pct_max: float = 0.10      # ancho < 10% del precio
    atr_lookback: int = 14
    atr_falling_min_bars: int = 10
    vol_breakout_mult: float = 1.5      # volumen del día >= 1.5x promedio 20
    vol_sma: int = 20

    # Momentum/RS
    rsi_min: int = 55
    rs_vs_spy_min: float = 1.0          # >1 implica mejor que SPY

    # Estructura
    ema_fast: int = 20
    ema_slow: int = 50
    ema_long: int = 100
    need_higher_lows: bool = True

# -------- Filtro macro --------
@dataclass
class MacroFilter:
    require_spy_above_ema50w: bool = True   # Si hay problemas de datos, puedes poner False
    vix_max: float = 25.0                   # Umbral para VIX (solo si se obtiene ^VIX real)

# -------- Ponderaciones del ranking --------
@dataclass
class RankingWeights:
    rs: float = 0.35
    structure: float = 0.30
    volume: float = 0.20
    potential: float = 0.15

# -------- Backtest / Producción --------
@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    reinvest_profits: bool = True
    topn_max: int = 20
    prioritize_by_score: bool = True

# -------- Telegram --------
@dataclass
class TelegramConfig:
    enabled: bool = True
    token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN"))
    chat_id: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    # Mensaje minimalista y accionable
    template: str = (
        "Ticker: {ticker}\n"
        "Plan de entradas:\n"
        " - Entrada 1 (40%): {e1:.2f}\n"
        " - Entrada 2 (35%): {e2:.2f}\n"
        " - Entrada 3 (25%): {e3:.2f}\n\n"
        "Plan de salidas:\n"
        " - TP1 (30%): {tp1:.2f}\n"
        " - TP2 (40%): {tp2:.2f}\n"
        " - TP3 (30%): Trailing Stop EMA50 semanal\n\n"
        "Stop Loss inicial: {sl:.2f}\n"
        "Timeframe: Diario (confirmado en Semanal)"
    )

# ========================
# NUEVO: Registro (lista blanca/negra) y Caché/Resume
# ========================

@dataclass
class RegistryConfig:
    # Archivos CSV para listas persistentes
    enabled_csv: str = "data/enabled.csv"
    disabled_csv: str = "data/disabled.csv"
    # Fallos consecutivos permitidos antes de pasar a lista negra
    max_failures: int = 3
    # Baseline de calidad para permanecer habilitado
    min_price: float = 5.0
    min_avg_dollar_volume: float = 1_000_000.0  # USD/día (precio*volumen 60d)
    lookback_for_filters_days: int = 60
    # Sectores/activos no deseados (opcional, texto libre para filtrar si los etiquetas)
    excluded_industries: List[str] = field(default_factory=list)

@dataclass
class CacheConfig:
    # Caché por ticker para evitar reprocesar y permitir reanudar
    dir: str = "storage/cache_ohlcv"
    freshness_days: int = 3                 # si el parquet < N días, no se re-descarga
    block_size: int = 20                    # tamaño de bloque objetivo (informativo)
    max_consecutive_errors_to_pause: int = 5  # al detectar N errores seguidos → pausar

# ========================
# APP CONFIG (agrega todo)
# ========================

@dataclass
class AppConfig:
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    entries: EntryPlan = field(default_factory=EntryPlan)
    exits: ExitPlan = field(default_factory=ExitPlan)
    rules: SignalRules = field(default_factory=SignalRules)
    macro: MacroFilter = field(default_factory=MacroFilter)
    ranking: RankingWeights = field(default_factory=RankingWeights)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    # NUEVO
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def validate(self) -> None:
        # Entradas
        assert abs(sum(self.entries.tranches_pct) - 1.0) < 1e-6, "Las entradas deben sumar 100%"
        assert len(self.entries.tranches_pct) == len(self.entries.spacing_atr), "Entradas y espaciados deben tener misma longitud"
        # Riesgo
        assert 0.0 < self.risk.risk_per_trade <= 0.02, "Riesgo por trade fuera de rango razonable (0-2%)"
        assert 0.0 < self.risk.max_exposure <= 1.0, "Exposición máxima inválida"
        assert 1 <= self.risk.max_open_positions <= 50, "Número máximo de posiciones inválido"
        # Universo
        assert 0 < self.universe.min_market_cap
        assert self.universe.liquidity.min_price >= 1.0
        assert self.universe.liquidity.min_avg_volume >= 100_000
        # Registro (listas)
        assert self.registry.min_price >= 1.0
        assert self.registry.min_avg_dollar_volume >= 100_000
        assert self.registry.max_failures >= 1
        # Telegram
        if self.telegram.enabled:
            if not self.telegram.token or not self.telegram.chat_id:
                print("[WARN] TELEGRAM_TOKEN o TELEGRAM_CHAT_ID no configurados. Notificaciones deshabilitadas en runtime.")

    def to_dict(self) -> Dict:
        return asdict(self)

# Carga y validación
def load_config() -> AppConfig:
    cfg = AppConfig()
    cfg.validate()
    return cfg

# Uso de ejemplo:
# from config.config import load_config
# CFG = load_config()
# print(CFG.registry.min_avg_dollar_volume)
