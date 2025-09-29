# src/main.py
"""
Orquestador principal del pipeline S&P 500.

Flujo por defecto:
  1) Cargar/generar universo (S&P 500)
  2) Descargar/cargar datos (caché parquet)
  3) Calcular régimen (SPY/QQQ)
  4) Rankear universo (score 0..100)
  5) Guardar CSV en output y (opcional) enviar a Telegram

Uso:
  python -m src.main --limit 50 --send-telegram
  python -m src.main --refresh-universe

Parámetros clave (ver --help):
  --limit              Limita número de tickers (útil para pruebas)
  --refresh-universe   Regenera data/universe/sp500.csv desde Wikipedia
  --no-telegram        Fuerza no enviar por Telegram
  --no-cache-refresh   No fuerza refresco de yfinance (usa parquet)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from config.config import get_logger, get_settings
from universe.sp500 import load_sp500, load_sp500_with_metadata, refresh_sp500_csv
from data.loader import fetch_universe, fetch_history
from regime.detector import compute_regime, regime_summary
from scoring.ranker import rank
from notify.telegram import TelegramClient, TelegramConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline S&P 500 (rank + notify)")
    p.add_argument("--limit", type=int, default=None, help="Limitar tickers del universo (debug)")
    p.add_argument("--refresh-universe", action="store_true", help="Refrescar sp500.csv desde Wikipedia")
    p.add_argument("--no-telegram", action="store_true", help="Desactivar envío Telegram (independiente de Settings)")
    p.add_argument("--no-cache-refresh", action="store_true", help="No forzar refresh de datos (usar caché)")
    p.add_argument("--output", type=str, default=None, help="Ruta CSV de salida (opcional)")
    return p.parse_args()


def _maybe_refresh_universe(logger) -> None:
    logger.info("Actualizando universo S&P 500 desde Wikipedia…")
    df = refresh_sp500_csv()
    logger.info("Universo actualizado: %d tickers", len(df))


def _init_telegram():
    s = get_settings()
    cfg = TelegramConfig(
        bot_token=s.telegram_bot_token,
        default_chat_id=s.telegram_chat_id,
        parse_mode="MarkdownV2",
        disable_web_page_preview=True,
        dry_run=s.telegram_dry_run or not s.telegram_enabled,
    )
    return TelegramClient(cfg)


def main() -> int:
    s = get_settings()
    logger = get_logger()

    args = _parse_args()

    if args.refresh_universe:
        _maybe_refresh_universe(logger)

    # 1) Universo
    meta = load_sp500_with_metadata()
    universe = meta["ticker"].tolist()
    if args.limit:
        universe = universe[: args.limit]
        logger.warning("Usando universo limitado (%d tickers) para pruebas.", len(universe))

    logger.info("Universo listo: %d tickers", len(universe))

    # 2) Datos de mercado
    logger.info("Descargando/cargando datos del universo…")
    prices: Dict[str, pd.DataFrame] = fetch_universe(
        universe,
        period=s.yfinance_period,
        interval=s.yfinance_interval,
        force_refresh=(s.force_refresh and not args.no_cache_refresh),
    )
    logger.info("Datos cargados: %d tickers con histórico.", len(prices))

    # SPY/QQQ para régimen
    spy = fetch_history("SPY", period=s.yfinance_period, interval=s.yfinance_interval, force_refresh=(s.force_refresh and not args.no_cache_refresh))
    qqq = fetch_history("QQQ", period=s.yfinance_period, interval=s.yfinance_interval, force_refresh=(s.force_refresh and not args.no_cache_refresh))

    # 3) Régimen
    state = compute_regime(spy, qqq)
    logger.info(regime_summary(state))

    # 4) Ranking
    logger.info("Calculando ranking…")
    ranked = rank(
        prices=prices,
        spy_prices=spy,
        regime=state if s.use_regime_filter else None,
        breakout_lookback=s.breakout_lookback,
        vol_confirm=s.vol_confirm,
        vol_window=s.vol_window,
        min_close_buffer=s.min_close_buffer,
        atr_window=s.atr_window,
        atr_mult=s.atr_mult,
        min_price=s.min_price,
        min_avg_dollar_vol=s.min_avg_dollar_vol,
        sector_map=meta[["ticker", "sector"]],
    )

    logger.info("Ranking completado. %d filas.", len(ranked))

    # 5) Guardar CSV
    output_dir = Path(s.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output) if args.output else (output_dir / s.output_rank_csv)
    ranked.to_csv(out_csv, index=False)
    logger.info("Archivo guardado: %s", out_csv)

    # 6) Telegram (opcional)
    if s.telegram_enabled and not args.no_telegram:
        try:
            logger.info("Enviando resumen a Telegram…")
            tg = _init_telegram()
            # Resumen de régimen
            tg.send_regime_summary(
                label=state.label,
                confidence=state.confidence,
                risk_multiplier=state.risk_multiplier,
                asof=str(state.asof.date()),
            )
            # Top-N como CSV
            caption = (
                "Top por *score* (S&P 500)\n"
                f"Régimen: *{state.label}* `{state.confidence:.2f}`  Riesgo `x{state.risk_multiplier:.2f}`"
            )
            tg.send_rank_table(ranked, top_n=s.top_n, filename="rank_top.csv", caption=caption)
            logger.info("Notificaciones enviadas.")
        except Exception as e:
            logger.exception("Error enviando a Telegram: %s", e)
    else:
        logger.info("Telegram desactivado (settings o flag).")

    logger.info("Pipeline finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
