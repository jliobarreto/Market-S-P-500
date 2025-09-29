# src/data/loader.py
"""
Módulo para descarga y caché de datos de acciones del S&P 500.

- Descarga datos con yfinance
- Cachea en parquet (data/cache/equities/{TICKER}.parquet)
- Reutiliza caché si existe
- Permite bajar todo el universo de una vez
"""

from __future__ import annotations
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Iterable

# Carpeta de caché
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "equities"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str) -> Path:
    """Ruta de caché para un ticker dado."""
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def fetch_history(
    ticker: str,
    period: str = "max",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Descarga el histórico de un ticker y lo guarda en caché.

    Args:
        ticker: símbolo (ej. "AAPL")
        period: rango temporal (por defecto "max")
        interval: intervalo de datos (ej. "1d", "1wk")
        force_refresh: si True, ignora caché y descarga de nuevo

    Returns:
        DataFrame con OHLCV ajustado
    """
    path = _cache_path(ticker)

    # Usar caché si existe y no se fuerza refresh
    if path.exists() and not force_refresh:
        try:
            return pd.read_parquet(path)
        except Exception:
            print(f"[WARN] Caché corrupta para {ticker}, re-descargando...")

    # Descarga robusta con reintentos
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=True
            )
            if df.empty:
                raise ValueError(f"Datos vacíos para {ticker}")

            df = df.reset_index()
            # Normalizar nombres de columnas
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Guardar en caché
            df.to_parquet(path, index=False)
            return df

        except Exception as e:
            print(f"[ERROR] Fallo al descargar {ticker}: {e}")
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"No se pudo descargar datos para {ticker}")


def fetch_universe(
    universe: Iterable[str],
    period: str = "max",
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Descarga datos históricos de todo un universo de tickers.

    Args:
        universe: lista de tickers
        period: rango temporal (por defecto "max")
        interval: intervalo de datos (ej. "1d", "1wk")
        force_refresh: si True, ignora caché y descarga de nuevo

    Returns:
        Diccionario {ticker: DataFrame}
    """
    out: dict[str, pd.DataFrame] = {}
    for t in universe:
        try:
            out[t] = fetch_history(
                t, period=period, interval=interval, force_refresh=force_refresh
            )
        except Exception as e:
            print(f"[WARN] No se pudo procesar {t}: {e}")
    return out


if __name__ == "__main__":
    # Ejemplo de uso: descargar AAPL
    df = fetch_history("AAPL", period="5y")
    print(df.tail())

    # Ejemplo: descargar varios
    from universe.sp500 import load_sp500

    tickers = load_sp500()[:5]  # solo 5 primeros para demo
    data = fetch_universe(tickers, period="1y")
    print(f"Se descargaron {len(data)} tickers")
