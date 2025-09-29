# src/universe/sp500.py
"""
Módulo para gestionar el universo del S&P 500.

- Descarga el listado oficial desde Wikipedia
- Guarda/lee un CSV local en data/universe/sp500.csv
- Devuelve tickers limpios o DataFrame con metadata
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# Ruta al CSV local (se actualiza con refresh_sp500_csv)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "universe" / "sp500.csv"


@dataclass(frozen=True)
class UniverseConfig:
    """
    Configuración del universo de trabajo.
    """
    min_history_days: int = 252 * 3  # exigir 3 años de datos
    blacklist: tuple[str, ...] = ()  # excluir tickers manualmente
    upper_case: bool = True          # normalizar tickers en mayúscula


def refresh_sp500_csv() -> pd.DataFrame:
    """
    Descarga el listado del S&P 500 desde Wikipedia y lo guarda en CSV.

    Returns:
        DataFrame con columnas [ticker, company, sector]
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, header=0)
    df = tables[0][["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "company", "GICS Sector": "sector"}
    )
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return df


def load_sp500(config: UniverseConfig = UniverseConfig()) -> list[str]:
    """
    Carga el universo del S&P 500 desde CSV local.
    Si no existe, lo descarga automáticamente.

    Returns:
        Lista de tickers (list[str])
    """
    if not DATA_PATH.exists():
        print("CSV no encontrado, descargando desde Wikipedia...")
        refresh_sp500_csv()

    df = pd.read_csv(DATA_PATH)

    if "ticker" not in df.columns:
        raise ValueError("El CSV debe contener una columna 'ticker'.")

    tickers = df["ticker"].astype(str).str.strip()

    if config.upper_case:
        tickers = tickers.str.upper()

    tickers = tickers.dropna().drop_duplicates().tolist()

    if config.blacklist:
        tickers = [t for t in tickers if t not in set(config.blacklist)]

    return tickers


def load_sp500_with_metadata() -> pd.DataFrame:
    """
    Carga el universo del S&P 500 como DataFrame completo
    (incluye sector, nombre, etc.).
    """
    if not DATA_PATH.exists():
        refresh_sp500_csv()

    df = pd.read_csv(DATA_PATH)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


if __name__ == "__main__":
    # Ejemplo de uso
    tickers = load_sp500()
    print(f"Se cargaron {len(tickers)} tickers del S&P 500")
    print("Ejemplo:", tickers[:10])
