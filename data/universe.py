from __future__ import annotations
from typing import List, Iterable, Optional
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None


# -----------------------------
# Helpers para obtención de candidatos
# -----------------------------

def _wikipedia_table(url: str, symbol_col_candidates=("Symbol", "Ticker")) -> List[str]:
    """Descarga una tabla de Wikipedia y devuelve una lista de tickers.
    Requiere conexión a internet cuando se ejecute localmente.
    """
    tables = pd.read_html(url)
    for tbl in tables:
        for col in symbol_col_candidates:
            if col in tbl.columns:
                tickers = (
                    tbl[col]
                    .astype(str)
                    .str.replace("\n", "", regex=False)
                    .str.strip()
                    .tolist()
                )
                # Limpieza de sufijos de clase de acciones comunes
                tickers = [t.replace(".", "-") for t in tickers]  # BRK.B → BRK-B
                return tickers
    return []


def get_candidate_universe(source: str = "combined") -> List[str]:
    """Devuelve una lista de tickers candidatos según la fuente elegida.
    Fuentes admitidas:
      - 'sp500'
      - 'nasdaq100'
      - 'russell1000'
      - 'combined' (sp500 ∪ nasdaq100 ∪ russell1000)
      - 'custom_csv' (lee 'data/custom_universe.csv' columna 'ticker')
    """
    source = (source or "combined").lower()
    tickers: List[str] = []

    if source in ("sp500", "combined"):
        try:
            tickers += _wikipedia_table(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
        except Exception:
            pass

    if source in ("nasdaq100", "combined"):
        try:
            tickers += _wikipedia_table(
                "https://en.wikipedia.org/wiki/Nasdaq-100",
                symbol_col_candidates=("Ticker", "Symbol"),
            )
        except Exception:
            pass

    if source in ("russell1000", "combined"):
        try:
            tickers += _wikipedia_table(
                "https://en.wikipedia.org/wiki/Russell_1000_Index",
                symbol_col_candidates=("Ticker", "Symbol"),
            )
        except Exception:
            pass

    if source == "custom_csv":
        try:
            df = pd.read_csv("data/custom_universe.csv")
            if "ticker" in df.columns:
                tickers += df["ticker"].astype(str).str.upper().tolist()
        except Exception:
            pass

    # Quitar duplicados y limpiar valores raros
    tickers = sorted({t.upper() for t in tickers if t and t.isascii()})
    # Excluir símbolos no compatibles con yfinance (filtros básicos)
    tickers = [t for t in tickers if all(x not in t for x in [" ", "/", "_", ":"])]
    return tickers


# -----------------------------
# Filtro por liquidez (precio y volumen)
# -----------------------------

def _avg_volume_and_last_price(tickers: Iterable[str], period_days: int = 60) -> pd.DataFrame:
    """Descarga OHLCV reciente y calcula volumen promedio y último precio de cierre.
    Devuelve DataFrame con columnas: ['avg_volume', 'last_close'] index=ticker.
    """
    tickers = list(dict.fromkeys(tickers))  # único y orden estable
    if not tickers:
        return pd.DataFrame(columns=["avg_volume", "last_close"]).astype({"avg_volume": float, "last_close": float})

    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days + 10)

    # yfinance.download soporta múltiples tickers; devolvemos un panel con columnas multi-índice
    raw = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    rows = []
    for t in tickers:
        try:
            df = raw[t]
        except Exception:
            # Si viene en formato de DataFrame simple (un solo ticker), manejar aparte
            if isinstance(raw, pd.DataFrame) and {"Open","High","Low","Close","Volume"}.issubset(raw.columns):
                df = raw.copy()
            else:
                continue
        df = df.dropna(subset=["Close", "Volume"]).copy()
        if df.empty:
            continue
        # Usar últimos 'period_days' disponibles
        df = df.tail(period_days)
        avg_vol = float(df["Volume"].mean()) if not df.empty else 0.0
        last_close = float(df["Close"].iloc[-1]) if not df.empty else float("nan")
        rows.append({"ticker": t, "avg_volume": avg_vol, "last_close": last_close})

    out = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(columns=["avg_volume","last_close"]).astype({"avg_volume": float, "last_close": float})
    return out


def filter_liquid_universe(
    candidates: Iterable[str],
    min_price: float = 10.0,
    min_avg_volume: int = 1_500_000,
) -> List[str]:
    """Filtra tickers por precio último y volumen promedio diario.
    """
    stats = _avg_volume_and_last_price(candidates, period_days=60)
    if stats.empty:
        return []
    mask = (stats["last_close"] >= float(min_price)) & (stats["avg_volume"] >= float(min_avg_volume))
    return stats.loc[mask].index.tolist()


# -----------------------------
# API principal
# -----------------------------

def build_universe(
    source: str = "combined",
    min_price: Optional[float] = None,
    min_avg_volume: Optional[int] = None,
) -> List[str]:
    """Construye el universo aplicado a la configuración actual.
    Intenta usar CFG si está disponible; de lo contrario, aplica parámetros por defecto.
    """
    cand = get_candidate_universe(source=source)

    # Parámetros por defecto de CFG si existe
    if CFG is not None:
        if min_price is None:
            min_price = CFG.universe.liquidity.min_price
        if min_avg_volume is None:
            min_avg_volume = int(CFG.universe.liquidity.min_avg_volume)
    else:
        min_price = float(min_price or 10.0)
        min_avg_volume = int(min_avg_volume or 1_500_000)

    universe = filter_liquid_universe(
        cand,
        min_price=float(min_price),
        min_avg_volume=int(min_avg_volume),
    )
    return universe


if __name__ == "__main__":
    src = "combined"
    uni = build_universe(source=src)
    print(f"Universe({src}) size:", len(uni))
    print(uni[:25])
