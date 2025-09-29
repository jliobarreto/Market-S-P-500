import yfinance as yf
import pandas_datareader.data as web
import time
import os
import random

# ====== Parámetros ======
PERIOD = "1y"
INTERVAL = "1d"
BLOCK_SIZE = 50
SLEEP_BETWEEN_SYMBOLS = (1.0, 2.5)  # segundos
SLEEP_BETWEEN_BLOCKS = (3, 6)

# ====== Tickers (personalizable) ======
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA", "JPM", "V", "UNH",
    "PG", "HD", "KO", "PEP", "DIS", "INTC", "CSCO", "XOM", "CVX", "MRK",
    "PFE", "NFLX", "BA", "IBM", "ORCL", "ADBE", "CRM", "NKE", "WMT", "COST",
    "QCOM", "MDT", "GE", "HON", "MCD", "T", "SBUX", "GS", "BLK", "AMD",
    "BKNG", "CAT", "MMM", "PYPL", "AXP", "DE", "DHR", "ABBV", "TXN", "VRTX",
]

# ====== Preparar carpeta ======
os.makedirs("data_cache", exist_ok=True)

def download_ticker(ticker):
    print(f"📥 Descargando {ticker}...")
    try:
        data = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
        if data.empty:
            raise ValueError("❌ Vacío o bloqueado desde Yahoo")
        path = f"data_cache/{ticker}.csv"
        data.to_csv(path)
        print(f"✅ Guardado en {path}")
    except Exception as e:
        print(f"⚠️ Error con {ticker} en yfinance: {e}")
        print("🔁 Intentando con Stooq...")
        try:
            stooq_ticker = f"{ticker}.US"
            stooq_data = web.DataReader(stooq_ticker, data_source='stooq')
            stooq_path = f"data_cache/{ticker}_stooq.csv"
            stooq_data.to_csv(stooq_path)
            print(f"✅ Guardado con Stooq en {stooq_path}")
        except Exception as se:
            print(f"❌ Falló también en Stooq: {se}")

def download_block(block):
    for ticker in block:
        download_ticker(ticker)
        wait = random.uniform(*SLEEP_BETWEEN_SYMBOLS)
        time.sleep(wait)

# ====== Procesamiento en bloques ======
total_blocks = (len(TICKERS) + BLOCK_SIZE - 1) // BLOCK_SIZE
for i in range(0, len(TICKERS), BLOCK_SIZE):
    block_num = i // BLOCK_SIZE + 1
    block = TICKERS[i:i + BLOCK_SIZE]
    print(f"\n🧩 Procesando bloque {block_num} de {total_blocks}")
    download_block(block)
    if block_num < total_blocks:
        wait_block = random.uniform(*SLEEP_BETWEEN_BLOCKS)
        print(f"⏳ Esperando {wait_block:.2f}s antes del siguiente bloque...\n")
        time.sleep(wait_block)

print("\n✅ Descarga completada.")
