# dev/healthcheck.py
"""
Healthcheck / Diagnóstico del entorno
-------------------------------------
Prueba dependencias, red y acceso a datos. Si yfinance bloquea,
usa Stooq como refuerzo (fallback) para validar que el pipeline
puede seguir funcionando.

Uso:
    python dev/healthcheck.py
"""

from __future__ import annotations
import os
import sys
import socket
import time
import traceback
from dataclasses import dataclass
from typing import Callable, List, Tuple

# --- Utilidades de impresión ---
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def ok(msg: str): print(f"{GREEN}✔{RESET} {msg}")
def warn(msg: str): print(f"{YELLOW}⚠{RESET} {msg}")
def fail(msg: str): print(f"{RED}✖{RESET} {msg}")
def info(msg: str): print(f"{CYAN}•{RESET} {msg}")

def section(title: str):
    print()
    print(f"{'-'*8} {title} {'-'*8}")

@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str = ""
    suggestion: str = ""

def run_check(name: str, fn: Callable[[], Tuple[bool, str, str]]) -> CheckResult:
    try:
        passed, details, suggestion = fn()
        if passed:
            ok(f"{name}: {details}")
        else:
            fail(f"{name}: {details}")
            if suggestion:
                warn(f"Sugerencia: {suggestion}")
        return CheckResult(name, passed, details, suggestion)
    except Exception as e:
        fail(f"{name}: {e}")
        tb = traceback.format_exc(limit=1)
        warn(tb.strip())
        return CheckResult(name, False, str(e), "Revisa el stacktrace y la red/credenciales.")

# =========================
# Checks individuales
# =========================

def check_python() -> Tuple[bool, str, str]:
    v = sys.version.split()[0]
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 10:
        return True, f"Python {v}", ""
    return False, f"Python {v}", "Usa Python 3.10+ para compatibilidad."

def check_packages() -> Tuple[bool, str, str]:
    missing = []
    versions = {}
    pkgs = {
        "pandas": "pd",
        "numpy": "np",
        "yfinance": "yf",
        "requests": "requests",
        "python-dotenv": "dotenv",
        "pandas_datareader": "pdr",
        "pyarrow": "pyarrow",     # para parquet de la caché
    }
    for mod, _ in pkgs.items():
        try:
            if mod == "python-dotenv":
                import dotenv  # noqa
                versions["python-dotenv"] = getattr(sys.modules["dotenv"], "__version__", "?")
            else:
                __import__(mod)
                versions[mod] = getattr(sys.modules[mod], "__version__", "?")
        except Exception:
            missing.append(mod)
    if missing:
        return False, f"Faltan paquetes: {', '.join(missing)}", "Ejecuta: pip install -r requirements.txt"
    return True, "Dependencias OK: " + ", ".join(f"{k}={v}" for k, v in versions.items()), ""

def check_dns_https() -> Tuple[bool, str, str]:
    import requests
    hosts = [
        ("query2.finance.yahoo.com", "https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=price"),
        ("stooq.com", "https://stooq.com"),
        ("api.telegram.org", "https://api.telegram.org"),
    ]
    for host, url in hosts:
        try:
            socket.gethostbyname(host)
        except Exception:
            return False, f"DNS no resuelve {host}", "Revisa DNS/Firewall o cambia de red."
        try:
            r = requests.get(url, timeout=8)
            if r.status_code >= 400:
                return False, f"HTTP {r.status_code} al acceder {host}", "Posible rate-limit/bloqueo. Prueba modo Stooq."
        except Exception as e:
            return False, f"Error HTTPS {host}: {e}", "Prueba otra red o desactiva VPN/Proxy."
    return True, "DNS/HTTPS alcanzables (Yahoo/Stooq/Telegram)", ""

def check_time_sync() -> Tuple[bool, str, str]:
    import datetime as dt
    local = dt.datetime.now()
    utc = dt.datetime.utcnow()
    return True, f"Horas visibles → local={local:%Y-%m-%d %H:%M:%S} / utc={utc:%Y-%m-%d %H:%M:%S}", "Activa sincronización automática de hora en el sistema."

# --- .env / Telegram

def check_env() -> Tuple[bool, str, str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if token and chat:
        return True, "Variables .env (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID) cargadas", ""
    elif token or chat:
        return False, "Variables .env incompletas (falta una de TELEGRAM_TOKEN/TELEGRAM_CHAT_ID)", "Completa ambas en .env o desactiva Telegram en config."
    else:
        return True, "Variables .env no configuradas (Telegram desactivado) — OK", "Si deseas notificaciones, añade TELEGRAM_TOKEN y TELEGRAM_CHAT_ID."

def check_storage_permissions() -> Tuple[bool, str, str]:
    path = os.path.join("storage", "history")
    os.makedirs(path, exist_ok=True)
    testfile = os.path.join(path, f"__write_test_{int(time.time())}.tmp")
    try:
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testfile)
        return True, f"Escritura habilitada en {path}", ""
    except Exception as e:
        return False, f"No se puede escribir en {path}: {e}", "Ejecuta con permisos o cambia la ruta de salida en config."

# --- Datos: yfinance + Stooq (fallback)

def check_yfinance_single() -> Tuple[bool, str, str]:
    import yfinance as yf
    try:
        df = yf.Ticker("AAPL").history(period="1y", interval="1d", auto_adjust=False, actions=False, prepost=False)
        if df is not None and not df.empty and "Close" in df.columns:
            return True, f"yfinance OK (AAPL 1y: {len(df)} velas)", ""
        # Si no hay datos, intentamos Stooq como refuerzo
        from pandas_datareader import data as pdr
        import pandas as pd
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.DateOffset(years=1)
        ds = pdr.DataReader("AAPL.US", "stooq", start=start, end=end)
        if ds is not None and not ds.empty:
            return True, "Yahoo vacío, pero Stooq OK → puedes usar provider='stooq' temporalmente", "Cambia DataConfig.provider='stooq' en config/config.py"
        return False, "Ni Yahoo ni Stooq devolvieron datos para AAPL", "Revisa conexión/red; intenta con hotspot o VPN diferente."
    except Exception as e:
        # Error en yfinance: intentamos Stooq
        try:
            from pandas_datareader import data as pdr
            import pandas as pd
            end = pd.Timestamp.utcnow().normalize()
            start = end - pd.DateOffset(years=1)
            ds = pdr.DataReader("AAPL.US", "stooq", start=start, end=end)
            if ds is not None and not ds.empty:
                return True, f"yfinance error ({e.__class__.__name__}), pero Stooq OK", "Usa provider='stooq' hasta que Yahoo se libere."
        except Exception as e2:
            return False, f"yfinance error: {e} | Stooq error: {e2}", "Instala pandas-datareader y verifica la red."

def check_stooq_single() -> Tuple[bool, str, str]:
    try:
        from pandas_datareader import data as pdr
        import pandas as pd
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.DateOffset(years=1)
        df = pdr.DataReader("AAPL.US", "stooq", start=start, end=end)
        if df is not None and not df.empty:
            return True, f"Stooq OK (AAPL.US 1y: {len(df)})", ""
        return False, "Stooq devolvió vacío para AAPL.US", "Stooq puede estar caído. Prueba más tarde."
    except Exception as e:
        return False, f"Stooq error: {e}", "Instala pandas-datareader y revisa HTTPS."

def check_loader_small_universe() -> Tuple[bool, str, str]:
    """
    Intenta primero con el proveedor configurado (yfinance con caché/reanudación si existe).
    Si falla, reintenta con Stooq-only para validar que el pipeline puede avanzar.
    """
    # Cargar proveedor desde config si está disponible
    provider = "yfinance"
    try:
        from config.config import load_config
        provider = load_config().data.provider.lower()
    except Exception:
        pass

    # Importamos funciones de descarga
    try:
        from data.loader_yf import (
            download_ohlcv_cached_resume,
            download_ohlcv,
            download_ohlcv_stooq_only,   # puede que no exista en tu versión; cubrimos con except
        )
        have_stooq_only = True
    except Exception:
        try:
            from data.loader_yf import download_ohlcv, download_ohlcv_cached_resume
            have_stooq_only = False
        except Exception as e:
            return False, f"No se pudieron importar funciones del loader: {e}", "Revisa data/loader_yf.py"

    tickers = ["AAPL", "MSFT", "NVDA"]
    # 1) Proveedor configurado
    try:
        if provider == "stooq" and have_stooq_only:
            df = download_ohlcv_stooq_only(tickers, lookback_years=1, interval="1d")
        else:
            # preferir caché/reanudación si existe
            try:
                df = download_ohlcv_cached_resume(tickers, lookback_years=1, interval="1d")
            except Exception:
                df = download_ohlcv(tickers, lookback_years=1, interval="1d")
    except Exception as e:
        df = None
        err1 = e
    else:
        err1 = None

    if df is not None and hasattr(df, "empty") and not df.empty:
        syms = ", ".join(sorted(df["Ticker"].unique()))
        return True, f"Loader ({provider}) OK (tickers: {syms}; filas={len(df)})", ""

    # 2) Fallback explícito a Stooq si la primera ruta no dio datos
    if have_stooq_only:
        try:
            dfs = download_ohlcv_stooq_only(tickers, lookback_years=1, interval="1d")
            if dfs is not None and not dfs.empty:
                syms = ", ".join(sorted(dfs["Ticker"].unique()))
                msg = "Provider principal falló, pero Stooq-only OK"
                if err1:
                    msg += f" (error previo: {err1.__class__.__name__})"
                return True, f"{msg} (tickers: {syms}; filas={len(dfs)})", "Usa provider='stooq' temporalmente en config/config.py"
        except Exception as e2:
            return False, f"Loader falló y Stooq-only también: {e2}", "Verifica red/HTTPS y pandas-datareader."
    else:
        # Si no existe la función stooq_only en tu loader
        return False, "Loader vacío y no hay función Stooq-only disponible", "Añade download_ohlcv_stooq_only en data/loader_yf.py"

    return False, "Loader devolvió vacío en ambas rutas", "Reduce universo y prueba con AAPL/MSFT/NVDA; cambia provider='stooq'."

def check_weekly_ranker_pipeline() -> Tuple[bool, str, str]:
    """
    Ejecuta rank_universe con un universo pequeño.
    Si falla con yfinance, sugiere cambiar a Stooq.
    """
    try:
        from reports.weekly_ranker import rank_universe
        df = rank_universe(source="combined")
        if df is None:
            return False, "rank_universe devolvió None", "Revisa reports/weekly_ranker.py"
        if df.empty:
            return True, "rank_universe ejecutó (0 filas). Puede ser macro/filtros estrictos.", "Desactiva filtro macro o simplifica reglas."
        return True, f"rank_universe OK (filas={len(df)})", ""
    except Exception as e:
        return False, f"weekly_ranker error: {e}", "Si es bloqueo de Yahoo, cambia provider='stooq' en config y reintenta."

def check_telegram() -> Tuple[bool, str, str]:
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return True, "Telegram no configurado (omitido) — OK", "Agrega TELEGRAM_TOKEN/CHAT_ID si quieres alertas."
    try:
        import requests
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=8)
        if r.status_code == 200 and r.json().get("ok"):
            return True, "Telegram bot accesible (getMe OK)", ""
        return False, f"Telegram getMe HTTP {r.status_code}", "Verifica token/chat_id y permisos del bot en el canal."
    except Exception as e:
        return False, f"Telegram error: {e}", "Revisa conexión a api.telegram.org y el token."

def check_scheduler_hint() -> Tuple[bool, str, str]:
    if os.name == "nt":
        return True, "Windows detectado: usa Task Scheduler para programar main.py --weekly", "Crea tarea cada lunes 07:30 hora local."
    else:
        return True, "POSIX detectado: usa cron para programar main.py --weekly", "Ej: '30 7 * * 1 /usr/bin/python /ruta/main.py --weekly'"

# =========================
# Main
# =========================

def main():
    print("\n=== Healthcheck del entorno (dev/healthcheck.py) ===")
    results: List[CheckResult] = []

    section("Sistema")
    results.append(run_check("Python", check_python))
    results.append(run_check("Paquetes", check_packages))
    results.append(run_check("DNS/HTTPS", check_dns_https))
    results.append(run_check("Hora del sistema", check_time_sync))
    results.append(run_check("Permisos de escritura", check_storage_permissions))

    section("Datos")
    results.append(run_check("yfinance (AAPL 1y) con refuerzo Stooq", check_yfinance_single))
    results.append(run_check("Stooq directo (AAPL.US 1y)", check_stooq_single))
    results.append(run_check("Loader OHLCV (fallback a Stooq)", check_loader_small_universe))

    section("Pipeline")
    results.append(run_check("Weekly Ranker", check_weekly_ranker_pipeline))

    section(".env / Telegram")
    results.append(run_check(".env (dotenv)", check_env))
    results.append(run_check("Telegram (getMe)", check_telegram))

    section("Programación")
    results.append(run_check("Sugerencia de Scheduler", check_scheduler_hint))

    # Resumen
    section("Resumen")
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"Checks: {passed}/{total} OK")
    if passed < total:
        print("\nMejoras sugeridas:")
        for r in results:
            if not r.passed and r.suggestion:
                print(f" - {r.name}: {r.suggestion}")

    # Tips finales
    print()
    info("Tips:")
    print("  - Si yfinance falla por rate-limit: en PowerShell ejecuta: $env:YF_NO_THREADING='1'")
    print("  - Cambia temporalmente a Stooq: edita config/config.py → DataConfig.provider='stooq'")
    print("  - Reduce el universo a 5 tickers para pruebas (data/custom_universe.csv).")
    print("  - Caché/reanudación: verifica archivos en storage/cache_ohlcv/*.parquet.")

if __name__ == "__main__":
    main()
