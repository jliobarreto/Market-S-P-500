# Sistema de Señales Anticipadas para Acciones USA (Holding Estratégico)

Generador de **señales técnicas** (sin ejecución) para **holding estratégico** en acciones de Estados Unidos, con enfoque en **acumulación → ruptura**, entradas/salidas **escalonadas**, control de **riesgo** y **ranking semanal**. Entrega resultados por **Telegram** y guarda historial en **CSV**.

---

## 🎯 Objetivo

Detectar, cada **lunes** (2 horas antes de la apertura en NY), las **mejores oportunidades** de compra anticipada en acciones liquidas y de calidad, priorizando **menos pérdidas, más ganancias** y manteniendo posiciones durante **meses** (hasta ≥1 año) mientras la tendencia siga sana.

---

## 🔍 Alcance

* **Solo generación de señales** (sin trading automático).
* Temporalidades: **Semanal** (tendencia mayor) + **Diario** (setup y disparo).
* Universo: Acciones USA con **cap. > 2B**, **volumen ≥ 1M** acc./día, **precio ≥ 10 USD** (ajustable).
* Datos: **yfinance** (OHLCV) con **lookback de 3 años**.
* Reporte: **Telegram** (formato mínimo y accionable) + **CSV** de historial.

---

## 🧠 Lógica (resumen)

**Acumulación (previa a ruptura)**

* Compresión de volatilidad: BB(20,2) estrechas; **ATR** decreciente (10–15 velas).
* Volumen: **seco** en la base + picos verdes puntuales (posible acumulación institucional).
* Estructura: rango con **mínimos crecientes**; **EMA20 ≈ EMA50** (consolidación).
* Fortaleza relativa (**RS**) > SPY y preferible > ETF sectorial.

**Disparo de señal (anticipada)**

* Cierre diario **> resistencia** de la base.
* Volumen del día **≥ 1.5×** promedio 20 días.
* Momentum acompañando: **RSI > 55** y **OBV** rompiendo máximo reciente.

**Filtro macro**

* Abrir nuevas solo si **SPY > EMA50 semanal** y **VIX < 25** (ajustable).

---

## 🧮 Entradas/Salidas escalonadas

**Entradas (3 tramos)**: 40% / 35% / 25% con espaciado por **ATR**: 0.0 / +0.5 / +1.0 ATR desde el gatillo.

**Salidas**

* **TP1 (30%)**: +20% desde entrada promedio (asegurar ganancia).
* **TP2 (40%)**: +50% o resistencia mayor.
* **TP3 (30%)**: **trailing** por **EMA50 semanal** (mantener tendencia).

**Stop**

* Inicial: bajo soporte semanal o **−10%** (lo que ocurra primero).
* Dinámico: a **break-even** tras TP1; luego trailing por estructura.

---

## 📊 Ranking semanal (Top-N ≤ 20)

Puntaje = **RS (35%)** + **Estructura/Compresión (30%)** + **Volumen institucional (20%)** + **Potencial a resistencia (15%)**.

* Top-N máximo: **20** (si hay menos de calidad, enviar solo esas).
* En semanas con capital limitado: **priorizar por score** (no dividir entre todas).

---

## 📅 Programación

* **Ejecución semanal:** Lunes, **2 h antes** de apertura NY (≈ **07:30 ET**).

  * Referencia local: ajustar según tu huso horario; el proyecto incluye script para cron/Task Scheduler.
* **Sin alertas intradía**: todo el seguimiento se consolida en el informe del lunes siguiente.

---

## 🗂️ Estructura del repositorio

```
project/
├─ config/
│  ├─ config.py            # Parámetros globales (riesgo, filtros, calendario)
│  └─ secrets_template.env # Plantilla .env
├─ data/
│  ├─ universe.py          # Universo de símbolos (filtros de liquidez, precio, cap.)
│  ├─ loader_yf.py         # Conector yfinance (OHLCV, 3y)
│  └─ cache/               # (opcional) caché local de datos
├─ features/
│  ├─ volatility.py        # ATR, ancho de Bandas de Bollinger
│  ├─ volume.py            # Volumen medio, picos, OBV
│  ├─ trend.py             # EMAs, RS vs SPY y sector
│  └─ structure.py         # Detección de bases, compresión, niveles
├─ signals/
│  ├─ rules.py             # Reglas de acumulación → ruptura (anticipada)
│  └─ planner.py           # Cálculo de entradas/salidas escalonadas y SL
├─ risk/
│  ├─ sizing.py            # Tamaño de posición por % riesgo y ATR
│  └─ portfolio.py         # Límite de exposición, num. posiciones, priorización
├─ backtest/
│  ├─ engine.py            # Motor de backtest (compuesto, en cartera)
│  ├─ metrics.py           # CAGR, MaxDD, WinRate, Expectancy, Sharpe, etc.
│  └─ reports.py           # Curva de equity y cuadros resumen (CSV)
├─ reports/
│  ├─ weekly_ranker.py     # Orquestador del ranking semanal (Top-N)
│  └─ export_csv.py        # Exportación de señales e historial a CSV
├─ notify/
│  └─ telegram.py          # Envío de mensajes a Telegram
├─ storage/
│  └─ history/             # CSV de historial y resultados de backtest
├─ utils/
│  ├─ dates.py             # Calendario de mercado, zonas horarias
│  ├─ logger.py            # Logging uniforme
│  └─ helpers.py           # Utilidades varias
├─ main.py                 # Punto de entrada por CLI (workflow semanal)
├─ README.md               # (este archivo)
└─ requirements.txt        # Dependencias mínimas
```

---

## 🧪 Backtest (3 años)

* **Capital inicial**: configurable (ej. 10.000 USD); **reinversión de ganancias** (crecimiento compuesto).
* **Gestión de capital**: por score cuando hay demasiadas señales.
* **Métricas** (a nivel **portafolio** y **ticker**):

  * **CAGR**, **Max Drawdown**, **Win Rate**, **Expectancy (R)**, **Profit Factor**, **Sharpe**, **Time in Market**, **Average Holding Period**, **Rendimiento por sector**.
* **Salidas**: CSV en `storage/history/` + curva de equity exportable.

---

## 📤 Notificación por Telegram (formato mínimo)

Ejemplo por señal:

```
Ticker: NVDA
Plan de entradas:
 - Entrada 1 (40%): 495.20
 - Entrada 2 (35%): 497.50
 - Entrada 3 (25%): 500.80

Plan de salidas:
 - TP1 (30%): 520.00
 - TP2 (40%): 545.00
 - TP3 (30%): Trailing Stop EMA50 semanal

Stop Loss inicial: 478.50
Timeframe: Diario (confirmado en Semanal)
```

---

## 🧾 Historial (CSV)

Carpeta: `storage/history/`

**Campos sugeridos**

* `date_signal` (YYYY-MM-DD)
* `ticker`, `sector`, `industry`
* `score_total`, `score_rs`, `score_structure`, `score_volume`, `score_potential`
* `entry1`, `entry2`, `entry3`
* `tp1`, `tp2`, `tp3`, `sl_initial`, `sl_final`
* `risk_pct`, `position_size_usd`
* `result_R`, `result_pct`, `holding_days`
* `notes` (opcional)

---

## 🔐 Credenciales y configuración

Archivo **`.env`** (no subir a git):

```
TELEGRAM_TOKEN=xxxx
TELEGRAM_CHAT_ID=xxxx
```

Parámetros clave (en `config/config.py`):

* **Riesgo por operación**: `0.5%`
* **Exposición máxima**: `65%`
* **Máximo posiciones**: `12`
* **Liquidez mínima**: `vol_avg_daily ≥ 1.5M`, `price ≥ 10`
* **TopN semanal**: `≤ 20`
* **Horario**: lunes, 2h antes apertura NY

---

## 🚀 Instalación rápida

Requisitos: **Python 3.10+**

```bash
# Clonar el repo
git clone <url> project && cd project

# (Opcional) crear venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias mínimas
pip install -r requirements.txt

# Crear .env desde plantilla
cp config/secrets_template.env .env  # Windows: copy config\secrets_template.env .env
```

`requirements.txt` (mínimo sugerido):

```
yfinance
pandas
numpy
scipy
matplotlib
python-dotenv
requests
```

---

## 🕒 Programación del job semanal

**Windows (Task Scheduler):**

* Acción: `python main.py --weekly`
* Programar: Lunes a la hora local correspondiente (2h antes de apertura NY).

**macOS/Linux (cron):**

```
# Ejemplo: 7:30 America/New_York → ajustar TZ o usar UTC
30 7 * * 1 /usr/bin/env bash -lc 'cd /ruta/project && . .venv/bin/activate && python main.py --weekly'
```

---

## 🗺️ Roadmap de entregas

1. **config/** y **data/** (yfinance) ✅
2. **features/** (ATR, BB, EMAs, RS, OBV)
3. **signals/** (reglas + planner escalonado)
4. **risk/** (sizing, portfolio)
5. **backtest/** (engine, metrics, reports CSV)
6. **reports/** + **notify/** (Telegram)
7. **main.py** (orquestación CLI) y tareas programadas
