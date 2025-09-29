# src/notify/telegram.py
"""
Cliente de notificaciones para Telegram (Bot API).

Características:
- Envío de mensajes de texto, fotos y documentos.
- Reintentos con backoff exponencial ante errores temporales.
- Segmentación (chunking) automática para mensajes largos.
- Escapes para MarkdownV2.
- Modo dry_run para pruebas (no envía, solo imprime).
- Helpers de formateo para señales, updates de gestión y resúmenes de régimen/backtest.

Requisitos:
- Variable de entorno TELEGRAM_BOT_TOKEN
- Variable de entorno TELEGRAM_CHAT_ID (o pasar chat_id manualmente)

Ejemplo rápido:
    from notify.telegram import TelegramClient, format_entry_signal
    tg = TelegramClient()  # lee env vars
    msg = format_entry_signal(
        ticker="AAPL", entry=215.34, stop=205.1, tps=[225.0, 240.0],
        reasons=["Breakout 55", "RS>SPY", "Volumen+ z=1.3"],
        regime=("BULL", 0.72), risk_pct=1.0
    )
    tg.send_message(msg)

Notas:
- El parseo por defecto usa MarkdownV2. Si quieres HTML, usa parse_mode="HTML" y no llames a escape_markdown_v2.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import requests


# =========================
# Utilidades
# =========================

MDV2_SPECIALS = r"_*[]()~`>#+-=|{}.!"  # caracteres especiales de MarkdownV2

def escape_markdown_v2(text: str) -> str:
    """
    Escapa caracteres especiales para MarkdownV2.
    """
    if text is None:
        return ""
    out = []
    for ch in str(text):
        if ch in MDV2_SPECIALS:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)

def chunk_text(text: str, limit: int = 3950) -> List[str]:
    """
    Telegram limita ~4096 chars por mensaje. Dejamos margen.
    Divide por líneas para evitar cortar palabras.
    """
    lines = text.splitlines(True)  # keepends
    chunks: List[str] = []
    buf = ""
    for ln in lines:
        if len(buf) + len(ln) > limit:
            if buf:
                chunks.append(buf)
            buf = ln
            if len(buf) > limit:  # línea enorme: cortar duro
                for i in range(0, len(buf), limit):
                    chunks.append(buf[i:i+limit])
                buf = ""
        else:
            buf += ln
    if buf:
        chunks.append(buf)
    return chunks

def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


# =========================
# Config y Cliente
# =========================

@dataclass
class TelegramConfig:
    bot_token: str = _coalesce(os.getenv("TELEGRAM_BOT_TOKEN"), default="")
    default_chat_id: str = _coalesce(os.getenv("TELEGRAM_CHAT_ID"), default="")
    parse_mode: str = "MarkdownV2"  # "MarkdownV2" | "HTML" | ""
    disable_web_page_preview: bool = True
    disable_notification: bool = False
    dry_run: bool = False          # True = no envía, solo imprime
    timeout_sec: int = 20          # timeout por request
    max_retries: int = 4
    backoff_base: float = 0.8      # segundos
    proxy: Optional[str] = None    # ej: "http://user:pass@host:port"

class TelegramClient:
    def __init__(self, config: Optional[TelegramConfig] = None):
        self.cfg = config or TelegramConfig()
        self.session = requests.Session()
        if self.cfg.proxy:
            self.session.proxies.update({"http": self.cfg.proxy, "https": self.cfg.proxy})

        if not self.cfg.bot_token and not self.cfg.dry_run:
            raise ValueError("Falta TELEGRAM_BOT_TOKEN (env var) o usar dry_run=True.")
        # chat_id puede pasarse por send_*; si no, usa default_chat_id

    # ------------- núcleo HTTP -------------
    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.cfg.bot_token}/{method}"

    def _request(self, method: str, data: dict, files: Optional[dict] = None) -> dict:
        """
        Hace la request con reintentos/backoff. Devuelve el JSON del Bot API.
        """
        if self.cfg.dry_run:
            print(f"[DRY RUN] {method}: {json.dumps({k: (str(v)[:200] if k!='text' else (v[:200]+'...')) for k,v in data.items()}, ensure_ascii=False)}")
            return {"ok": True, "result": {"message_id": 0}}

        url = self._api_url(method)
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.post(url, data=data, files=files, timeout=self.cfg.timeout_sec)
                if resp.status_code == 200:
                    payload = resp.json()
                    if payload.get("ok", False):
                        return payload
                    # errores del lado de Telegram (p.ej., flood control)
                    last_err = payload
                else:
                    last_err = {"status_code": resp.status_code, "text": resp.text}

            except Exception as e:
                last_err = {"exception": str(e)}

            # backoff exponencial suave
            sleep_s = self.cfg.backoff_base * (2 ** attempt)
            time.sleep(sleep_s)

        raise RuntimeError(f"Telegram API error tras reintentos: {last_err}")

    # ------------- envíos -------------
    def send_message(
        self,
        text: str,
        chat_id: Optional[Union[str, int]] = None,
        parse_mode: Optional[str] = None,
        disable_web_page_preview: Optional[bool] = None,
        disable_notification: Optional[bool] = None,
        protect_content: Optional[bool] = None,
    ) -> List[dict]:
        """
        Envía un mensaje (con chunking). Devuelve lista de payloads (uno por chunk).
        """
        chat = _coalesce(chat_id, self.cfg.default_chat_id)
        if not chat and not self.cfg.dry_run:
            raise ValueError("Falta chat_id (TELEGRAM_CHAT_ID o parámetro).")

        pmode = _coalesce(parse_mode, self.cfg.parse_mode, default="")
        chunks = chunk_text(text)

        results = []
        for chunk in chunks:
            data = {
                "chat_id": chat,
                "text": chunk,
                "disable_web_page_preview": _coalesce(disable_web_page_preview, self.cfg.disable_web_page_preview),
                "disable_notification": _coalesce(disable_notification, self.cfg.disable_notification),
            }
            if protect_content is not None:
                data["protect_content"] = bool(protect_content)
            if pmode:
                data["parse_mode"] = pmode

            results.append(self._request("sendMessage", data))
        return results

    def send_photo(
        self,
        photo_bytes: bytes,
        caption: Optional=str,
        chat_id: Optional[Union[str, int]] = None,
        parse_mode: Optional[str] = None,
    ) -> dict:
        chat = _coalesce(chat_id, self.cfg.default_chat_id)
        if not chat and not self.cfg.dry_run:
            raise ValueError("Falta chat_id para foto.")

        pmode = _coalesce(parse_mode, self.cfg.parse_mode, default="")
        data = {"chat_id": chat}
        if caption:
            data["caption"] = caption
            if pmode:
                data["parse_mode"] = pmode
        files = {"photo": ("image.png", io.BytesIO(photo_bytes))}
        return self._request("sendPhoto", data=data, files=files)

    def send_document(
        self,
        file_bytes: bytes,
        filename: str,
        caption: Optional[str] = None,
        chat_id: Optional[Union[str, int]] = None,
        parse_mode: Optional[str] = None,
    ) -> dict:
        chat = _coalesce(chat_id, self.cfg.default_chat_id)
        if not chat and not self.cfg.dry_run:
            raise ValueError("Falta chat_id para documento.")

        pmode = _coalesce(parse_mode, self.cfg.parse_mode, default="")
        data = {"chat_id": chat}
        if caption:
            data["caption"] = caption
            if pmode:
                data["parse_mode"] = pmode
        files = {"document": (filename, io.BytesIO(file_bytes))}
        return self._request("sendDocument", data=data, files=files)

    # ------------- helpers de alto nivel -------------
    def send_signal_card(
        self,
        *,
        ticker: str,
        close: float,
        entry: Optional[float],
        stop: Optional[float],
        tps: Sequence[float] | None,
        notes: str = "",
        risk_pct: Optional[float] = None,
        regime: Optional[Tuple[str, float]] = None,  # ("BULL"|"NEUTRAL"|"BEAR", 0..1)
        chat_id: Optional[Union[str, int]] = None,
    ):
        """
        Envía una tarjeta de señal (entrada/gestión) con formato consistente.
        """
        msg = format_signal_card(
            ticker=ticker, close=close, entry=entry, stop=stop, tps=tps,
            notes=notes, risk_pct=risk_pct, regime=regime
        )
        return self.send_message(msg, chat_id=chat_id)

    def send_rank_table(
        self,
        df: pd.DataFrame,
        *,
        top_n: int = 20,
        filename: str = "rank_top.csv",
        caption: Optional[str] = None,
        chat_id: Optional[Union[str, int]] = None,
    ):
        """
        Envía el top-N como CSV adjunto (y opcionalmente una cabecera como texto).
        """
        df = df.head(top_n).copy()
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        cap = caption or "Top por score"
        return self.send_document(csv_bytes, filename=filename, caption=cap, chat_id=chat_id)

    def send_regime_summary(
        self,
        label: str,
        confidence: float,
        risk_multiplier: float,
        asof: Optional[str] = None,
        chat_id: Optional[Union[str, int]] = None,
    ):
        msg = format_regime_summary(label=label, confidence=confidence, risk_multiplier=risk_multiplier, asof=asof)
        return self.send_message(msg, chat_id=chat_id)

    def send_backtest_summary(
        self,
        metrics: pd.Series,
        chat_id: Optional[Union[str, int]] = None,
    ):
        msg = format_backtest_summary(metrics)
        return self.send_message(msg, chat_id=chat_id)


# =========================
# Formateadores de mensajes
# =========================

def format_signal_card(
    *,
    ticker: str,
    close: float,
    entry: Optional[float],
    stop: Optional[float],
    tps: Sequence[float] | None,
    notes: str = "",
    risk_pct: Optional[float] = None,
    regime: Optional[Tuple[str, float]] = None,
) -> str:
    """
    Arma una tarjeta compacta de señal con MarkdownV2.
    """
    def fnum(x, nd=2):
        try:
            return f"{float(x):,.{nd}f}"
        except Exception:
            return str(x)

    header = f"*{escape_markdown_v2(ticker)}*  —  Close: `{fnum(close)}`\n"
    lines = [header]

    if entry is not None and stop is not None:
        R = max(entry - stop, 0.0)
        lines.append(f"Entrada: `{fnum(entry)}`   Stop: `{fnum(stop)}`   R: `{fnum(R)}`")
    elif entry is not None:
        lines.append(f"Entrada: `{fnum(entry)}`")
    elif stop is not None:
        lines.append(f"Stop: `{fnum(stop)}`")

    if tps:
        tps_fmt = ", ".join(f"`{fnum(tp)}`" for tp in tps)
        lines.append(f"TPs: {tps_fmt}")

    if risk_pct is not None:
        lines.append(f"Riesgo sugerido: `{fnum(risk_pct, 2)}%`")

    if regime:
        r_label, r_conf = regime
        lines.append(f"Régimen: *{escape_markdown_v2(r_label)}* `{r_conf:.2f}`")

    if notes:
        notes_clean = escape_markdown_v2(notes)
        lines.append(f"_Notas:_ {notes_clean}")

    return "\n".join(lines)

def format_entry_signal(
    ticker: str,
    entry: float,
    stop: float,
    tps: Sequence[float] | None,
    reasons: Sequence[str],
    regime: Optional[Tuple[str, float]] = None,
    risk_pct: float = 1.0,
) -> str:
    """
    Mensaje para nueva ENTRADA.
    """
    base = format_signal_card(
        ticker=ticker, close=entry, entry=entry, stop=stop, tps=tps,
        notes="; ".join(reasons), risk_pct=risk_pct, regime=regime
    )
    return f"🚀 *Señal de Entrada*\n{base}"

def format_manage_update(
    ticker: str,
    new_stop: Optional[float] = None,
    tps_hit: Optional[Sequence[float]] = None,
    closed: bool = False,
    pnl: Optional[float] = None,
    extra: Optional[str] = None,
) -> str:
    """
    Mensaje de gestión (mover stop, tocar TP, cierre, etc.).
    """
    lines = [f"🛠️ *Gestión* — *{escape_markdown_v2(ticker)}*"]
    if new_stop is not None:
        lines.append(f"Nuevo Stop: `{new_stop:,.2f}`")
    if tps_hit:
        tps_fmt = ", ".join(f"`{tp:,.2f}`" for tp in tps_hit)
        lines.append(f"TPs alcanzados: {tps_fmt}")
    if closed:
        lines.append("Posición: *CERRADA*")
    if pnl is not None:
        sign = "➕" if pnl >= 0 else "➖"
        lines.append(f"PnL: {sign} `{pnl:,.2f}`")
    if extra:
        lines.append(escape_markdown_v2(extra))
    return "\n".join(lines)

def format_regime_summary(
    *,
    label: str,
    confidence: float,
    risk_multiplier: float,
    asof: Optional[str] = None,
) -> str:
    s_label = escape_markdown_v2(label)
    line = f"🧭 *Régimen de Mercado*: *{s_label}* | Confianza `{confidence:.2f}` | Riesgo `x{risk_multiplier:.2f}`"
    if asof:
        line += f" | Asof `{escape_markdown_v2(asof)}`"
    return line

def format_backtest_summary(metrics: pd.Series) -> str:
    """
    Formatea las métricas del backtest (pd.Series) en una tarjeta.
    Espera keys comunes: CAGR, MaxDD, Sharpe, Sortino, HitRate, ProfitFactor, AvgWin, AvgLoss, FinalEquity, Trades
    """
    get = lambda k, d="": metrics.get(k, d)
    def f(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    lines = [
        "📊 *Backtest Summary*",
        f"CAGR: `{f(get('CAGR'), 4)}`   MaxDD: `{f(get('MaxDD'), 4)}`",
        f"Sharpe: `{f(get('Sharpe'))}`   Sortino: `{f(get('Sortino'))}`",
        f"Hit Rate: `{f(get('HitRate'))}`   PF: `{f(get('ProfitFactor'))}`",
        f"AvgWin: `{f(get('AvgWin'), 2)}`   AvgLoss: `{f(get('AvgLoss'), 2)}`",
        f"Trades: `{int(get('Trades') or 0)}`   Final Equity: `{float(get('FinalEquity') or 0):,.2f}`",
    ]
    return "\n".join(lines)


# =========================
# Modo script
# =========================

if __name__ == "__main__":
    # Demo local:
    #  - Exporta TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID para enviar de verdad
    #  - O pon dry_run=True en el config para ver el output sin enviar
    cfg = TelegramConfig(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        default_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        parse_mode="MarkdownV2",
        dry_run=(os.getenv("TG_DRY_RUN", "0") == "1"),
    )
    tg = TelegramClient(cfg)

    # 1) Resumen de régimen
    tg.send_regime_summary(label="BULL", confidence=0.72, risk_multiplier=1.0, asof="2025-09-28")

    # 2) Señal de entrada
    msg = format_entry_signal(
        ticker="AAPL",
        entry=215.34,
        stop=205.10,
        tps=[225.00, 240.00],
        reasons=["Breakout 55", "RS>SPY", "Volumen+ z=1.3"],
        regime=("BULL", 0.72),
        risk_pct=1.0,
    )
    tg.send_message(msg)

    # 3) Enviar top como CSV
    df_demo = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "NVDA"],
        "score": [88.5, 84.1, 81.3],
        "bucket": ["A+", "A", "A"],
    })
    tg.send_rank_table(df_demo, top_n=3, caption="Top 3 semanal")
