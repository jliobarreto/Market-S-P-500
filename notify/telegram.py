from __future__ import annotations
from typing import List, Optional
import requests

try:
    from config.config import load_config
    CFG = load_config()
except Exception:
    CFG = None


def _endpoint(path: str) -> str:
    if CFG is None or CFG.telegram.token is None:
        raise RuntimeError("Telegram no configurado")
    return f"https://api.telegram.org/bot{CFG.telegram.token}/{path}"


def send_message(text: str) -> Optional[dict]:
    if CFG is None or not CFG.telegram.chat_id:
        print("[WARN] TELEGRAM_CHAT_ID no configurado. Mensaje no enviado.")
        return None
    url = _endpoint("sendMessage")
    payload = {
        "chat_id": CFG.telegram.chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return None


def send_bulk(messages: List[str]) -> None:
    for msg in messages:
        send_message(msg)
