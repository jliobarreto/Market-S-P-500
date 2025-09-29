from __future__ import annotations
import argparse
from pathlib import Path

from config.config import load_config
from reports.weekly_ranker import rank_universe, make_telegram_messages
from notify.telegram import send_bulk
from reports.export_csv import export_weekly_rank


def run_weekly() -> None:
    cfg = load_config()
    print("[INFO] Ejecutando ranking semanal…")
    df_rank = rank_universe(source="combined")
    if df_rank.empty:
        print("[INFO] Sin oportunidades esta semana (filtros o mercado no favorable).")
        return

    # Export CSV
    out_path = export_weekly_rank(df_rank)
    print(f"[OK] CSV exportado en: {out_path}")

    # Enviar Telegram
    if cfg.telegram.enabled and cfg.telegram.token and cfg.telegram.chat_id:
        msgs = make_telegram_messages(df_rank)
        send_bulk(msgs)
        print(f"[OK] {len(msgs)} mensajes enviados a Telegram.")
    else:
        print("[WARN] Telegram no configurado. Se omitió el envío.")


def main():
    parser = argparse.ArgumentParser(description="Orquestador del sistema semanal")
    parser.add_argument("--weekly", action="store_true", help="Ejecuta el flujo semanal")
    args = parser.parse_args()

    if args.weekly:
        run_weekly()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
