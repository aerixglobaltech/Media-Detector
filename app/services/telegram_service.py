from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("telegram_service")


def send_message(
    chat_id: int,
    text: str,
    reply_markup: dict | None = None,
    bot_token: str | None = None,
) -> bool:
    """Send a Telegram message via Bot API."""
    token = (
        bot_token
        or os.environ.get("BOT_TOKEN", "").strip()
        or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    )
    if not token:
        log.error("BOT_TOKEN is missing. Cannot send Telegram message.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: dict = {"chat_id": chat_id, "text": text}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            log.error("Telegram sendMessage failed: status=%s body=%s", resp.status_code, resp.text)
            return False
        return True
    except Exception as exc:
        log.error("Telegram API error while sending message: %s", exc)
        return False

