
from __future__ import annotations

import logging

from app.services.telegram_service import send_message
from app.models.telegram_user_model import register_telegram_user
from app.services.telegram_utils import is_valid_phone_number, normalize_phone_number

log = logging.getLogger("telegram_bot")
WELCOME_TEXT = (
    "👋 Welcome to Aerix Attendance Bot\n\n"
    "To receive attendance notifications, please share your phone number."
)


def _contact_keyboard() -> dict:
    return {
        "keyboard": [[{"text": "Share phone number", "request_contact": True}]],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }


def _remove_keyboard() -> dict:
    return {"remove_keyboard": True}


def handle_start(chat_id: int) -> None:
    send_message(
        chat_id=chat_id,
        text=WELCOME_TEXT,
        reply_markup=_contact_keyboard(),
    )


def handle_contact(phone_number: str, chat_id: int) -> tuple[bool, str]:
    normalized_phone = normalize_phone_number(phone_number)
    if not is_valid_phone_number(normalized_phone):
        return False, "Invalid phone number format. Please send a valid number."

    register_telegram_user(normalized_phone, chat_id)
    send_message(
        chat_id=chat_id,
        text=f"Registration successful for {normalized_phone}. You will receive attendance updates.",
        reply_markup=_remove_keyboard(),
    )
    return True, "Registered"


def handle_webhook_update(update: dict) -> dict:
    """
    Process Telegram webhook update.
    Handles /start command and contact-sharing flow.
    """
    message = (
        update.get("message")
        or update.get("edited_message")
        or update.get("channel_post")
        or update.get("edited_channel_post")
        or {}
    )
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if not chat_id:
        log.warning("Telegram update ignored: no chat_id. keys=%s", list(update.keys()))
        return {"handled": False, "reason": "no_chat_id"}

    text = (message.get("text") or "").strip()
    if text.startswith("/start"):
        log.info("Received /start from chat_id=%s", chat_id)
        handle_start(int(chat_id))
        return {"handled": True, "action": "start"}

    contact = message.get("contact")
    if contact:
        log.info("Received contact share from chat_id=%s", chat_id)
        sender = message.get("from") or {}
        contact_user_id = contact.get("user_id")
        sender_user_id = sender.get("id")
        # Only accept contacts sent by the same Telegram user.
        if contact_user_id and sender_user_id and contact_user_id != sender_user_id:
            send_message(
                chat_id=int(chat_id),
                text="Please share your own phone number using the provided button.",
            )
            return {"handled": True, "action": "contact_rejected"}

        ok, msg = handle_contact(contact.get("phone_number", ""), int(chat_id))
        if not ok:
            send_message(chat_id=int(chat_id), text=msg)
            return {"handled": True, "action": "contact_invalid"}
        return {"handled": True, "action": "contact_registered"}

    return {"handled": False, "reason": "unsupported_message"}

