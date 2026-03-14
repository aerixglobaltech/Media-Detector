from __future__ import annotations

import json
import logging

from flask import Blueprint, jsonify, request

from app.models.telegram_user_model import register_telegram_user
from app.services.attendance_service import mark_attendance
from app.db.session import get_db_connection
from app.services.telegram_service import send_message
from app.services.telegram_utils import is_valid_phone_number, normalize_phone_number

log = logging.getLogger("telegram_routes")

telegram_bp = Blueprint("telegram_integration", __name__)


@telegram_bp.route("/telegram/webhook", methods=["POST"])
def telegram_webhook():
    """
    Telegram webhook receiver with verbose debug logging.
    Always returns HTTP 200 so Telegram webhook delivery succeeds.
    """
    payload = request.get_json(silent=True)
    print("TELEGRAM UPDATE:", payload)
    log.info("TELEGRAM UPDATE: %s", json.dumps(payload, ensure_ascii=False, default=str))
    try:
        if not isinstance(payload, dict):
            log.warning("Telegram webhook payload is not an object: %s", type(payload).__name__)
            return jsonify({"success": True, "message": "ignored_non_object_payload"}), 200

        message = payload.get("message") or {}
        if not isinstance(message, dict):
            message = {}

        text = message.get("text")
        chat_id = (message.get("chat") or {}).get("id")
        contact = message.get("contact") or {}
        phone_number = contact.get("phone_number")

        log.info("Telegram fields: message.text=%r", text)
        log.info("Telegram fields: message.chat.id=%s", chat_id)
        log.info("Telegram fields: message.contact.phone_number=%s", phone_number)

        if text and str(text).strip().startswith("/start"):
            log.info("START COMMAND RECEIVED FROM: %s", chat_id)
            if chat_id:
                welcome_text = (
                    "👋 Welcome to Aerix Attendance Bot\n\n"
                    "To receive attendance notifications, please share your phone number using the button below."
                )
                keyboard = {
                    "keyboard": [[{"text": "Share phone number", "request_contact": True}]],
                    "resize_keyboard": True,
                    "one_time_keyboard": True,
                }
                sent = send_message(chat_id=int(chat_id), text=welcome_text, reply_markup=keyboard)
                if not sent:
                    log.error("Failed to send /start welcome message to chat_id=%s", chat_id)

        if phone_number and chat_id:
            log.info("CONTACT SHARED phone_number=%s chat_id=%s", phone_number, chat_id)
            normalized_phone = normalize_phone_number(str(phone_number))
            if is_valid_phone_number(normalized_phone):
                register_telegram_user(normalized_phone, int(chat_id))
                confirm = f"✅ Registration complete for {normalized_phone}."
                if not send_message(chat_id=int(chat_id), text=confirm):
                    log.error("Failed to send contact confirmation to chat_id=%s", chat_id)
            else:
                invalid = "Invalid phone number format. Please share a valid phone number."
                send_message(chat_id=int(chat_id), text=invalid)
                log.warning("Invalid phone number received from chat_id=%s: %s", chat_id, phone_number)

        return jsonify({"success": True}), 200
    except Exception as exc:
        log.exception("Telegram webhook processing failed: %s", exc)
        return jsonify({"success": True, "error": "Webhook processing failed"}), 200


@telegram_bp.route("/api/attendance", methods=["GET"])
def api_get_attendance():
    """Returns attendance history from database."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, employee_name, phone_number, status, timestamp FROM attendance ORDER BY timestamp DESC LIMIT 100")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert timestamp to ISO format for JSON
        for row in rows:
            if row.get('timestamp'):
                row['timestamp'] = row['timestamp'].isoformat()
                
        return jsonify(rows)
    except Exception as e:
        log.error("Failed to fetch attendance: %s", e)
        return jsonify({"error": str(e)}), 500


@telegram_bp.route("/attendance/mark", methods=["POST"])
def attendance_mark():
    """
    Records attendance and sends Telegram notification if user is registered.
    Request JSON:
      {employee_name|name, phone_number|phone, status}
    """
    data = request.get_json(silent=True) or {}
    employee_name = data.get("employee_name") or data.get("name", "")
    phone_number = data.get("phone_number") or data.get("phone", "")
    status = data.get("status", "")

    try:
        result = mark_attendance(employee_name, phone_number, status)
        return jsonify({"success": True, "attendance": result}), 200
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        log.error("Attendance mark failed: %s", exc)
        return jsonify({"success": False, "error": "Attendance processing failed"}), 500

