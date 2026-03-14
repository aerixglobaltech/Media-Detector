from __future__ import annotations

import logging
from datetime import datetime

from app.db.session import get_db_connection
from app.services.telegram_service import send_message
from app.models.telegram_user_model import get_chat_id_by_phone
from app.services.telegram_utils import (
    format_attendance_message,
    is_valid_phone_number,
    normalize_phone_number,
)

log = logging.getLogger("attendance_service")


def mark_attendance(employee_name: str, phone_number: str, status: str) -> dict:
    normalized_phone = normalize_phone_number(phone_number)
    normalized_status = (status or "").strip().upper()
    clean_name = (employee_name or "").strip()

    if not clean_name and not normalized_phone:
        raise ValueError("Either employee_name or phone_number is required")
    if normalized_status not in {"IN", "OUT"}:
        raise ValueError("status must be either IN or OUT")
    if normalized_phone and not is_valid_phone_number(normalized_phone):
        raise ValueError("Invalid phone number format")

    event_time = datetime.now()
    in_time = event_time if normalized_status == "IN" else None
    out_time = event_time if normalized_status == "OUT" else None

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Resolve staff record using staff table columns (name/email/phone)
        staff_row = None
        if normalized_phone:
            cur.execute(
                "SELECT id, name, email, phone FROM staff_profiles WHERE phone = %s LIMIT 1",
                (normalized_phone,),
            )
            staff_row = cur.fetchone()
        if not staff_row:
            cur.execute(
                "SELECT id, name, email, phone FROM staff_profiles WHERE name = %s LIMIT 1",
                (clean_name,),
            )
            staff_row = cur.fetchone()

        if not staff_row:
            raise ValueError("Staff member not found in staff_profiles")

        resolved_name = staff_row["name"]
        resolved_email = staff_row.get("email")
        resolved_phone = staff_row.get("phone") or normalized_phone
        resolved_staff_id = staff_row["id"]

        cur.execute(
            """
            INSERT INTO attendance (staff_id, status, in_time, out_time, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, staff_id, status, in_time, out_time, timestamp
            """,
            (resolved_staff_id, normalized_status, in_time, out_time, event_time),
        )
        row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()

    chat_id = get_chat_id_by_phone(resolved_phone)
    notified = False
    if chat_id is not None:
        msg = format_attendance_message(
            employee_name=row["name"],
            status=normalized_status,
            when=event_time,
        )
        notified = send_message(chat_id=chat_id, text=msg)
        if notified:
            log.info(
                "Attendance notification sent: employee=%s phone=%s chat_id=%s status=%s",
                resolved_name,
                resolved_phone,
                chat_id,
                normalized_status,
            )
        else:
            log.error(
                "Attendance notification failed: employee=%s phone=%s chat_id=%s status=%s",
                resolved_name,
                resolved_phone,
                chat_id,
                normalized_status,
            )
    else:
        log.info("No Telegram registration found for phone=%s. Attendance saved only.", resolved_phone)

    return {
        "id": row["id"],
        "staff_id": row["staff_id"],
        "name": resolved_name,
        "email": resolved_email,
        "phone": resolved_phone,
        "status": row["status"],
        "in_time": row["in_time"].isoformat() if row.get("in_time") else None,
        "out_time": row["out_time"].isoformat() if row.get("out_time") else None,
        "timestamp": row["timestamp"].isoformat() if row.get("timestamp") else None,
        "telegram_notified": notified,
        "chat_id_found": chat_id is not None,
    }

