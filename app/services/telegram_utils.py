from __future__ import annotations

import re
from datetime import datetime

PHONE_RE = re.compile(r"^\+?\d{8,15}$")


def normalize_phone_number(phone_number: str) -> str:
    """
    Normalize phone format:
    - remove spaces, dashes, and parentheses
    - keep optional leading +
    """
    raw = (phone_number or "").strip()
    if not raw:
        return ""
    plus = "+" if raw.startswith("+") else ""
    digits = re.sub(r"\D", "", raw)
    return f"{plus}{digits}" if digits else ""


def is_valid_phone_number(phone_number: str) -> bool:
    normalized = normalize_phone_number(phone_number)
    return bool(PHONE_RE.match(normalized))


def format_attendance_message(employee_name: str, status: str, when: datetime) -> str:
    ts = when.strftime("%Y-%m-%d %H:%M:%S")
    if status == "IN":
        return (
            "✅ Attendance Marked\n"
            f"Employee: {employee_name}\n"
            f"Time: {ts}\n"
            "Status: IN"
        )
    return (
        "🔔 Checkout Recorded\n"
        f"Employee: {employee_name}\n"
        f"Time: {ts}\n"
        "Status: OUT"
    )

