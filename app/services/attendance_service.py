from __future__ import annotations

import logging
from datetime import datetime, date

from app.db.session import get_db_connection
from app.services.telegram_service import send_message
from app.services.telegram_user_model import get_chat_id_by_phone
from app.services.telegram_utils import (
    format_attendance_message,
    is_valid_phone_number,
    normalize_phone_number,
)
from app.core.extensions import sse_manager

log = logging.getLogger("attendance_service")



def log_movement(camera_id: str, image_path: str, detected_at: datetime | None = None, track_id: int | None = None, event_type: str | None = None, staff_id: int | None = None, staff_name: str | None = None, person_type: str | None = None) -> int | None:
    """Store movement event with optional identity info."""
    log.debug(f"log_movement called for {camera_id} with image: {image_path}")
    detected_at = detected_at or datetime.now()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO movement_log (camera_id, camera_name, image_path, detected_at, track_id, event_type, staff_id, staff_name, person_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (camera_id, camera_id, image_path, detected_at, track_id, event_type, staff_id, staff_name, person_type))
        row = cur.fetchone()
        conn.commit()
        ret_id = row["id"] if row else None
        log.debug(f"log_movement saved to DB as ID: {ret_id}")
        
        # SSE UPDATE
        if ret_id:
            sse_manager.announce({
                "id": ret_id,
                "camera_id": camera_id,
                "image_path": image_path,
                "detected_at": detected_at.isoformat()
            }, event_type="movement_update")
            
        return ret_id
    except Exception as e:
        log.error(f"Failed to log movement: {e}")
        log.error(f"log_movement DATABASE ERROR: {e}")
        return None
    finally:
        conn.close()


def update_movement_classification(movement_id: int, object_type: str, confidence: float = 0.0) -> None:
    """Second-stage classification result for a movement row."""
    if not movement_id:
        return
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE movement_log
            SET person_type = %s,
                confidence_score = %s
            WHERE id = %s
        """, (object_type, confidence, movement_id))
        conn.commit()
    except Exception as e:
        log.error("Failed to classify movement_log id=%s: %s", movement_id, e)
    finally:
        conn.close()


def log_person(
    camera_id: str,
    person_type: str,
    staff_id: int | None,
    image_path: str,
    confidence: float,
    staff_name: str | None = None,
    detected_at: datetime | None = None,
    track_id: int | None = None,
    event_type: str | None = 'ENTRY',
    entry_time: datetime | None = None,
    exit_time: datetime | None = None,
    roles: list[str] | None = None
):
    """Store human detections (staff/unknown) in member_timestamp."""
    detected_at = detected_at or datetime.now()
    # Force native Python types to avoid numpy/psycopg2 issues
    camera_id = str(camera_id)
    person_type = str(person_type)
    staff_id = int(staff_id) if staff_id is not None else None
    staff_name = str(staff_name) if staff_name is not None else None
    image_path = str(image_path)
    confidence = float(confidence)
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO member_timestamp 
            (camera_id, camera_name, person_type, staff_id, staff_name, image_path, confidence_score, detected_at, track_id, event_type, entry_time, exit_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (camera_id, camera_id, person_type, staff_id, staff_name, image_path, confidence, detected_at, track_id, event_type, entry_time, exit_time))
        row = cur.fetchone()
        conn.commit()
        res_id = row["id"] if row else None
        
        # Trigger attendance update if staff (Redundant safety check)
        if staff_id and person_type.lower() == 'staff':
            try:
                from app.services.attendance_service import track_staff_attendance
                track_staff_attendance(staff_id, staff_name=staff_name, entry_image=image_path, camera_name=camera_id, roles=roles)
            except Exception as e:
                log.warning(f"Failed to auto-trigger attendance in log_person: {e}")

        # SSE UPDATE
        if res_id:
            sse_manager.announce({
                "id": res_id,
                "camera_id": camera_id,
                "person_type": person_type,
                "staff_id": staff_id,
                "staff_name": staff_name,
                "image_path": image_path,
                "confidence": confidence,
                "detected_at": detected_at.isoformat()
            }, event_type="member_log_update")

        return res_id
    except Exception as e:
        log.error(f"Failed to log person detection: {e}")
        return None
    finally:
        conn.close()


def update_person_identity(member_id: int, staff_id: int, staff_name: str, track_id: int | None = None, movement_id: int | None = None, image_path: str | None = None, confidence: float | None = None):
    """Update existing logs with staff info (used if recognized late or corrected)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # 1. Update Member Timestamp
        if member_id:
            log.info(f"DB: Updating member_timestamp #{member_id} to staff {staff_name} (ID: {staff_id})")
            cur.execute("""
                UPDATE member_timestamp 
                SET person_type = 'staff', staff_id = %s, staff_name = %s,
                    image_path = COALESCE(%s, image_path),
                    confidence_score = COALESCE(%s, confidence_score)
                WHERE id = %s
                RETURNING camera_id, image_path
            """, (staff_id, staff_name, image_path, confidence, member_id))
            row = cur.fetchone()
            
            # TRIGGER ATTENDANCE TABLE (Crucial Fix: Ensure staff gets marked present even if recognized late)
            if row:
                cam_id = row["camera_id"]
                effective_image = image_path or row["image_path"]
                try:
                    track_staff_attendance(staff_id, staff_name=staff_name, entry_image=effective_image, camera_name=cam_id)
                except Exception as ex:
                    log.warning(f"Failed to trigger attendance in update_person_identity: {ex}")

            # SSE UPDATE
            sse_manager.announce({
                "id": member_id,
                "person_type": "staff",
                "staff_id": staff_id,
                "staff_name": staff_name,
                "update_type": "late_recognition"
            }, event_type="member_log_update")

        if track_id:
            # Update ALL member logs for this track today (Safety)
            cur.execute("""
                UPDATE member_timestamp 
                SET person_type = 'staff', staff_id = %s, staff_name = %s
                WHERE track_id = %s AND detected_at::date = CURRENT_DATE
            """, (staff_id, staff_name, track_id))

        # 2. Update Movement Logs
        if movement_id:
            cur.execute("""
                UPDATE movement_log 
                SET person_type = 'staff', staff_id = %s, staff_name = %s
                WHERE id = %s
            """, (staff_id, staff_name, movement_id))
            
        if track_id:
            # Update ALL movement logs for this track today
            cur.execute("""
                UPDATE movement_log 
                SET person_type = 'staff', staff_id = %s, staff_name = %s
                WHERE track_id = %s AND detected_at::date = CURRENT_DATE
            """, (staff_id, staff_name, track_id))

        conn.commit()
    except Exception as e:
        log.error(f"Failed to update person identity: {e}")
    finally:
        conn.close()


def get_recent_sighting(staff_id: int, camera_id: str, minutes: int = 5) -> int | None:
    """Find a recent entry for the same staff member to avoid duplicate rows."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM member_timestamp
            WHERE staff_id = %s AND camera_id = %s 
              AND detected_at > NOW() - INTERVAL '%s minutes'
              AND event_type = 'ENTRY'
            ORDER BY detected_at DESC LIMIT 1
        """, (staff_id, camera_id, minutes))
        row = cur.fetchone()
        return row["id"] if row else None
    except Exception as e:
        log.error(f"Error checking recent sightings: {e}")
        return None
    finally:
        conn.close()


def update_exit_logs(member_id: int, movement_id: int, exit_image: str, merged_image: str, track_id: int | None = None, exit_camera_id: str | None = None, exit_camera_name: str | None = None):
    """Update both tables with exit and merged images, and recording the exit camera."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        if member_id:
            cur.execute("""
                UPDATE member_timestamp 
                SET exit_image = %s, merged_image = %s, exit_time = NOW(), event_type = 'EXIT',
                    exit_camera_id = %s, exit_camera_name = %s
                WHERE id = %s
            """, (exit_image, merged_image, exit_camera_id, exit_camera_name, member_id))
        if movement_id:
            cur.execute("""
                UPDATE movement_log 
                SET exit_image = %s, merged_image = %s, exit_time = NOW(), event_type = 'EXIT',
                    exit_camera_id = %s, exit_camera_name = %s
                WHERE id = %s
            """, (exit_image, merged_image, exit_camera_id, exit_camera_name, movement_id))
        conn.commit()
    except Exception as e:
        log.error(f"Failed to update exit logs: {e}")
    finally:
        conn.close()


def track_staff_attendance(
    staff_id: int,
    staff_name: str | None = None,
    detected_at: datetime | None = None,
    entry_image: str | None = None,
    camera_name: str | None = "Camera",
    roles: list[str] | None = None
) -> dict:
    """One-row-per-day attendance: first entry fixed, last exit keeps updating."""
    detected_at = detected_at or datetime.now()
    today = detected_at.date()
    result = {"is_first_entry": False, "attendance_date": today}
    
    # Determine status based on camera roles
    # If it's an 'exit' role and NOT 'entry' or 'general', mark as OUT
    is_exit_only = False
    if roles:
        role_list = [r.lower() for r in roles]
        if 'exit' in role_list and 'entry' not in role_list and 'general' not in role_list:
            is_exit_only = True
    
    status_to_set = 'OUT' if is_exit_only else 'IN'
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        log.info(f"DEBUG: Tracking attendance for {staff_id} ({staff_name}) at {today}")
        cur.execute("""
            SELECT id FROM attendance
            WHERE staff_id = %s AND attendance_date = %s
        """, (staff_id, today))
        record = cur.fetchone()
        log.info(f"DEBUG: Existing record: {record}")

        try:
            if not record:
                log.info("DEBUG: Inserting NEW attendance record")
                result["is_first_entry"] = True
                cur.execute("""
                    INSERT INTO attendance
                    (
                        staff_id, staff_name, attendance_date, first_entry_time, last_exit_time,
                        in_time, out_time, entry_image, in_image, out_image, status, movement_count, day_status, timestamp, camera_name
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1, 'open', %s, %s)
                """, (
                    staff_id, staff_name, today, detected_at, detected_at,
                    detected_at, detected_at, entry_image, entry_image, entry_image, status_to_set, detected_at, camera_name
                ))
            else:
                log.info(f"DEBUG: Updating EXISTING attendance record {record['id']}")
                cur.execute("""
                    UPDATE attendance
                    SET last_exit_time = %s,
                        out_time = %s,
                        status = %s,
                        movement_count = COALESCE(movement_count, 0) + 1,
                        timestamp = %s,
                        staff_name = COALESCE(staff_name, %s),
                        entry_image = COALESCE(entry_image, %s),
                        in_image = COALESCE(in_image, %s),
                        out_image = %s,
                        camera_name = COALESCE(camera_name, %s)
                    WHERE staff_id = %s AND attendance_date = %s
                """, (
                    detected_at, detected_at, status_to_set, detected_at, staff_name, entry_image, entry_image, entry_image, camera_name,
                    staff_id, today
                ))
        except Exception as schema_exc:
            # Fallback for deployments where attendance schema is older/incomplete.
            conn.rollback()
            cur = conn.cursor()
            msg = str(schema_exc).lower()
            schema_mismatch = ("column" in msg and "does not exist" in msg) or "undefinedcolumn" in msg
            if not schema_mismatch:
                raise

            try:
                if not record:
                    result["is_first_entry"] = True
                    cur.execute("""
                        INSERT INTO attendance
                        (
                            staff_id, attendance_date, first_entry_time, last_exit_time,
                            in_time, out_time, in_image, status, movement_count, day_status, timestamp
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'IN', 1, 'open', %s)
                    """, (
                        staff_id, today, detected_at, detected_at,
                        detected_at, detected_at, entry_image, detected_at
                    ))
                else:
                    cur.execute("""
                        UPDATE attendance
                        SET last_exit_time = %s,
                            out_time = %s,
                            status = 'IN',
                            movement_count = COALESCE(movement_count, 0) + 1,
                            timestamp = %s,
                            in_image = COALESCE(in_image, %s)
                        WHERE staff_id = %s AND attendance_date = %s
                    """, (detected_at, detected_at, detected_at, entry_image, staff_id, today))
            except Exception:
                # Minimal legacy fallback: supports only classic columns.
                conn.rollback()
                cur = conn.cursor()
                if not record:
                    result["is_first_entry"] = True
                    cur.execute("""
                        INSERT INTO attendance
                        (staff_id, attendance_date, first_entry_time, last_exit_time)
                        VALUES (%s, %s, %s, %s)
                    """, (staff_id, today, detected_at, detected_at))
                else:
                    cur.execute("""
                        UPDATE attendance
                        SET last_exit_time = %s
                        WHERE staff_id = %s AND attendance_date = %s
                    """, (detected_at, staff_id, today))

        conn.commit()
        
        # SSE UPDATE
        sse_manager.announce({
            "staff_id": staff_id,
            "staff_name": staff_name,
            "attendance_date": str(today),
            "is_first_entry": result["is_first_entry"],
            "detected_at": detected_at.isoformat()
        }, event_type="attendance_update")

        return result
    except Exception as e:
        log.error(f"Failed to track staff attendance: {e}")
        return result
    finally:
        conn.close()


def mark_attendance(employee_name: str, phone_number: str = "", status: str = "", image_path: str = None) -> dict:
    """Legacy bridge for marking attendance. Attempts to match staff name."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Try to find staff by name
        cur.execute("SELECT id FROM staff_profiles WHERE name = %s", (employee_name,))
        row = cur.fetchone()
        if row:
            staff_db_id = row['id']
            track_staff_attendance(staff_db_id, staff_name=employee_name, entry_image=image_path)
            return {"status": "success", "staff_id": staff_db_id, "name": employee_name}
        else:
            # Log as unknown person detection if no match
            log_person("Telegram/Manual", "unknown", None, image_path or "", 0.0)
            return {"status": "unknown", "name": employee_name}
    except Exception as e:
        log.error(f"Error in legacy mark_attendance: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


def update_attendance_name(attendance_id: int, employee_name: str) -> bool:
    """Legacy bridge for updating attendance name."""
    # Since we use staff_id linked to staff_profiles now, 
    # 'updating name' is basically redundant if staff_id is set.
    # We'll just return True to avoid breaking callers.
    return True
