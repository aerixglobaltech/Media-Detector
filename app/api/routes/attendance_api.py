from flask import Blueprint, jsonify, request, Response, send_file
from typing import Any
from app.db.session import get_db_connection
from datetime import date, datetime
from app.core.security import login_required
import csv
import io
import logging
from pathlib import Path
from urllib.parse import quote

log = logging.getLogger("attendance_api")

attendance_bp = Blueprint('attendance_api', __name__, url_prefix='/api')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_ROOT = (PROJECT_ROOT / "static").resolve()
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _safe_image_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    try:
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    except Exception:
        return None

    if not resolved.exists() or not resolved.is_file():
        return None
    if resolved.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return None

    try:
        resolved.relative_to(PROJECT_ROOT)
        return resolved
    except ValueError:
        return None


def _member_log_image_url(raw_path: str) -> str:
    if not raw_path:
        return ""
    normalized = str(raw_path).strip().replace("\\", "/")
    if not normalized:
        return ""

    if normalized.startswith(("http://", "https://")):
        return normalized
    if normalized.startswith("/static/"):
        return normalized
    if normalized.startswith("static/"):
        return f"/{normalized}"

    # If it's potentially just a filename, try to find it in standard folders
    if "/" not in normalized and "\\" not in normalized:
        folders = ["uploads/snapshots", "uploads/movement", "uploads/merged", "uploads"]
        for folder in folders:
            candidate = STATIC_ROOT / folder / normalized
            if candidate.exists() and candidate.is_file():
                return f"/static/{folder}/{normalized}"

    safe_path = _safe_image_path(raw_path)
    if safe_path is None:
        return ""
    try:
        rel_static = safe_path.relative_to(STATIC_ROOT).as_posix()
        return f"/static/{rel_static}"
    except ValueError:
        return f"/api/member-log-image?path={quote(str(safe_path))}"


@attendance_bp.route('/member-log-image', methods=['GET'])
@login_required
def member_log_image():
    image_path = _safe_image_path(request.args.get("path", ""))
    if image_path is None:
        return jsonify({"status": "error", "message": "Image not found"}), 404
    return send_file(str(image_path))

@attendance_bp.route('/attendance/today', methods=['GET'])
def get_today_attendance():
    """Get today's attendance list."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT a.staff_id, s.name, a.first_entry_time, a.last_exit_time
            FROM attendance a
            JOIN staff_profiles s ON a.staff_id = s.id
            WHERE a.attendance_date = CURRENT_DATE
        """)
        rows = cur.fetchall()
        # Ensure JSON serializable
        data = []
        for r in rows:
            data.append({
                "staff_id": r['staff_id'],
                "name": r['name'],
                "first_entry_time": r['first_entry_time'].strftime("%Y-%m-%d %H:%M:%S") if r['first_entry_time'] else "-",
                "last_exit_time": r['last_exit_time'].strftime("%Y-%m-%d %H:%M:%S") if r['last_exit_time'] else "-"
            })
        return jsonify(data)
    except Exception as e:
        log.error(f"Error in get_today_attendance: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route('/logs/movement', methods=['GET'])
def get_movement_logs():
    """Get all movement logs."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM movement_log ORDER BY detected_at DESC")
        rows = cur.fetchall()
        # Ensure dates are JSON serializable
        formatted_rows = []
        for r in rows:
            formatted_rows.append({
                "id": r['id'],
                "camera_id": r['camera_id'],
                "camera_name": r.get('camera_name', r['camera_id']),
                "entry_image": _member_log_image_url(r.get('entry_image') or r.get('image_path')),
                "image_url": _member_log_image_url(r.get('entry_image') or r.get('image_path')),
                "exit_image": _member_log_image_url(r.get('exit_image')),
                "merged_image": _member_log_image_url(r.get('merged_image')),
                "entry_time": r.get('entry_time', r.get('detected_at')).strftime("%Y-%m-%d %H:%M:%S") if r.get('entry_time') or r.get('detected_at') else "-",
                "exit_time": r.get('exit_time').strftime("%Y-%m-%d %H:%M:%S") if r.get('exit_time') else "-"
            })
        return jsonify(formatted_rows)
    except Exception as e:
        log.error(f"Error in get_movement_logs: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route('/logs/persons', methods=['GET'])
def get_person_logs():
    """Get detected persons (staff + unknown)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT m.*, s.name as staff_name
            FROM member_timestamp m
            LEFT JOIN staff_profiles s ON m.staff_id = s.id
            ORDER BY m.detected_at DESC
        """)
        rows = cur.fetchall()
        data = []
        for r in rows:
            data.append({
                "id": r['id'],
                "person_type": r['person_type'],
                "staff_name": r['staff_name'] or "Unknown",
                "entry_time": (r.get('entry_time') or r.get('detected_at')).strftime("%Y-%m-%d %H:%M:%S") if r.get('entry_time') or r.get('detected_at') else "-",
                "exit_time": r.get('exit_time').strftime("%Y-%m-%d %H:%M:%S") if r.get('exit_time') else "-",
                "entry_image": _member_log_image_url(r.get('entry_image') or r.get('image_path')),
                "exit_image": _member_log_image_url(r.get('exit_image')),
                "merged_image": _member_log_image_url(r.get('merged_image'))
            })
        return jsonify(data)
    except Exception as e:
        log.error(f"Error in get_person_logs: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route('/logs/unknown', methods=['GET'])
def get_unknown_person_logs():
    """Get unknown persons list."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, camera_id, image_path, detected_at, entry_image, exit_image, merged_image, entry_time, exit_time FROM member_timestamp WHERE person_type = 'unknown' ORDER BY detected_at DESC")
        rows = cur.fetchall()
        # Ensure dates are JSON serializable
        formatted_rows = []
        for r in rows:
            formatted_rows.append({
                "id": r['id'],
                "camera_id": r['camera_id'],
                "entry_image": _member_log_image_url(r.get('entry_image') or r.get('image_path')),
                "exit_image": _member_log_image_url(r.get('exit_image')),
                "merged_image": _member_log_image_url(r.get('merged_image')),
                "entry_time": (r.get('entry_time') or r.get('detected_at')).strftime("%Y-%m-%d %H:%M:%S") if r.get('entry_time') or r.get('detected_at') else "-",
                "exit_time": r.get('exit_time').strftime("%Y-%m-%d %H:%M:%S") if r.get('exit_time') else "-"
            })
        return jsonify(formatted_rows)
    except Exception as e:
        log.error(f"Error in get_unknown_person_logs: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route('/attendance/count/<date_str>', methods=['GET'])
def get_staff_count(date_str):
    """Get staff entry/exit count for a specific day."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT staff_id, s.name, COUNT(*) as detections
            FROM member_timestamp m
            JOIN staff_profiles s ON m.staff_id = s.id
            WHERE person_type = 'staff' AND DATE(detected_at) = %s
            GROUP BY staff_id, s.name
        """, (date_str,))
        rows = cur.fetchall()
        # Ensure serializable
        data = []
        for r in rows:
            data.append({
                "staff_id": r['staff_id'],
                "name": r['name'],
                "detections": r['detections']
            })
        return jsonify(data)
    except Exception as e:
        log.error(f"Error in get_staff_count: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route('/member-logs', methods=['GET'])
@login_required
def get_member_logs():
    """Get member logs (staff + unknown) with pagination and filtering from member_timestamp."""
    return _fetch_logs_from_table(table_name='member_timestamp')

@attendance_bp.route('/general-movement', methods=['GET'])
@login_required
def get_general_movement_logs():
    """Get all movement logs (including MOVE_TRACK) with pagination and filtering from movement_log."""
    return _fetch_logs_from_table(table_name='movement_log')

def _fetch_logs_from_table(table_name: str):
    """Helper to fetch logs with pagination and filtering."""
    person_type = request.args.get('type') # staff, unknown
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', request.args.get('page_size', 20)))
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    period = request.args.get('period', 'all') # all, today
    offset = (page - 1) * limit
    q = request.args.get('q')

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        where_clauses = []
        params: list[Any] = []
        
        if person_type:
            where_clauses.append("m.person_type = %s")
            params.append(person_type)
        
        staff_id_filter = request.args.get('staff_id')
        if staff_id_filter:
            where_clauses.append("m.staff_id = %s")
            params.append(staff_id_filter)
        
        if period == 'today':
            where_clauses.append("m.detected_at::date = %s")
            params.append(datetime.now().date())
        else:
            if date_from:
                where_clauses.append("m.detected_at >= %s")
                params.append(date_from)
            if date_to:
                where_clauses.append("m.detected_at <= %s")
                params.append(date_to)
                
        if q:
            where_clauses.append("(COALESCE(s.name, '') ILIKE %s OR COALESCE(s.staff_id, '') ILIKE %s OR COALESCE(m.staff_name, '') ILIKE %s)")
            params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
            
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        # Get total count
        cur.execute(f"SELECT COUNT(*) as count FROM {table_name} m LEFT JOIN staff_profiles s ON m.staff_id = s.id {where_sql}", tuple(params))
        total = cur.fetchone()['count']
        
        # Get specific counts for summary
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE person_type ILIKE 'staff' AND (detected_at::date = CURRENT_DATE OR %s = 'all')", (period,))
        staff_count = cur.fetchone()['count']
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE person_type ILIKE 'unknown' AND (detected_at::date = CURRENT_DATE OR %s = 'all')", (period,))
        unknown_count = cur.fetchone()['count']
        
        # Get data
        cur.execute(f"""
            SELECT 
                m.id, 
                m.camera_id,
                COALESCE(m.entry_image, m.image_path) as entry_image,
                m.image_path, 
                m.exit_image,
                m.merged_image,
                m.person_type, 
                m.staff_id, 
                COALESCE(s.name, m.staff_name) AS member_name, 
                COALESCE(m.entry_time, m.detected_at) as entry_time,
                m.detected_at,
                m.exit_time,
                m.camera_name as entry_camera_name,
                m.exit_camera_name,
                s.staff_id as display_id
            FROM {table_name} m
            LEFT JOIN staff_profiles s ON m.staff_id = s.id
            {where_sql}
            ORDER BY m.detected_at DESC
            LIMIT %s OFFSET %s
        """, tuple(params + [limit, offset]))
        
        rows = cur.fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r['id'],
                "person_type": (r['person_type'] or "unknown").upper(),
                "member_id": r['staff_id'],
                "member_name": r['member_name'] or "-",
                "display_id": r['display_id'] or "-",
                "camera_id": r['camera_id'],
                "entry_camera_name": r['entry_camera_name'] or r['camera_id'] or "-",
                "entry_image": _member_log_image_url(r['image_path']),
                "image_url": _member_log_image_url(r['image_path']),
                "exit_image": _member_log_image_url(r['exit_image']),
                "merged_image": _member_log_image_url(r['merged_image']),
                "entry_time": r['entry_time'].strftime("%Y-%m-%d %H:%M:%S") if r['entry_time'] else "-",
                "exit_time": r['exit_time'].strftime("%Y-%m-%d %H:%M:%S") if r['exit_time'] else "-",
                "exit_camera_name": r.get('exit_camera_name') or "-"
            })
            
        return jsonify({
            "status": "success",
            "data": results,
            "counts": {
                "total": total,
                "staff": staff_count,
                "unknown": unknown_count
            },
            "pagination": {
                "total": total,
                "page": page,
                "limit": limit,
                "pages": (int(total) + limit - 1) // limit if limit > 0 else 0
            }
        })
    except Exception as e:
        log.error(f"Error fetching logs from {table_name}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route("/member-logs/export", methods=['GET'])
@login_required
def export_member_logs():
    """Export member logs from member_timestamp as CSV."""
    return _export_logs_from_table(table_name='member_timestamp', filename_prefix='member_logs')

@attendance_bp.route("/general-movement/export", methods=['GET'])
@login_required
def export_general_movement_logs():
    """Export general movement logs from movement_log as CSV."""
    return _export_logs_from_table(table_name='movement_log', filename_prefix='general_movement')

def _export_logs_from_table(table_name: str, filename_prefix: str):
    """Helper to export logs from a specific table as CSV."""
    person_type = request.args.get("type")
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        where_clauses = []
        params: list[Any] = []
        
        if person_type:
            where_clauses.append("m.person_type = %s")
            params.append(person_type)
        if date_from:
            where_clauses.append("m.detected_at >= %s")
            params.append(date_from)
        if date_to:
            where_clauses.append("m.detected_at <= %s")
            params.append(date_to)
            
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        cur.execute(f"""
            SELECT 
                m.detected_at,
                m.person_type, 
                s.staff_id as display_id,
                COALESCE(s.name, m.staff_name) AS staff_name,
                m.camera_id,
                m.confidence_score
            FROM {table_name} m
            LEFT JOIN staff_profiles s ON m.staff_id = s.id
            {where_sql}
            ORDER BY m.detected_at DESC
        """, tuple(params))
        
        rows = cur.fetchall()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Detection Time', 'Person Type', 'Staff ID', 'Staff Name', 'Camera', 'Confidence Score'])
        
        for r in rows:
            writer.writerow([
                r['detected_at'].strftime('%Y-%m-%d %H:%M:%S') if r['detected_at'] else '',
                r['person_type'].capitalize() if r['person_type'] else 'Unknown',
                r['display_id'] or '',
                r['staff_name'] or '',
                r['camera_id'] or '',
                f"{r['confidence_score']:.2f}" if r['confidence_score'] else '0.00'
            ])
            
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        log.error(f"Error exporting {table_name}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route('/telegram-alerts', methods=['GET'])
@login_required
def get_telegram_alerts():
    """Get telegram notifications with pagination and filtering."""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', request.args.get('page_size', 20)))
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    period = request.args.get('period', 'all')
    offset = (page - 1) * limit
    q = request.args.get('q')

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        where_clauses = []
        params: list[Any] = []
        
        if period == 'today':
            where_clauses.append("timestamp::date = %s")
            params.append(datetime.now().date())
        else:
            if date_from:
                where_clauses.append("timestamp >= %s")
                params.append(date_from)
            if date_to:
                where_clauses.append("timestamp <= %s")
                params.append(date_to)
        if q:
            where_clauses.append("(message_text ILIKE %s OR camera_name ILIKE %s OR action ILIKE %s)")
            params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
            
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        cur.execute(f"SELECT COUNT(*) as count FROM telegram_alerts {where_sql}", tuple(params))
        total = cur.fetchone()['count']

        # Get summary counts for cards (handle empty WHERE)
        base_filter = where_sql if where_sql else "WHERE TRUE"
        cur.execute(f"SELECT COUNT(*) as count FROM telegram_alerts {base_filter} AND status ILIKE 'sent'", tuple(params))
        sent_count = cur.fetchone()['count']
        cur.execute(f"SELECT COUNT(*) as count FROM telegram_alerts {base_filter} AND status NOT ILIKE 'sent'", tuple(params))
        failed_count = cur.fetchone()['count']
        
        cur.execute(f"""
            SELECT id, track_id, camera_name, action, message_text, status, timestamp, chat_id
            FROM telegram_alerts
            {where_sql}
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
        """, tuple(params + [limit, offset]))
        
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r['id'],
                "track_id": r['track_id'],
                "camera_name": r['camera_name'] or "-",
                "action": r['action'] or "-",
                "message_text": r['message_text'],
                "status": r['status'],
                "chat_id": r['chat_id'] or "-",
                "timestamp": r['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if r['timestamp'] else "-"
            })
            
        return jsonify({
            "status": "success",
            "data": results,
            "counts": {
                "total": total,
                "sent": sent_count,
                "failed": failed_count
            },
            "pagination": {
                "total": total,
                "sent_count": sent_count,
                "failed_count": failed_count,
                "page": page,
                "limit": limit,
                "pages": (total + limit - 1) // limit if limit > 0 else 0
            }
        })
    except Exception as e:
        log.error(f"Error fetching telegram alerts: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route("/telegram-alerts/export", methods=['GET'])
@login_required
def export_telegram_alerts():
    """Export telegram alerts as CSV."""
    period = request.args.get('period', 'all')
    q = request.args.get('q')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        where_clauses = []
        params: list[Any] = []
        
        if period == 'today':
            where_clauses.append("timestamp::date = %s")
            params.append(datetime.now().date())
        else:
            if date_from:
                where_clauses.append("timestamp >= %s")
                params.append(date_from)
            if date_to:
                where_clauses.append("timestamp <= %s")
                params.append(date_to)
        if q:
            where_clauses.append("(message_text ILIKE %s OR camera_name ILIKE %s OR action ILIKE %s)")
            params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
            
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        cur.execute(f"""
            SELECT id, track_id, camera_name, action, message_text, status, timestamp, chat_id
            FROM telegram_alerts
            {where_sql}
            ORDER BY timestamp DESC
        """, tuple(params))
        
        rows = cur.fetchall()
        
        # Generate XLSX for professional Excel experience
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font
            
            wb = Workbook()
            ws = wb.active
            ws.title = "Telegram Alerts"
            
            # Header with bold font
            headers_list = ['Time', 'Track ID', 'Camera', 'Action', 'Status', 'Chat ID', 'Message']
            ws.append(headers_list)
            for cell in ws[1]:
                cell.font = Font(bold=True)
                
            for r in rows:
                ts_str = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if r['timestamp'] else ''
                ws.append([
                    ts_str,
                    r['track_id'] or '',
                    r['camera_name'] or '',
                    r['action'] or '',
                    r['status'] or '',
                    r['chat_id'] or '',
                    r['message_text'] or ''
                ])
                
            # Basic column width adjustment
            for col in ws.columns:
                max_length: int = 0
                column_letter = col[0].column_letter
                for cell in col:
                    try:
                        val = str(cell.value) if cell.value is not None else ""
                        if len(val) > max_length:
                            max_length = len(val)
                    except: pass
                ws.column_dimensions[column_letter].width = min(float(max_length) + 2.0, 50.0)

            output = io.BytesIO()
            wb.save(output)
            output.seek(0)
            
            filename = f"telegram_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return Response(
                output.read(),
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-disposition": f"attachment; filename={filename}"}
            )
        except ImportError:
            log.warning("openpyxl not found, falling back to CSV with BOM")
            # Fallback to CSV if openpyxl fails for some reason
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Time', 'Track ID', 'Camera', 'Action', 'Status', 'Chat ID', 'Message'])
            for r in rows:
                ts_str = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if r['timestamp'] else ''
                writer.writerow([ts_str, r['track_id'] or '', r['camera_name'] or '', r['action'] or '', r['status'] or '', r['chat_id'] or '', r['message_text'] or ''])
            
            csv_data = output.getvalue()
            return Response(
                csv_data.encode('utf-8-sig'),
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename=telegram_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx.csv"}
            )
            
    except Exception as e:
        log.error(f"Error exporting telegram alerts: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

