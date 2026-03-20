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
@attendance_bp.route('/general-movement', methods=['GET'])
@login_required
def get_member_logs():
    """Get member logs (staff + unknown) with pagination and filtering."""
    person_type = request.args.get('type') # staff, unknown
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', request.args.get('page_size', 20)))
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    period = request.args.get('period', 'all') # all, today
    offset = (page - 1) * limit
    
    # Simple search for staff name or display_id
    q = request.args.get('q')

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        where_clauses = []
        params: list[Any] = []
        
        if person_type:
            where_clauses.append("m.person_type = %s")
            params.append(person_type)
        
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
            where_clauses.append("(s.name ILIKE %s OR s.staff_id ILIKE %s)")
            params.append(f"%{q}%")
            params.append(f"%{q}%")
            
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        # Get total and counts for pagination & summary
        cur.execute(f"""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE m.person_type = 'staff') as staff_count,
                COUNT(*) FILTER (WHERE m.person_type != 'staff' OR m.person_type IS NULL) as unknown_count
            FROM member_timestamp m 
            LEFT JOIN staff_profiles s ON m.staff_id = s.id 
            {where_sql}
        """, tuple(params))
        counts_row = cur.fetchone()
        total = counts_row['total'] if counts_row else 0
        staff_count = counts_row['staff_count'] if counts_row else 0
        unknown_count = counts_row['unknown_count'] if counts_row else 0
        
        # Get data
        cur.execute(f"""
            SELECT 
                m.id, 
                m.camera_id,
                m.entry_image,
                m.image_path, 
                m.exit_image,
                m.merged_image,
                m.person_type, 
                m.staff_id, 
                s.name AS member_name, 
                m.entry_time,
                m.detected_at,
                m.exit_time,
                s.staff_id as display_id
            FROM member_timestamp m
            LEFT JOIN staff_profiles s ON m.staff_id = s.id
            {where_sql}
            ORDER BY m.detected_at DESC
            LIMIT %s OFFSET %s
        """, tuple(params + [limit, offset]))
        
        rows = cur.fetchall()
        
        # Format for JSON
        results = []
        for r in rows:
            p_type = r['person_type'] or "unknown"
            image_path = r['image_path'] if r['image_path'] else ""
            results.append({
                "id": r['id'],
                "person_type": p_type.upper(),
                "member_id": r['staff_id'],
                "member_name": r['member_name'] or "-",
                "display_id": r['display_id'] or "-",
                "camera_id": r['camera_id'],
                "entry_image": _member_log_image_url(r['entry_image'] or r['image_path']),
                "image_url": _member_log_image_url(r['entry_image'] or r['image_path']),
                "exit_image": _member_log_image_url(r['exit_image']),
                "merged_image": _member_log_image_url(r['merged_image']),
                "entry_time": (r['entry_time'] or r['detected_at']).strftime("%Y-%m-%d %H:%M:%S") if r['entry_time'] or r['detected_at'] else "-",
                "exit_time": r['exit_time'].strftime("%Y-%m-%d %H:%M:%S") if r['exit_time'] else "-"
            })
            
        return jsonify({
            "status": "success",
            "data": results,
            "counts": {
                "total": total,
                "staff": staff_count,
                "unknown": unknown_count
            },
            "period": period,
            "pagination": {
                "total": total,
                "page": page,
                "limit": limit,
                "pages": (int(total) + limit - 1) // limit if limit > 0 else 0
            }
        })
    except Exception as e:
        log.error(f"Error fetching member logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route("/member-logs/export", methods=['GET'])
@attendance_bp.route("/general-movement/export", methods=['GET'])
@login_required
def export_member_logs():
    """Export member logs as CSV."""
    
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
                s.name AS staff_name,
                m.camera_id,
                m.confidence_score
            FROM member_timestamp m
            LEFT JOIN staff_profiles s ON m.staff_id = s.id
            {where_sql}
            ORDER BY m.detected_at DESC
        """, tuple(params))
        
        rows = cur.fetchall()
        
        # Generate CSV
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
            headers={"Content-disposition": f"attachment; filename=member_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
            
    except Exception as e:
        log.error(f"Error exporting member logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route("/general-movement", methods=['GET'])
@login_required
def get_general_movement():
    """Get general movement logs with pagination, filtering, and counts."""
    person_type = request.args.get('type')  # staff, unknown
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', request.args.get('page_size', 20)))
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    period = request.args.get('period', 'all')  # all, today
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
            where_clauses.append("(s.name ILIKE %s OR s.staff_id ILIKE %s)")
            params.append(f"%{q}%")
            params.append(f"%{q}%")
            
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        # Get total count
        cur.execute(f"SELECT COUNT(*) as count FROM member_timestamp m LEFT JOIN staff_profiles s ON m.staff_id = s.id {where_sql}", tuple(params))
        total_row = cur.fetchone()
        total = total_row['count'] if total_row else 0
        
        # Get staff and unknown counts for summary
        cur.execute(f"SELECT COUNT(*) FILTER (WHERE person_type = 'staff') as staff_count, COUNT(*) FILTER (WHERE person_type = 'unknown') as unknown_count FROM member_timestamp m LEFT JOIN staff_profiles s ON m.staff_id = s.id {where_sql}", tuple(params))
        counts_res = cur.fetchone()
        staff_count = counts_res['staff_count'] or 0
        unknown_count = counts_res['unknown_count'] or 0
        
        # Get data
        cur.execute(f"""
            SELECT 
                m.id, m.camera_id, m.entry_image, m.image_path, m.exit_image, m.merged_image,
                m.person_type, m.staff_id, s.name AS member_name, m.entry_time, m.detected_at, m.exit_time,
                s.staff_id as display_id
            FROM member_timestamp m
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
                "entry_image": _member_log_image_url(r['entry_image'] or r['image_path']),
                "exit_image": _member_log_image_url(r['exit_image']),
                "merged_image": _member_log_image_url(r['merged_image']),
                "entry_time": (r['entry_time'] or r['detected_at']).strftime("%Y-%m-%d %H:%M:%S") if r['entry_time'] or r['detected_at'] else "-",
                "exit_time": r['exit_time'].strftime("%Y-%m-%d %H:%M:%S") if r['exit_time'] else "-"
            })
            
        return jsonify({
            "status": "success",
            "data": results,
            "counts": {"staff": staff_count, "unknown": unknown_count},
            "pagination": {
                "total": total,
                "page": page,
                "limit": limit,
                "pages": (int(total) + limit - 1) // limit if limit > 0 else 0
            }
        })
    except Exception as e:
        log.error(f"Error fetching general movement logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@attendance_bp.route("/general-movement/export", methods=['GET'])
@login_required
def export_general_movement():
    """Export general movement logs as CSV."""
    person_type = request.args.get("type")
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    q = request.args.get('q')
    period = request.args.get('period', 'all')
    
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        where_clauses = []
        params: list[Any] = []
        
        if person_type:
            where_clauses.append("m.person_type = %s")
            params.append(person_type)
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
            where_clauses.append("(s.name ILIKE %s OR s.staff_id ILIKE %s)")
            params.append(f"%{q}%")
            params.append(f"%{q}%")
            
        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)
            
        cur.execute(f"""
            SELECT 
                m.detected_at, m.person_type, s.staff_id as display_id, s.name AS staff_name,
                m.camera_id, m.confidence_score, m.entry_time, m.exit_time
            FROM member_timestamp m
            LEFT JOIN staff_profiles s ON m.staff_id = s.id
            {where_sql}
            ORDER BY m.detected_at DESC
        """, tuple(params))
        
        rows = cur.fetchall()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Detection Time', 'Person Type', 'Staff ID', 'Staff Name', 'Camera', 'Confidence Score', 'Entry Time', 'Exit Time'])
        
        for r in rows:
            writer.writerow([
                r['detected_at'].strftime('%Y-%m-%d %H:%M:%S') if r['detected_at'] else '',
                r['person_type'].capitalize() if r['person_type'] else 'Unknown',
                r['display_id'] or '',
                r['staff_name'] or '',
                r['camera_id'] or '',
                f"{r['confidence_score']:.2f}" if r['confidence_score'] else '0.00',
                (r['entry_time'] or r['detected_at']).strftime('%Y-%m-%d %H:%M:%S') if r['entry_time'] or r['detected_at'] else '-',
                r['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if r['exit_time'] else '-'
            ])
            
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=general_movement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        log.error(f"Error exporting general movement logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()
