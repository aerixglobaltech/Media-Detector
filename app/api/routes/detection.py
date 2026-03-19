"""
app/routes/api.py  –  General API Routes Blueprint
──────────────────────────────────────────────────────────────────────────────
Provides JSON API endpoints for:
  • Real-time detection stats
  • AI feature toggles
  • Dashboard info
  • Telegram notifications
  • Settings info
  • User profile
  • Staff photo management (upload / list / delete)

Routes:
  GET        /api/stats              → Live detection stats
  GET/POST   /api/toggles            → Read/write AI feature toggles
  GET        /api/dashboard_info     → Camera online/offline counts + motion events
  POST       /api/notify_manual      → Send a manual Telegram alert
  GET        /api/settings_info      → Current .env credentials shown on settings page
  GET        /api/user_profile       → Logged-in user's profile data
  POST       /api/upload_staff       → Upload staff reference photos
  GET        /api/staff_profiles     → List all staff profiles with thumbnails
  GET        /api/staff_profiles/<name>  → Photos for a specific staff member
  DELETE     /api/staff_profiles/<name>/<file>  → Delete a single photo
  DELETE     /api/staff_profiles/<name>          → Delete entire staff profile
"""

from __future__ import annotations

import logging
import os
import shutil

from flask import Blueprint, jsonify, request, session, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
import io

from app.core.security import login_required
from app.db.session import get_db_connection
from app.core.extensions import cam_mgr, notifier, AI_TOGGLES
from app.api.routes.camera import get_all_cameras

log = logging.getLogger("api")

api_bp = Blueprint("detection", __name__)


# ─── Detection Stats ──────────────────────────────────────────────────────────

@api_bp.route("/api/stats")
def api_stats():
    """Live detection stats: persons, actions, emotions, FPS."""
    return jsonify(cam_mgr.get_stats())


# ─── AI Feature Toggles ───────────────────────────────────────────────────────

@api_bp.route("/api/toggles", methods=["GET", "POST"])
def api_toggles():
    """GET returns current toggle states; POST updates them."""
    if request.method == "POST":
        data = request.get_json(force=True)
        if "person"  in data: AI_TOGGLES["person"]  = bool(data["person"])
        if "action"  in data: AI_TOGGLES["action"]  = bool(data["action"])
        if "emotion" in data: AI_TOGGLES["emotion"] = bool(data["emotion"])
    return jsonify(AI_TOGGLES)


# ─── Dashboard Info ───────────────────────────────────────────────────────────

@api_bp.route("/api/dashboard_info")
def api_dashboard_info():
    """Stats card data: camera counts, staff counts, active users."""
    staff_count = 0
    user_count = 0
    db_status = "error"
    
    try:
        # Get DB counts with explicit aliasing for RealDictCursor compatibility
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) as count FROM staff_profiles")
        row = cur.fetchone()
        if row: staff_count = int(row['count'])
        
        cur.execute("SELECT COUNT(*) as count FROM users")
        row = cur.fetchone()
        if row: user_count = int(row['count'])
        
        cur.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        log.error("DASHBOARD DB ERROR: %s", e)
        db_status = f"error: {str(e)}"

    user_email = session.get("user")
    cams = get_all_cameras(user_email=user_email)
    
    online_count  = 0
    for c in cams:
        status = c.get("status", "").lower()
        if "online" in status or "🟢" in status:
            online_count += 1
            
    total_cameras = len(cams)

    log.info("DASHBOARD_INFO [v2] -> Online: %d, Total: %d, Staff: %d, Users: %d, DB: %s", 
             online_count, total_cameras, staff_count, user_count, db_status)

    # System Health logic
    ai_status = "error"
    if cam_mgr and hasattr(cam_mgr, "_pipeline") and cam_mgr._pipeline:
        # Check if the AI thread is running
        if hasattr(cam_mgr._pipeline, "is_alive") and cam_mgr._pipeline.is_alive():
            ai_status = "running"
        else:
            ai_status = "idle"

    # Media server is "active" if we can see cameras
    media_status = "active" if total_cameras > 0 else "idle"

    # Use unique keys to avoid conflict with any old cached JS
    return jsonify({
        "dashboard_v":   "2.0",
        "cam_online":    online_count,
        "cam_total":     total_cameras,
        "staff_total":   staff_count,
        "user_total":    user_count,
        "db_status":     db_status,
        "ai_status":     ai_status,
        "media_status":  media_status
    })


@api_bp.route("/api/dashboard_v3")
def api_dashboard_v3():
    """Version 3 of Dashboard API: includes trend and presence."""
    staff_count = 0
    user_count = 0
    db_status = "error"
    online_count = 0
    total_cameras = 0
    attendance_trend = []
    daily_presence = []
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Staff count
        cur.execute("SELECT COUNT(*)::int as count FROM staff_profiles")
        row = cur.fetchone()
        if row: staff_count = int(row['count'] or 0)
        
        # 2. User count
        cur.execute("SELECT COUNT(*)::int as count FROM users")
        row = cur.fetchone()
        if row: user_count = int(row['count'] or 0)
        
        # 3. Attendance Trend (Last 7 days)
        # Using attendance table for daily unique entries/exits
        cur.execute("""
            SELECT 
                attendance_date as day,
                COUNT(first_entry_time)::int as check_in,
                COUNT(last_exit_time)::int as check_out
            FROM attendance
            WHERE attendance_date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY attendance_date
            ORDER BY attendance_date ASC
        """)
        rows = cur.fetchall()
        for r in rows:
            attendance_trend.append({
                "day": r['day'].strftime("%m/%d") if hasattr(r['day'], 'strftime') else str(r['day']),
                "check_in": int(r['check_in'] or 0),
                "check_out": int(r['check_out'] or 0)
            })
            
        # 4. Daily Presence (Today)
        cur.execute("""
            SELECT s.name, a.status, a.last_exit_time as last_seen
            FROM attendance a
            JOIN staff_profiles s ON a.staff_id = s.id
            WHERE a.attendance_date = CURRENT_DATE
            ORDER BY a.last_exit_time DESC
        """)
        rows = cur.fetchall()
        for r in rows:
            daily_presence.append({
                "name": r['name'],
                "status": r['status'] or 'IN',
                "last_seen": r['last_seen'].isoformat() if hasattr(r['last_seen'], 'isoformat') else str(r['last_seen'])
            })
            
        cur.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        log.error("DASHBOARD_V3 ERROR: %s", e)
        db_status = f"error: {str(e)}"

    # 5. Camera counts
    user_email = session.get("user")
    cams = get_all_cameras(user_email=user_email)
    for c in cams:
        st = c.get("status", "").lower()
        if "online" in st or "🟢" in st:
            online_count += 1
    total_cameras = len(cams)

    # 6. AI & Media status
    ai_status = "error"
    if cam_mgr and hasattr(cam_mgr, "_pipeline") and cam_mgr._pipeline:
        if hasattr(cam_mgr._pipeline, "is_alive") and cam_mgr._pipeline.is_alive():
            ai_status = "running"
        else:
            ai_status = "idle"
    media_status = "active" if total_cameras > 0 else "idle"

    return jsonify({
        "dashboard_v": "3.0",
        "cam_online": online_count,
        "cam_total": total_cameras,
        "staff_total": staff_count,
        "user_total": user_count,
        "db_status": db_status,
        "ai_status": ai_status,
        "media_status": media_status,
        "attendance_trend": attendance_trend,
        "daily_presence": daily_presence
    })


# ─── Telegram Notifications ───────────────────────────────────────────────────

@api_bp.route("/api/notify_manual", methods=["POST"])
def api_notify_manual():
    """Send a manual Telegram alert triggered from the UI."""
    msg = (
        "📢 *MANUAL ALERT* from MISSION CONTROL UI\n\n"
        "A user has triggered a manual notification from the control panel."
    )
    success = notifier.send_message(msg)
    return jsonify({"success": success})


# ─── Settings ─────────────────────────────────────────────────────────────────

@api_bp.route("/api/telegram_bots", methods=["GET"])
@login_required
def api_list_telegram_bots():
    """List all configured Telegram bots."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, bot_name, bot_token, chat_ids, phone_number, is_active FROM telegram_bots ORDER BY id ASC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        log.error("Error listing bots: %s", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/telegram_bots", methods=["POST"])
@login_required
def api_upsert_telegram_bot():
    """Add or update a Telegram bot."""
    try:
        data = request.get_json(force=True)
        bot_id = data.get("id")
        name   = data.get("bot_name", "Telegram Bot")
        token  = data.get("bot_token")
        chats  = data.get("chat_ids")
        phone  = data.get("phone_number")
        
        if not token or not chats:
            return jsonify({"success": False, "error": "Token and Chat IDs are required"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        if bot_id:
            cur.execute(
                "UPDATE telegram_bots SET bot_name=%s, bot_token=%s, chat_ids=%s, phone_number=%s WHERE id=%s",
                (name, token, chats, phone, bot_id)
            )
        else:
            cur.execute(
                "INSERT INTO telegram_bots (bot_name, bot_token, chat_ids, phone_number) VALUES (%s, %s, %s, %s)",
                (name, token, chats, phone)
            )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        log.error("Error upserting bot: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/telegram_bots/<int:bot_id>", methods=["DELETE"])
@login_required
def api_delete_telegram_bot(bot_id):
    """Delete a Telegram bot."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM telegram_bots WHERE id=%s", (bot_id,))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        log.error("Error deleting bot: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/telegram_bots/<int:bot_id>/toggle", methods=["POST"])
@login_required
def api_toggle_telegram_bot(bot_id):
    """Toggle a bot's active status."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE telegram_bots SET is_active = NOT is_active WHERE id=%s RETURNING is_active", (bot_id,))
        res = cur.fetchone()
        if not res:
            return jsonify({"success": False, "error": "Bot not found"}), 404
        new_status = res['is_active']
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True, "active": new_status})
    except Exception as e:
        log.error("Error toggling bot: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ─── User Profile ─────────────────────────────────────────────────────────────

@api_bp.route("/api/user_profile")
@login_required
def api_user_profile():
    """Return the logged-in user's name, email, and company."""
    email = session.get("user")
    if not email:
        return jsonify({"error": "Not logged in"}), 401

    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT u.email, u.name, u.company, r.name as role_name 
        FROM users u
        LEFT JOIN roles r ON u.role_id = r.id
        WHERE u.email = %s
    """, (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if user:
        return jsonify({
            "email": user["email"], 
            "name": user["name"], 
            "company": user["company"],
            "role": user["role_name"]
        })
    return jsonify({"error": "User not found"}), 404


# ─── Staff Photo Management ───────────────────────────────────────────────────

def _upload_folder() -> str:
    """Return the absolute path to the staff photo upload folder."""
    from flask import current_app
    return current_app.config["UPLOAD_FOLDER"]


@api_bp.route("/api/upload_staff", methods=["POST"])
@login_required
def api_upload_staff():
    """Upload reference photos for a named staff member."""
    if "staff_name" not in request.form:
        return jsonify({"success": False, "error": "Member name is required"})

    staff_name = secure_filename(request.form["staff_name"])
    if not staff_name:
        return jsonify({"success": False, "error": "Invalid member name"})

    is_update = request.form.get("is_update") == "true"
    files = request.files.getlist("photos") if "photos" in request.files else []
    
    if not is_update:
        if not files or (len(files) == 1 and files[0].filename == ""):
            return jsonify({"success": False, "error": "No valid files selected"})

    original_name = request.form.get("original_name", "")
    sanitized_original = secure_filename(original_name) if original_name else ""
    
    if is_update and sanitized_original:
        # We are updating. Use the original folder.
        staff_dir = os.path.join(_upload_folder(), sanitized_original)
        
        # If the name has changed, rename the folder first
        if sanitized_original != staff_name:
            new_dir = os.path.join(_upload_folder(), staff_name)
            if os.path.exists(staff_dir):
                os.rename(staff_dir, new_dir)
            staff_dir = new_dir
    else:
        # We are creating new
        staff_dir = os.path.join(_upload_folder(), staff_name)
        
    os.makedirs(staff_dir, exist_ok=True)

    email = request.form.get("email", "")
    phone = request.form.get("phone", "")
    staff_id_val = request.form.get("staff_id", "").strip()
    address = request.form.get("address", "")
    communication = request.form.get("communication", "") # JSON string from frontend

    # Auto-generate Staff ID if missing
    if not staff_id_val:
        from datetime import datetime
        today_prefix = datetime.now().strftime("STF-%Y%m%d-")
        
        conn_temp = get_db_connection()
        cur_temp = conn_temp.cursor()
        try:
            # Find the highest sequence number for today
            cur_temp.execute(
                "SELECT staff_id FROM staff_profiles WHERE staff_id LIKE %s ORDER BY staff_id DESC LIMIT 1",
                (today_prefix + "%",)
            )
            last_id_row = cur_temp.fetchone()
            if last_id_row and last_id_row['staff_id']:
                last_id = last_id_row['staff_id']
                try:
                    last_seq = int(last_id.split('-')[-1])
                    new_seq = last_seq + 1
                except (IndexError, ValueError):
                    new_seq = 1
            else:
                new_seq = 1
            
            staff_id_val = f"{today_prefix}{new_seq:04d}"
        except Exception as e:
            log.error("Error auto-generating staff_id: %s", e)
            # fallback to timestamp if sequence fails
            staff_id_val = datetime.now().strftime("STF-%Y%m%d%H%M%S")
        finally:
            cur_temp.close()
            conn_temp.close()

    saved_count = 0
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(staff_dir, filename))
            saved_count += 1

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if is_update and original_name:
            # FORCE update based on the EXACT original name stored in DB
            cur.execute("""
                UPDATE staff_profiles 
                SET name = %s, email = %s, phone = %s, staff_id = %s, address = %s, communication = %s, folder_path = %s 
                WHERE name = %s
            """, (staff_name, email, phone, staff_id_val, address, communication, staff_dir, original_name))
            
            # If no rows were updated (maybe name was already changed/sanitized), 
            # try finding by sanitized original name
            if cur.rowcount == 0 and sanitized_original:
                cur.execute("""
                    UPDATE staff_profiles 
                    SET name = %s, email = %s, phone = %s, address = %s, communication = %s, folder_path = %s 
                    WHERE name = %s
                """, (staff_name, email, phone, address, communication, staff_dir, sanitized_original))
        else:
            # Pure CREATE or UPSERT by new name
            cur.execute("SELECT id FROM staff_profiles WHERE name = %s", (staff_name,))
            row = cur.fetchone()
            if row:
                cur.execute("""
                    UPDATE staff_profiles 
                    SET email = %s, phone = %s, staff_id = %s, address = %s, communication = %s, folder_path = %s 
                    WHERE id = %s
                """, (email, phone, staff_id_val, address, communication, staff_dir, row["id"]))
            else:
                cur.execute("""
                    INSERT INTO staff_profiles (name, email, phone, staff_id, address, communication, folder_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (staff_name, email, phone, staff_id_val, address, communication, staff_dir))
        conn.commit()
    except Exception as e:
        log.error("DB error saving staff profile: %s", e)
    finally:
        if conn:
            conn.close()

    return jsonify({"success": True, "saved_count": saved_count})


@api_bp.route("/api/staff_profiles", methods=["GET"])
@login_required
def api_staff_profiles():
    """List all staff profiles with a thumbnail URL and photo count."""
    profiles     = []
    upload_folder = _upload_folder()
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM staff_profiles ORDER BY created_at DESC")
        rows = cur.fetchall()
        for row in rows:
            staff_name = row["name"]
            safe_name = secure_filename(staff_name)
            staff_dir = os.path.join(upload_folder, safe_name)
            photo_count = 0
            thumbnail_url = ""
            if os.path.exists(staff_dir) and os.path.isdir(staff_dir):
                photos = [f for f in os.listdir(staff_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                photo_count = len(photos)
                if photos:
                    thumbnail_url = url_for("static", filename=f"uploads/{safe_name}/{photos[0]}")
            profiles.append({
                "id":          row["id"],
                "name":        staff_name,
                "staff_id":    row["staff_id"] or "",
                "email":       row["email"],
                "phone":       row["phone"],
                "address":     row["address"],
                "communication": row["communication"] or "",
                "status":      row["status"] or "active",
                "photo_count": photo_count,
                "thumbnail":   thumbnail_url,
            })
    except Exception as e:
        log.error("Failed to load staff profiles from DB: %s", e)
    finally:
        if conn:
            conn.close()

    return jsonify({"profiles": profiles})


@api_bp.route("/api/staff_profiles/<staff_name>", methods=["GET", "DELETE"])
@login_required
def api_staff_profile(staff_name: str):
    """
    GET  → list photos for one staff member
    DELETE → remove the entire staff profile folder
    """
    safe_name = secure_filename(staff_name)
    staff_dir = os.path.join(_upload_folder(), safe_name)

    if request.method == "DELETE":
        if os.path.isdir(staff_dir):
            shutil.rmtree(staff_dir)
            
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM staff_profiles WHERE name = %s", (staff_name,))
            conn.commit()
        except Exception as e:
            log.error("Error deleting staff from DB: %s", e)
        finally:
            if conn:
                conn.close()
                
        return jsonify({"success": True})

    # GET
    if not os.path.isdir(staff_dir):
        # Even if directory is missing (0 photos), we should still check the DB
        pass

    staff_data = {"name": staff_name, "staff_id": "", "email": "", "phone": "", "address": "", "communication": ""}
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        requested_id = request.args.get("id", type=int)
        if requested_id is not None:
            cur.execute(
                "SELECT name, email, phone, address, communication FROM staff_profiles WHERE id = %s",
                (requested_id,),
            )
            row = cur.fetchone()
            if not row:
                cur.execute(
                    "SELECT name, email, phone, address, communication FROM staff_profiles WHERE name = %s",
                    (staff_name,),
                )
                row = cur.fetchone()
        else:
            cur.execute(
                "SELECT name, staff_id, email, phone, address, communication FROM staff_profiles WHERE name = %s",
                (staff_name,),
            )
            row = cur.fetchone()

        if row:
            resolved_name = row["name"] or staff_name
            staff_data["name"] = resolved_name
            staff_data["staff_id"] = row["staff_id"] or ""
            staff_data["email"] = row["email"] or ""
            staff_data["phone"] = row["phone"] or ""
            staff_data["address"] = row["address"] or ""
            staff_data["communication"] = row["communication"] or ""
            safe_name = secure_filename(resolved_name)
            staff_dir = os.path.join(_upload_folder(), safe_name)
            log.info("Loaded staff data with communication: %s", staff_data["communication"])
    except Exception as e:
        log.error("DB Error fetching single staff: %s", e)
    finally:
        if conn:
            conn.close()

    photos = []
    if os.path.exists(staff_dir) and os.path.isdir(staff_dir):
        photos = [
            f for f in os.listdir(staff_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    urls = [
        {
            "filename": p,
            "url": url_for("static", filename=f"uploads/{safe_name}/{p}"),
        }
        for p in photos
    ]
    return jsonify({"staff": staff_data, "photos": urls})


@api_bp.route("/api/staff_profiles/<staff_name>/toggle", methods=["POST"])
@login_required
def api_toggle_staff_status(staff_name: str):
    """Toggle staff status between active and inactive."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT status FROM staff_profiles WHERE name = %s", (staff_name,))
        row = cur.fetchone()
        if not row:
            return jsonify({"success": False, "error": "Staff not found"}), 404
        
        new_status = "inactive" if row["status"] == "active" else "active"
        cur.execute("UPDATE staff_profiles SET status = %s WHERE name = %s", (new_status, staff_name))
        conn.commit()
        return jsonify({"success": True, "new_status": new_status})
    except Exception as e:
        log.error("Error toggling staff status: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if conn:
            conn.close()

@api_bp.route("/api/staff_profiles/<staff_name>/<filename>", methods=["DELETE"])
@login_required
def api_delete_staff_photo(staff_name: str, filename: str):
    """Delete a single reference photo from a staff member's profile."""
    staff_name = secure_filename(staff_name)
    filename   = secure_filename(filename)
    staff_dir  = os.path.join(_upload_folder(), staff_name)
    file_path  = os.path.join(staff_dir, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        if not os.listdir(staff_dir):   # remove empty folder
            os.rmdir(staff_dir)
        return jsonify({"success": True})

    return jsonify({"error": "File not found"}), 404

@api_bp.route("/api/system_settings", methods=["GET"])
@login_required
def api_get_system_settings():
    """Retrieve all system settings (branding, appearance, etc)."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM system_settings")
        rows = cur.fetchall()
        settings = {r['key']: r['value'] for r in rows}
        cur.close()
        conn.close()
        return jsonify(settings)
    except Exception as e:
        log.error("Error fetching system settings: %s", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/api/system_settings", methods=["POST"])
@login_required
def api_update_system_settings():
    """Update system settings (branding, appearance, etc)."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        conn = get_db_connection()
        cur = conn.cursor()
        for key, value in data.items():
            cur.execute(
                "INSERT INTO system_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, str(value))
            )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        log.error("Error updating system settings: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/upload_branding", methods=["POST"])
@login_required
def api_upload_branding():
    """Upload branding assets (Logo or Favicon)."""
    try:
        if 'file' not in request.files or 'type' not in request.form:
             return jsonify({"success": False, "error": "Missing file or type"}), 400
        
        file = request.files['file']
        setting_type = request.form['type'] # 'logo' or 'favicon'
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400
            
        filename = secure_filename(file.filename)
        branding_dir = os.path.join(_upload_folder(), "branding")
        os.makedirs(branding_dir, exist_ok=True)
        
        # Create a unique filename to prevent browser caching issues
        import time
        ext = os.path.splitext(filename)[1]
        unique_filename = f"{setting_type}_{int(time.time())}{ext}"
        save_path = os.path.join(branding_dir, unique_filename)
        file.save(save_path)
        
        url = url_for('static', filename=f'uploads/branding/{unique_filename}')
        
        # Persistence: Update database
        key = 'logo_url' if setting_type == 'logo' else 'favicon_url'
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO system_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, url)
        )
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({"success": True, "url": url})
    except Exception as e:
        log.error("Error uploading branding: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

def download_image_from_url(url: str, dest_folder: str) -> str | None:
    """Downloads an image from a URL and saves it to a folder."""
    import requests
    import time
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            ext = os.path.splitext(url)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']: ext = '.jpg'
            filename = f"imported_{int(time.time())}{ext}"
            path = os.path.join(dest_folder, filename)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            log.info("Downloaded image %s to %s", url, path)
            return filename
    except Exception as e:
        log.error("Failed to download image %s: %s", url, e)
    return None

@api_bp.route("/api/staff/import", methods=["POST"])
@login_required
def api_staff_import():
    """Bulk import staff members from CSV."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        import csv
        import io
        
        # Read file as text
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)
        
        conn = get_db_connection()
        cur = conn.cursor()
        count = 0
        
        for row in csv_input:
            # Handle potential BOM or different case
            name = (row.get('name') or row.get('Name') or '').strip()
            if not name: continue
            
            email = (row.get('email') or row.get('Email') or '').strip()
            phone = (row.get('phone') or row.get('Phone') or '').strip()
            address = (row.get('address') or row.get('Address') or '').strip()
            image_url = (row.get('image_url') or row.get('Image URL') or '').strip()
            comm_details = (row.get('communication_details') or row.get('Communication') or '').strip()
            
            # Clean up communication details - if it looks like a list/dict string, try to normalize it
            if comm_details.startswith(('[', '{')):
                try:
                    import json
                    # Just verify it's valid JSON
                    json.loads(comm_details.replace("'", '"'))
                except:
                    log.warning("Invalid JSON in communication_details for %s", name)
            
            # Ensure a directory exists for the staff member
            safe_name = secure_filename(name)
            staff_dir = os.path.join(_upload_folder(), safe_name)
            os.makedirs(staff_dir, exist_ok=True)

            # Handle image download if provided
            if image_url:
                download_image_from_url(image_url, staff_dir)

            # Check if name exists to update instead of insert
            cur.execute("SELECT id FROM staff_profiles WHERE name = %s", (name,))
            if cur.fetchone():
                cur.execute(
                    "UPDATE staff_profiles SET email=%s, phone=%s, address=%s, communication=%s WHERE name=%s",
                    (email, phone, address, comm_details, name)
                )
            else:
                cur.execute(
                    "INSERT INTO staff_profiles (name, email, phone, address, communication, folder_path) VALUES (%s, %s, %s, %s, %s, %s)",
                    (name, email, phone, address, comm_details, staff_dir)
                )
            count += 1
            
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({"success": True, "count": count})
    except Exception as e:
        log.error("Error during staff import: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/staff/export", methods=["GET"])
@login_required
def api_staff_export():
    """Export staff directory to CSV or printable format."""
    fmt = request.args.get('format', 'csv')
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, email, phone, address, status, communication FROM staff_profiles ORDER BY name ASC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if fmt in ['csv', 'excel'] or request.args.get('ext') == 'xlsx':
            if fmt == 'excel' or request.args.get('ext') == 'xlsx':
                import pandas as pd
                import io
                from flask import Response
                df = pd.DataFrame(rows)
                # Rename columns for clarity in Excel
                df.columns = ['Name', 'Email', 'Phone', 'Address', 'Status', 'Communication Details']
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Members')
                
                return Response(
                    output.getvalue(),
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-disposition": f"attachment; filename=member_directory.xlsx"}
                )
            else:
                import csv
                import io
                from flask import Response
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(['name', 'email', 'phone', 'address', 'status', 'communication_details'])
                for r in rows:
                    writer.writerow([r['name'], r['email'], r['phone'], r['address'], r['status'], r['communication']])
                
                return Response(
                    output.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-disposition": f"attachment; filename=member_directory.csv"}
                )
        else:
            # Simple printable HTML for PDF export
            html = f"""
            <html>
            <head>
                <title>Member Directory Export</title>
                <style>
                    body {{ font-family: sans-serif; padding: 40px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f4f4f4; }}
                    h1 {{ color: #333; }}
                </style>
            </head>
            <body>
                <h1>Member Directory</h1>
                <p>Exported on: {os.popen('date /t').read().strip() if os.name == 'nt' else os.popen('date').read().strip()}</p>
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Email</th>
                            <th>Phone</th>
                            <th>Address</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for r in rows:
                html += f"""
                        <tr>
                            <td>{r['name']}</td>
                            <td>{r['email']}</td>
                            <td>{r['phone']}</td>
                            <td>{r['address']}</td>
                            <td>{r['status']}</td>
                        </tr>
                """
            html += """
                    </tbody>
                </table>
                <script>window.print();</script>
            </body>
            </html>
            """
            return html
    except Exception as e:
        log.error("Error during staff export: %s", e)
        return f"Export Error: {str(e)}", 500


@api_bp.route("/api/attendance/records", methods=["GET"])
@login_required
def api_attendance_records():
    """Return attendance records from compiled attendance table."""
    q = (request.args.get("q") or "").strip()
    date_from = (request.args.get("date_from") or "").strip()
    date_to = (request.args.get("date_to") or "").strip()
    period = (request.args.get("period") or "all").strip().lower()
    if period not in {"all", "today"}:
        period = "all"
    try:
        page = int(request.args.get("page", 1))
    except Exception:
        page = 1
    try:
        page_size = int(request.args.get("page_size", 10))
    except Exception:
        page_size = 10
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    offset = (page - 1) * page_size

    where = []
    params = []

    if period == "today":
        where.append("a.attendance_date = CURRENT_DATE")
    else:
        if date_from:
            where.append("a.attendance_date >= %s")
            params.append(date_from)
        if date_to:
            where.append("a.attendance_date <= %s")
            params.append(date_to)
    if q:
        like = f"%{q}%"
        where.append(
            "(COALESCE(s.name,'') ILIKE %s OR COALESCE(s.email,'') ILIKE %s OR COALESCE(s.phone,'') ILIKE %s OR COALESCE(s.staff_id,'') ILIKE %s)"
        )
        params.extend([like, like, like, like])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Count — from attendance table
        cur.execute(
            f"""
            SELECT COUNT(*)::int AS total_records
            FROM attendance a
            LEFT JOIN staff_profiles s ON s.id = a.staff_id
            {where_sql}
            """,
            tuple(params),
        )
        count_row = cur.fetchone() or {}
        total_records = count_row.get("total_records", 0) or 0

        # Records — aggregate session records from attendance table
        cur.execute(
            f"""
            SELECT
                a.id,
                a.staff_id as db_id,
                s.staff_id as display_id,
                COALESCE(s.name,  '-') AS name,
                COALESCE(s.email, '')  AS email,
                COALESCE(s.phone, '')  AS phone,
                a.status,
                a.first_entry_time     AS in_time,
                a.last_exit_time       AS out_time,
                a.camera_name,
                a.in_image,
                a.out_image,
                a.timestamp
            FROM attendance a
            LEFT JOIN staff_profiles s ON s.id = a.staff_id
            {where_sql}
            ORDER BY a.attendance_date DESC, a.first_entry_time DESC
            LIMIT %s OFFSET %s
            """,
            tuple(params + [page_size, offset]),
        )
        rows = cur.fetchall()

        # Summary — aggregate from attendance table
        cur.execute(
            f"""
            SELECT
                COUNT(*)::int AS total_records,
                COUNT(DISTINCT a.staff_id)::int AS total_present,
                SUM(CASE WHEN a.attendance_date = CURRENT_DATE THEN 1 ELSE 0 END)::int AS today_total
            FROM attendance a
            LEFT JOIN staff_profiles s ON s.id = a.staff_id
            {where_sql}
            """,
            tuple(params),
        )
        summary = cur.fetchone() or {}

        formatted_records = []
        for r in rows:
            formatted_records.append({
                "id": r["id"],
                "staff_id": r["db_id"],
                "display_id": r["display_id"],
                "name": r["name"],
                "email": r["email"],
                "phone": r["phone"],
                "status": r["status"],
                "in_time": r["in_time"].strftime("%Y-%m-%d %H:%M:%S") if r["in_time"] else "-",
                "out_time": r["out_time"].strftime("%Y-%m-%d %H:%M:%S") if r["out_time"] else "-",
                "camera_name": r.get("camera_name", "-"),
                "in_image": r.get("in_image"),
                "out_image": r.get("out_image"),
                "timestamp": r["timestamp"].strftime("%Y-%m-%d") if hasattr(r["timestamp"], "strftime") else str(r["timestamp"])
            })

        return jsonify(
            {
                "success": True,
                "records": formatted_records,
                "summary": {
                    "total_records": summary.get("total_records", 0) or 0,
                    "total_staff": summary.get("total_present", 0) or 0,
                    "today_total": summary.get("today_total", 0) or 0,
                },
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_records": total_records,
                    "total_pages": (total_records + page_size - 1) // page_size if total_records else 0,
                },
                "period": period,
            }
        )
    except Exception as e:
        log.error("Error loading attendance records: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if conn:
            conn.close()


@api_bp.route("/api/attendance/instant_mark", methods=["POST"])
@login_required
def api_attendance_instant_mark():
    """Capture a frame from the webcam and mark attendance if a staff member is recognized."""
    try:
        # 1. Get current frame from cam_mgr
        jpeg_bytes = cam_mgr.get_jpeg()
        if not jpeg_bytes:
            return jsonify({"success": False, "error": "No camera stream active"}), 400

        # 2. Decode JPEG to image array
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode camera frame"}), 500

        # 3. Get FaceRecognizer from AI Pipeline
        face_rec = None
        if hasattr(cam_mgr, "_pipeline") and cam_mgr._pipeline:
            face_rec = cam_mgr._pipeline.face_rec
        
        if not face_rec:
             return jsonify({"success": False, "error": "AI Pipeline or Face Recognition not initialized"}), 500

        if not hasattr(face_rec, "_available") or not face_rec._available:
             return jsonify({"success": False, "error": "Face Recognition engine (DeepFace) is missing or disabled"}), 503

        # 4. Run recognition on the full frame
        # (We use a dummy track_id -999 for manual capture)
        res = face_rec.recognize(frame, track_id=-999)
        
        name = res.get("name", "Unknown")
        staff_id = res.get("id")

        if name == "Unknown" or not staff_id:
            return jsonify({"success": False, "error": "No recognized staff member found in frame"}), 404

        # 5. Save the image first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_mark_{name.replace(' ', '_')}_{timestamp}.jpg"
        static_uploads = os.path.join("static", "uploads", "movement")
        os.makedirs(static_uploads, exist_ok=True)
        cv2.imwrite(os.path.join(static_uploads, filename), frame)
        img_url = f"static/uploads/movement/{filename}"

        # 6. Flow Step 1: Log to movement_log
        from app.services.attendance_service import log_movement, log_person, track_staff_attendance
        camera_name = getattr(getattr(cam_mgr, "_pipeline", None), "camera_name", "Camera")
        log_movement(camera_name, img_url)

        # 7. Flow Step 2: Log to member_timestamp for forensics
        log_person(camera_name, "staff", staff_id, img_url, 1.0, staff_name=name)

        # 8. Flow Step 3: Mark official Attendance
        track_staff_attendance(staff_id, staff_name=name, entry_image=img_url, camera_name=camera_name)

        log.info("MANUAL ATTENDANCE CHAIN COMPLETE: %s", name)
        
        return jsonify({
            "success": True, 
            "message": f"Successfully marked attendance for {name}",
            "name": name,
            "staff_id": staff_id
        })

    except Exception as e:
        log.error("Error in instant_mark: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/attendance/forensics", methods=["GET"])
@login_required
def api_attendance_forensics():
    """Return member_time_stamp rows with images for a given staff member and date."""
    staff_name = (request.args.get("staff_name") or "").strip()
    date_str   = (request.args.get("date") or "").strip()
    limit      = min(int(request.args.get("limit", 20)), 50)

    where = []
    params = []
    if staff_name:
        where.append("s.name ILIKE %s")
        params.append(f"%{staff_name}%")
    if date_str:
        where.append("m.detected_at::date = %s::date")
        params.append(date_str)
    else:
        where.append("m.detected_at::date = CURRENT_DATE")

    where_sql = "WHERE " + " AND ".join(where) if where else "WHERE m.entry_time::date = CURRENT_DATE"

    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(f"""
            SELECT
                m.id,
                m.staff_id,
                COALESCE(s.name, 'Unknown') as staff_name,
                m.camera_id as camera_name,
                m.entry_image,
                m.entry_time,
                m.exit_image,
                m.exit_time,
                m.merged_image,
                COALESCE(s.email, '') AS email,
                COALESCE(s.phone, '') AS phone
            FROM member_timestamp m
            LEFT JOIN staff_profiles s ON s.id = m.staff_id
            {where_sql}
            ORDER BY m.detected_at DESC
            LIMIT %s
        """, tuple(params + [limit]))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        def img_url(filename):
            if not filename:
                return None
            return f"/{filename}" if str(filename).startswith("static/") else f"/static/uploads/snapshots/{filename}"

        records = []
        for r in rows:
            records.append({
                "id":          r["id"],
                "staff_name":  r["staff_name"] or "Unknown",
                "email":       r["email"],
                "phone":       r["phone"],
                "camera_name": r["camera_name"],
                "entry_time":  r["entry_time"].strftime("%Y-%m-%d %H:%M:%S") if r["entry_time"] else None,
                "exit_time":   r["exit_time"].strftime("%Y-%m-%d %H:%M:%S")  if r["exit_time"]  else None,
                "entry_image": img_url(r["entry_image"]),
                "exit_image":  img_url(r["exit_image"]),
                "merged_image":img_url(r["merged_image"]),
            })
        return jsonify({"success": True, "records": records})
    except Exception as e:
        log.error("Error fetching forensics: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/attendance/export", methods=["GET"])
@login_required
def api_attendance_export():
    """Export filtered attendance records to CSV or Excel."""
    fmt = (request.args.get("format") or "excel").strip().lower()
    status = (request.args.get("status") or "").strip().upper()
    q = (request.args.get("q") or "").strip()
    date_from = (request.args.get("date_from") or "").strip()
    date_to = (request.args.get("date_to") or "").strip()
    period = (request.args.get("period") or "all").strip().lower()
    if period not in {"all", "today"}:
        period = "all"

    where = []
    params = []
    if status in {"IN", "OUT"}:
        where.append("m.event_type = %s")
        params.append(status)
    if period == "today":
        where.append("a.attendance_date = CURRENT_DATE")
    else:
        if date_from:
            where.append("m.attendance_date >= %s")
            params.append(date_from)
        if date_to:
            where.append("m.attendance_date <= %s")
            params.append(date_to)
    if q:
        like = f"%{q}%"
        where.append("(COALESCE(s.name,'') ILIKE %s OR COALESCE(s.email,'') ILIKE %s OR COALESCE(s.phone,'') ILIKE %s)")
        params.extend([like, like, like])

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                a.attendance_date,
                s.staff_id as display_id,
                COALESCE(s.name,  '-') AS name,
                COALESCE(s.email, '')  AS email,
                COALESCE(s.phone, '')  AS phone,
                a.status,
                a.first_entry_time     AS in_time,
                a.last_exit_time       AS out_time,
                a.camera_name
            FROM attendance a
            LEFT JOIN staff_profiles s ON s.id = a.staff_id
            {where_sql}
            ORDER BY a.attendance_date DESC, a.first_entry_time DESC
            """,
            tuple(params),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if fmt == "csv":
            import csv
            import io
            from flask import Response

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Date", "Staff ID", "Member Name", "Email", "Phone", "Status", "Camera", "First Entry", "Last Exit"])
            for r in rows:
                writer.writerow([
                    r["attendance_date"], 
                    r["display_id"], 
                    r["name"], 
                    r["email"], 
                    r["phone"], 
                    r["status"], 
                    r["camera_name"],
                    r["in_time"].strftime("%H:%M:%S") if r["in_time"] else "-", 
                    r["out_time"].strftime("%H:%M:%S") if r["out_time"] else "-"
                ])
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-disposition": "attachment; filename=attendance_records.csv"},
            )

        import io
        import pandas as pd
        from flask import Response

        df = pd.DataFrame(rows)
        if not df.empty:
            df.columns = ["Date", "Staff ID", "Name", "Email", "Phone", "Status", "First Entry", "Last Exit", "Camera"]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Attendance")

        return Response(
            output.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-disposition": "attachment; filename=attendance_records.xlsx"},
        )
    except Exception as e:
        log.error("Error exporting attendance records: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

