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
        return jsonify({"success": False, "error": "Staff name is required"})

    staff_name = secure_filename(request.form["staff_name"])
    if not staff_name:
        return jsonify({"success": False, "error": "Invalid staff name"})

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
    address = request.form.get("address", "")
    communication = request.form.get("communication", "") # JSON string from frontend

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
                SET name = %s, email = %s, phone = %s, address = %s, communication = %s, folder_path = %s 
                WHERE name = %s
            """, (staff_name, email, phone, address, communication, staff_dir, original_name))
            
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
                    SET email = %s, phone = %s, address = %s, communication = %s, folder_path = %s 
                    WHERE id = %s
                """, (email, phone, address, communication, staff_dir, row["id"]))
            else:
                cur.execute("""
                    INSERT INTO staff_profiles (name, email, phone, address, communication, folder_path)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (staff_name, email, phone, address, communication, staff_dir))
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
            staff_dir = os.path.join(upload_folder, staff_name)
            photo_count = 0
            thumbnail_url = ""
            if os.path.exists(staff_dir) and os.path.isdir(staff_dir):
                photos = [f for f in os.listdir(staff_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                photo_count = len(photos)
                if photos:
                    thumbnail_url = url_for("static", filename=f"uploads/{staff_name}/{photos[0]}")
            profiles.append({
                "id":          row["id"],
                "name":        staff_name,
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
    staff_name = secure_filename(staff_name)
    staff_dir  = os.path.join(_upload_folder(), staff_name)

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
        return jsonify({"error": "Staff not found"}), 404
        
    staff_data = {"name": staff_name, "email": "", "phone": "", "address": "", "communication": ""}
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT email, phone, address, communication FROM staff_profiles WHERE name = %s", (staff_name,))
        row = cur.fetchone()
        if row:
            staff_data["email"] = row["email"] or ""
            staff_data["phone"] = row["phone"] or ""
            staff_data["address"] = row["address"] or ""
            staff_data["communication"] = row["communication"] or "" 
            log.info("Loaded staff data with communication: %s", staff_data["communication"])
    except Exception as e:
        log.error("DB Error fetching single staff: %s", e)
    finally:
        if conn:
            conn.close()

    photos = [
        f for f in os.listdir(staff_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    urls = [
        {
            "filename": p,
            "url": url_for("static", filename=f"uploads/{staff_name}/{p}"),
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
