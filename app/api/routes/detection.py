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
    """Stats card data: motion events, unique people, camera online/offline counts."""
    from app.services.camera_service import _total_motion_events

    unique_people = 0
    try:
        if cam_mgr._pipeline and hasattr(cam_mgr._pipeline, "tracker"):
            ds = cam_mgr._pipeline.tracker.tracker
            unique_people = max(0, ds.tracker._next_id - 1)
    except Exception:
        pass

    user_email = session.get("user")
    cams = get_all_cameras(user_email=user_email)
    online_count  = sum(1 for c in cams if "Online"  in c.get("status", ""))
    offline_count = sum(1 for c in cams if "Offline" in c.get("status", ""))

    return jsonify({
        "total_motion_events": _total_motion_events,
        "unique_people":       unique_people,
        "online_cameras":      online_count,
        "offline_cameras":     offline_count,
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

@api_bp.route("/api/settings_info")
def api_settings_info():
    """Return current .env values shown as read-only on the settings page."""
    return jsonify({
        "app_id":   os.environ.get("IMOU_APP_ID",         ""),
        "tg_token": os.environ.get("TELEGRAM_BOT_TOKEN",  ""),
        "tg_chats": os.environ.get("TELEGRAM_CHAT_ID",    ""),
    })


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
    cur.execute("SELECT email, name, company FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if user:
        return jsonify({"email": user["email"], "name": user["name"], "company": user["company"]})
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
