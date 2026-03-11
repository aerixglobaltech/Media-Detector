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
    from app.services.camera_manager import _total_motion_events

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

    if "photos" not in request.files:
        return jsonify({"success": False, "error": "No files attached"})

    files = request.files.getlist("photos")
    if not files or (len(files) == 1 and files[0].filename == ""):
        return jsonify({"success": False, "error": "No valid files selected"})

    staff_dir = os.path.join(_upload_folder(), staff_name)
    os.makedirs(staff_dir, exist_ok=True)

    saved_count = 0
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(staff_dir, filename))
            saved_count += 1

    return jsonify({"success": True, "saved_count": saved_count})


@api_bp.route("/api/staff_profiles", methods=["GET"])
@login_required
def api_staff_profiles():
    """List all staff profiles with a thumbnail URL and photo count."""
    profiles     = []
    upload_folder = _upload_folder()
    if os.path.exists(upload_folder):
        for staff_name in os.listdir(upload_folder):
            staff_dir = os.path.join(upload_folder, staff_name)
            if os.path.isdir(staff_dir):
                photos = [
                    f for f in os.listdir(staff_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                if photos:
                    thumbnail_url = url_for(
                        "static", filename=f"uploads/{staff_name}/{photos[0]}"
                    )
                    profiles.append({
                        "name":        staff_name,
                        "photo_count": len(photos),
                        "thumbnail":   thumbnail_url,
                    })
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
            return jsonify({"success": True})
        return jsonify({"error": "Staff not found"}), 404

    # GET
    if not os.path.isdir(staff_dir):
        return jsonify({"error": "Staff not found"}), 404

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
    return jsonify({"staff": staff_name, "photos": urls})


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
