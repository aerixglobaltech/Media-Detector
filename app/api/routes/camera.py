"""
app/routes/camera.py  –  Camera Management Routes Blueprint
──────────────────────────────────────────────────────────────────────────────
Handles camera streaming, selection, and MJPEG video feed.

Routes:
  GET  /api/cameras            → JSON list of all cameras
  POST /api/select             → Start streaming a selected camera
  POST /api/stop               → Stop current camera stream
  GET  /api/stream_id          → Returns current stream ID for frontend polling
  GET  /video_feed             → MJPEG live stream endpoint
  GET  /api/local_cameras      → List user's local IP cameras
  POST /api/local_cameras      → Add a new local IP camera
  DELETE /api/local_cameras/<id> → Remove a local camera
"""

from __future__ import annotations

import logging
import os
import time

from flask import Blueprint, Response, jsonify, request, session

from app.core.security import login_required
from app.db.session import get_db_connection
from app.core.extensions import cam_mgr

log = logging.getLogger("camera")

camera_bp = Blueprint("camera", __name__)

# ── Imou credentials (loaded from environment) ────────────────────────────────
APP_ID     = os.environ.get("IMOU_APP_ID", "")
APP_SECRET = os.environ.get("IMOU_APP_SECRET", "")


# ── Camera cache ─────────────────────────────────────────────────────────────
_camera_cache: list[dict] = []
_camera_cache_ts: float   = 0.0
_camera_lock = __import__("threading").Lock()


def _fetch_imou_cameras() -> list[dict]:
    """Return list of Imou cameras with {id, name, status, type, dev_id, base}."""
    try:
        from app.services.ai.imou_connector import ImouAPI, _find_working_datacenter, _get_device_status
        base = _find_working_datacenter()
        if not base:
            return []
        api = ImouAPI(APP_ID, APP_SECRET, base)
        api.get_token()
        devices = api.list_devices()
        result = []
        for i, dev in enumerate(devices):
            dev_id = dev.get("deviceId") or dev.get("deviceID") or f"dev{i}"
            name   = dev.get("name") or dev.get("deviceName") or "Imou Camera"
            status = _get_device_status(dev)
            result.append({
                "id":     f"imou_{dev_id}",
                "name":   name,
                "status": status,
                "type":   "imou",
                "dev_id": dev_id,
                "base":   base,
            })
        return result
    except Exception as e:
        log.warning("Imou fetch failed: %s", e)
        return []


def get_all_cameras(force: bool = False, user_email: str | None = None) -> list[dict]:
    """
    Return all available cameras: webcam + Imou cloud + local IP cameras.
    Imou cameras are cached for 60 seconds to avoid re-authenticating on each request.
    """
    global _camera_cache, _camera_cache_ts

    with _camera_lock:
        if force or (time.time() - _camera_cache_ts) >= 60 or not _camera_cache:
            base_cams = [{"id": "webcam", "name": "Webcam (Built-in)", "status": "🟢 Online", "type": "webcam"}]
            base_cams.extend(_fetch_imou_cameras())
            _camera_cache    = base_cams
            _camera_cache_ts = time.time()
        cameras = list(_camera_cache)

    # Append user-specific local cameras from database
    if user_email:
        try:
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute(
                "SELECT id, name, brand, ip_address FROM local_cameras WHERE owner_email = %s",
                (user_email,),
            )
            for row in cur.fetchall():
                cameras.append({
                    "id":     f"local_{row['id']}",
                    "name":   row["name"],
                    "status": "🟢 Online",
                    "type":   "local",
                    "brand":  row["brand"],
                    "ip":     row["ip_address"],
                })
            cur.close()
            conn.close()
        except Exception as e:
            log.error("Error fetching local cameras: %s", e)

    return cameras


# ─── Routes ──────────────────────────────────────────────────────────────────

@camera_bp.route("/api/cameras")
def api_cameras():
    user_email = session.get("user")
    cameras = get_all_cameras(user_email=user_email)
    return jsonify(cameras)


@camera_bp.route("/api/select", methods=["POST"])
def api_select():
    data   = request.get_json(force=True)
    cam_id = data.get("id", "webcam")
    user_email = session.get("user")
    cameras = get_all_cameras(user_email=user_email)
    cam = next((c for c in cameras if c["id"] == cam_id), None)

    if cam is None:
        return jsonify({"error": "Camera not found"}), 404

    if cam["type"] == "webcam":
        cam_mgr.start(0, cam["name"])

    elif cam["type"] == "imou":
        try:
            from app.services.ai.imou_connector import ImouAPI
            base       = cam["base"]
            api        = ImouAPI(APP_ID, APP_SECRET, base)
            api.get_token()
            stream_url = api.get_rtsp(cam["dev_id"])
            if not stream_url:
                return jsonify({"error": "Could not get stream URL from Imou"}), 500
            cam_mgr.start(stream_url, cam["name"])
        except Exception as e:
            log.error("Imou start error: %s", e)
            return jsonify({"error": str(e)}), 500

    elif cam["type"] == "local":
        try:
            db_id = cam_id.replace("local_", "")
            conn  = get_db_connection()
            cur   = conn.cursor()
            cur.execute(
                "SELECT ip_address, port, username, password, stream_path FROM local_cameras WHERE id = %s",
                (db_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return jsonify({"error": "Local camera record not found"}), 404

            ip, port, user, pw, path = (
                row["ip_address"], row["port"], row["username"],
                row["password"], row["stream_path"],
            )
            if user and pw:
                rtsp_url = f"rtsp://{user}:{pw}@{ip}:{port}{path}"
            else:
                rtsp_url = f"rtsp://{ip}:{port}{path}"

            log.info("Starting local RTSP: %s", rtsp_url)
            cam_mgr.start(rtsp_url, cam["name"])
        except Exception as e:
            log.error("Local camera start error: %s", e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "camera": cam["name"]})


@camera_bp.route("/api/stop", methods=["POST"])
def api_stop():
    cam_mgr.stop_stream()
    return jsonify({"success": True})


@camera_bp.route("/api/stream_id")
def api_stream_id():
    """Returns current stream ID so the frontend can detect camera switches."""
    return jsonify({"stream_id": cam_mgr.stream_id, "active": cam_mgr._active})


@camera_bp.route("/video_feed")
def video_feed():
    """MJPEG live stream endpoint — one frame per ~33ms."""
    def generate():
        while True:
            frame = cam_mgr.get_jpeg()
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── Local camera CRUD ────────────────────────────────────────────────────────

@camera_bp.route("/api/local_cameras", methods=["GET", "POST"])
@login_required
def api_local_cameras():
    email = session.get("user")

    if request.method == "POST":
        data = request.get_json(force=True)
        name = data.get("name")
        brand = data.get("brand", "Generic")
        ip    = data.get("ip")
        port  = data.get("port", 554)
        user  = data.get("username", "")
        pw    = data.get("password", "")
        path  = data.get("path", "")

        if not name or not ip:
            return jsonify({"success": False, "error": "Name and IP are required"}), 400

        try:
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute(
                """INSERT INTO local_cameras
                   (name, brand, ip_address, port, username, password, stream_path, owner_email)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (name, brand, ip, port, user, pw, path, email),
            )
            conn.commit()
            cur.close()
            conn.close()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # GET – list cameras for this user
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, name, brand, ip_address, port, username, password, stream_path "
            "FROM local_cameras WHERE owner_email = %s",
            (email,),
        )
        cameras = [
            {
                "id": r["id"], "name": r["name"], "brand": r["brand"],
                "ip": r["ip_address"], "port": r["port"],
                "username": r["username"], "password": r["password"],
                "path": r["stream_path"],
            }
            for r in cur.fetchall()
        ]
        cur.close()
        conn.close()
        return jsonify(cameras)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@camera_bp.route("/api/local_cameras/<int:cam_id>", methods=["DELETE"])
@login_required
def api_delete_local_camera(cam_id: int):
    email = session.get("user")
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            "DELETE FROM local_cameras WHERE id = %s AND owner_email = %s",
            (cam_id, email),
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
