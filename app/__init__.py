"""
app/__init__.py  –  Flask Application Factory
──────────────────────────────────────────────────────────────────────────────
Creates and configures the Flask app instance.
Uses the Application Factory pattern so the app can be created on demand
(useful for testing and different deployment configurations).
"""

from __future__ import annotations

import atexit
import logging
import os

from flask import Flask

from app.db.session import init_db
from app.core.extensions import cam_mgr, notifier

log = logging.getLogger("app")


def create_app() -> Flask:
    """
    Application factory.

    Returns a fully configured Flask WSGI application.
    Blueprints are registered here so each module stays self-contained.
    """
    flask_app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
    )

    @flask_app.context_processor
    def inject_permissions():
        from flask import session
        try:
            from app.db.session import get_db_connection
            user_email = session.get("user")
            if not user_email: return dict(user_permissions={})
            
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT r.permissions FROM users u LEFT JOIN roles r ON u.role_id = r.id WHERE u.email = %s", (user_email,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            perms = {}
            if row and row.get('permissions'):
                perms = row['permissions']
                if isinstance(perms, str):
                    import json
                    perms = json.loads(perms)
            return dict(user_permissions=perms)
        except Exception:
            return dict(user_permissions={})
    flask_app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_mission_control_key_xyz")

    # ── Upload folder ────────────────────────────────────────────────────────
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    flask_app.config["UPLOAD_FOLDER"] = upload_folder

    # ── Initialize database ──────────────────────────────────────────────────
    init_db()

    # ── Register blueprints ──────────────────────────────────────────────────
    from app.api.routes.auth import auth_bp
    from app.api.routes.camera import camera_bp
    from app.api.routes.detection import api_bp as detection_bp
    from app.api.routes.dashboard import dashboard_bp
    from app.api.routes.user_mgmt import user_mgmt_bp

    flask_app.register_blueprint(auth_bp)
    flask_app.register_blueprint(dashboard_bp)
    flask_app.register_blueprint(camera_bp)
    flask_app.register_blueprint(detection_bp)
    flask_app.register_blueprint(user_mgmt_bp)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    atexit.register(cam_mgr.stop_all)

    log.info("Flask application created successfully.")
    return flask_app
