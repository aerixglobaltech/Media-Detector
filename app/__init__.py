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

    # ── Secret key ──────────────────────────────────────────────────────────
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

    flask_app.register_blueprint(auth_bp)
    flask_app.register_blueprint(dashboard_bp)
    flask_app.register_blueprint(camera_bp)
    flask_app.register_blueprint(detection_bp)

    # ── Graceful shutdown ────────────────────────────────────────────────────
    atexit.register(cam_mgr.stop_all)

    log.info("Flask application created successfully.")
    return flask_app
