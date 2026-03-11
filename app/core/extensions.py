"""
app/extensions.py  –  Shared Application-Level Singletons
──────────────────────────────────────────────────────────────────────────────
Creates global objects that are shared across the application:
  • cam_mgr  – CameraManager (holds the AI pipeline + render thread)
  • notifier – TelegramNotifier (for Telegram alerts)
  • AI_TOGGLES – runtime on/off switches for AI features

These are imported from blueprints rather than created inside the Flask factory
so they survive across requests without being re-created on app context teardown.
"""

from __future__ import annotations

import logging

log = logging.getLogger("extensions")

# ── AI Feature Toggles (runtime flags, mutated via /api/toggles) ─────────────
AI_TOGGLES: dict[str, bool] = {
    "person":  True,
    "action":  True,
    "emotion": True,
}

# ── Lazy-initialized singletons ──────────────────────────────────────────────
# These are initialized once at startup (not inside request context).

def _make_camera_manager():
    from app.services.camera_service import CameraManager
    log.info("Initializing Camera Manager and AI Pipeline …")
    mgr = CameraManager()
    log.info("Camera Manager ready.")
    return mgr


def _make_notifier():
    from app.services.ai.notifier import TelegramNotifier
    return TelegramNotifier()


# Build singletons at import time (module-level, happens once at startup)
cam_mgr  = _make_camera_manager()
notifier = _make_notifier()
