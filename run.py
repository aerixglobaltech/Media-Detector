"""
run.py  –  MISSION CONTROL  –  Application Entry Point
──────────────────────────────────────────────────────────────────────────────
Starts the MISSION CONTROL AI CCTV Surveillance System web server.

Usage
─────
    python run.py            (development – uses Waitress WSGI server)
    python run.py --dev      (Flask dev server with debug/reload)

Environment variables (set in .env)
─────────────────────────────────────
    PORT            – TCP port to listen on (default: 5000)
    DETECTION_MODE  – yolo_only | action_only | full (default: action_only)
    IMOU_APP_ID     – Imou cloud camera App ID
    IMOU_APP_SECRET – Imou cloud camera App Secret
    TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID / ENABLE_TELEGRAM – alerting
"""

from __future__ import annotations

import logging
import os
import sys

# ── Load .env BEFORE importing anything else ─────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run")

# ── Create Flask app ──────────────────────────────────────────────────────────
from app import create_app   # noqa: E402

flask_app = create_app()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    dev_mode = "--dev" in sys.argv

    if dev_mode:
        log.info("Starting in DEVELOPMENT mode on http://localhost:%d", port)
        flask_app.run(host="0.0.0.0", port=port, debug=True, use_reloader=True)
    else:
        try:
            from waitress import serve
            log.info("Starting MISSION CONTROL on http://localhost:%d  (Waitress)", port)
            # threads=32 prevents video_feed from exhausting worker threads
            serve(flask_app, host="0.0.0.0", port=port, threads=32)
        except ImportError:
            log.warning("waitress not installed — falling back to Flask dev server")
            flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
