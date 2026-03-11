"""
app/db/session.py  –  PostgreSQL Database Connection Layer
──────────────────────────────────────────────────────────────────────────────
Connects to the PostgreSQL Database and initializes tables safely.
"""

from __future__ import annotations

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

log = logging.getLogger("database")

def get_db_connection():
    """Returns a new psycopg2 connection. Caller must close it."""
    # Defaults to local postgres if DATABASE_URL is not set in .env
    db_url = os.environ.get(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/cctv_logs"
    )
    return psycopg2.connect(db_url, cursor_factory=RealDictCursor)

def init_db() -> None:
    """Create all required tables in PostgreSQL if they do not exist."""
    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email         TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                company       TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS local_cameras (
                id           SERIAL PRIMARY KEY,
                name         TEXT,
                brand        TEXT,
                ip_address   TEXT,
                port         INTEGER,
                username     TEXT,
                password     TEXT,
                stream_path  TEXT,
                owner_email  TEXT
            )
        """)

        conn.commit()
        cur.close()
        log.info("PostgreSQL Database initialized successfully.")
    except Exception as e:
        log.error("Failed to connect to PostgreSQL. Is the service running? Error: %s", e)
    finally:
        if conn:
            conn.close()
