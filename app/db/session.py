"""
app/db/session.py  -  Strict PostgreSQL Connection Layer
------------------------------------------------------------------------------
Description: Strictly requires PostgreSQL for all operations.
SQLite support has been removed to ensure data consistency.
"""

from __future__ import annotations

import os
import logging
from typing import Any

# PostgreSQL driver is now mandatory
try:
    import psycopg2
    import psycopg2.extras
    HAVE_POSTGRES = True
except ImportError:
    HAVE_POSTGRES = False

log = logging.getLogger("database")

class PostgresConnectionWrapper:
    """
    A wrapper for psycopg2 connection to provide a consistent interface.
    No longer needs placeholder translation as all endpoints now use %s.
    """
    def __init__(self, conn: Any):
        self.conn = conn

    def cursor(self, *args, **kwargs):
        # RealDictCursor is required for key-based access in routes
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return cur

    def commit(self):
        return self.conn.commit()

    def rollback(self):
        return self.conn.rollback()

    def close(self):
        return self.conn.close()

def get_db_url() -> str:
    """Read DATABASE_URL from .env using absolute path."""
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, ".env")
    
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("CRITICAL: DATABASE_URL not found in .env file.")
    return url

def get_db_connection():
    """Returns a PostgreSQL connection. Throws error if unavailable."""
    url = get_db_url()

    if not HAVE_POSTGRES:
        print("CRITICAL: psycopg2-binary is MISSING!")
        raise ImportError("psycopg2-binary is required for PostgreSQL support.")

    # Standardize URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    try:
        conn = psycopg2.connect(url)
        # Log connection success to terminal
        db_name = url.split('/')[-1].split('?')[0]
        print(f">>> DATABASE: Connected to PostgreSQL [{db_name}]")
        return PostgresConnectionWrapper(conn)
    except Exception as e:
        print(f">>> CRITICAL: PostgreSQL Connection Failed: {e}")
        log.error("Database connection failed: %s", e)
        raise e

def init_db() -> None:
    """Initialize mandatory PostgreSQL tables."""
    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor()

        # User Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email         VARCHAR(255) PRIMARY KEY,
                name          VARCHAR(255) NOT NULL,
                company       VARCHAR(255) NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)

        # Cameras Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS local_cameras (
                id           SERIAL PRIMARY KEY,
                name         VARCHAR(255),
                brand        VARCHAR(255),
                ip_address   VARCHAR(100),
                port         INTEGER,
                username     VARCHAR(255),
                password     VARCHAR(255),
                stream_path  VARCHAR(500),
                owner_email  VARCHAR(255)
            )
        """)

        # Staff Profiles Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS staff_profiles (
                id            SERIAL PRIMARY KEY,
                name          VARCHAR(255) NOT NULL,
                email         VARCHAR(255),
                phone         VARCHAR(100),
                address       TEXT,
                folder_path   TEXT,
                status        VARCHAR(50) DEFAULT 'active',
                communication TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Mandatory Schema Updates
        try:
            cur.execute("ALTER TABLE staff_profiles ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active'")
            cur.execute("ALTER TABLE staff_profiles ADD COLUMN IF NOT EXISTS communication TEXT")
        except: pass

        conn.commit()
        log.info("PostgreSQL database initialized successfully.")
    except Exception as e:
        log.error("Database initialization failed: %s", e)
        print(f"DATABASE ERROR: {e}")
    finally:
        if conn:
            conn.close()
