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

        # Roles Table (RBAC)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id          SERIAL PRIMARY KEY,
                name        VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                permissions JSONB DEFAULT '{}',
                is_system   BOOLEAN DEFAULT FALSE,
                status      VARCHAR(50) DEFAULT 'active'
            )
        """)

        # User Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email         VARCHAR(255) PRIMARY KEY,
                name          VARCHAR(255) NOT NULL,
                company       VARCHAR(255) NOT NULL,
                password_hash TEXT NOT NULL,
                role_id       INTEGER,
                phone         VARCHAR(50),
                status        VARCHAR(50) DEFAULT 'active',
                avatar        TEXT,
                last_login    TIMESTAMP,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure Role foreign key and other columns exist (for existing tables)
        try:
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role_id INTEGER REFERENCES roles(id)")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(50)")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active'")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            cur.execute("ALTER TABLE roles ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active'")
        except Exception: 
            conn.rollback()
            cur = conn.cursor()

        # Seed Default Roles
        cur.execute("SELECT COUNT(*) FROM roles")
        if cur.fetchone()['count'] == 0:
            print(">>> SEEDING: Default Roles")
            default_roles = [
                ('Administrator', 'Full system access', '{"all": true}', True),
                ('Manager', 'Management access', '{"cameras_view": true, "cameras_edit": true, "staff_view": true, "staff_edit": true}', False),
                ('User', 'Standard access', '{"cameras_view": true, "staff_view": true}', False)
            ]
            for r_name, r_desc, r_perms, r_sys in default_roles:
                cur.execute(
                    "INSERT INTO roles (name, description, permissions, is_system) VALUES (%s, %s, %s, %s)",
                    (r_name, r_desc, r_perms, r_sys)
                )

        # Assign Default Role (Administrator) to any user without a role
        cur.execute("SELECT id FROM roles WHERE name = 'Administrator'")
        admin_role = cur.fetchone()
        if admin_role:
            cur.execute("UPDATE users SET role_id = %s WHERE role_id IS NULL", (admin_role['id'],))

        # System Settings Table (Branding, Appearance, etc.)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key   VARCHAR(100) PRIMARY KEY,
                value TEXT
            )
        """)

        # Seed Default Settings if table is empty or missing keys
        default_settings = [
            ('company_name',     'MISSION CONTROL'),
            ('logo_url',         ''),
            ('favicon_url',      ''),
            ('theme_mode',       'light'),
            ('show_breadcrumbs', 'true')
        ]
        for key, val in default_settings:
            cur.execute("INSERT INTO system_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING", (key, val))

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
                staff_id      VARCHAR(100),
                status        VARCHAR(50) DEFAULT 'active',
                communication TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Telegram Bots Table (for multiple bot support)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telegram_bots (
                id SERIAL PRIMARY KEY,
                bot_name VARCHAR(100) NOT NULL,
                bot_token TEXT NOT NULL,
                chat_ids TEXT NOT NULL,
                phone_number VARCHAR(50),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Telegram Users Table (bot registration: phone -> chat_id)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telegram_users (
                id SERIAL PRIMARY KEY,
                phone_number VARCHAR(30) UNIQUE NOT NULL,
                chat_id BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Movement Log Table (Matched to member_timestamp as requested)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS movement_log (
                id SERIAL PRIMARY KEY,
                camera_id VARCHAR(100),
                camera_name VARCHAR(100),
                entry_image TEXT,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_image TEXT,
                exit_time TIMESTAMP,
                merged_image TEXT,
                image_path TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                person_type VARCHAR(20),
                staff_id INTEGER NULL REFERENCES staff_profiles(id),
                staff_name VARCHAR(100),
                confidence_score FLOAT
            )
        """)
        
        m_cols = [
            ("camera_name", "VARCHAR(100)"),
            ("camera_id", "VARCHAR(100)"),
            ("entry_image", "TEXT"),
            ("image_path", "TEXT"),
            ("entry_time", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ("detected_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ("exit_image", "TEXT"),
            ("exit_time", "TIMESTAMP"),
            ("merged_image", "TEXT"),
            ("person_type", "VARCHAR(20)"),
            ("staff_id", "INTEGER"),
            ("staff_name", "VARCHAR(100)"),
            ("confidence_score", "FLOAT")
        ]
        for col, dtype in m_cols:
            try:
                cur.execute(f"ALTER TABLE movement_log ADD COLUMN IF NOT EXISTS {col} {dtype}")
            except Exception:
                conn.rollback()
                cur = conn.cursor()
        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_movement_detected_at ON movement_log(detected_at)")

        # Member Timestamp Table (Restoring rich forensics)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS member_timestamp (
                id SERIAL PRIMARY KEY,
                camera_id VARCHAR(100),
                camera_name VARCHAR(100),
                person_type VARCHAR(20),     
                staff_id INTEGER NULL REFERENCES staff_profiles(id),
                staff_name VARCHAR(100),
                confidence_score FLOAT,
                entry_image TEXT,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_image TEXT,
                exit_time TIMESTAMP,
                merged_image TEXT,
                image_path TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Ensure all columns exist to satisfy both legacy and newer code
        cols = [
            ("camera_name", "VARCHAR(100)"),
            ("camera_id", "VARCHAR(100)"),
            ("entry_image", "TEXT"),
            ("image_path", "TEXT"),
            ("entry_time", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ("detected_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
            ("exit_image", "TEXT"),
            ("exit_time", "TIMESTAMP"),
            ("merged_image", "TEXT"),
            ("person_type", "VARCHAR(20)"),
            ("staff_id", "INTEGER"),
            ("staff_name", "VARCHAR(100)"),
            ("confidence_score", "FLOAT")
        ]
        for col, dtype in cols:
            try:
                cur.execute(f"ALTER TABLE member_timestamp ADD COLUMN IF NOT EXISTS {col} {dtype}")
            except Exception:
                conn.rollback()
                cur = conn.cursor()

        cur.execute("CREATE INDEX IF NOT EXISTS idx_member_timestamp_staff ON member_timestamp(staff_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_member_timestamp_entry ON member_timestamp(entry_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_member_timestamp_detected ON member_timestamp(detected_at)")

        # Mandatory updates for legacy table if it already exists
        try:
            cur.execute("ALTER TABLE member_timestamp ADD COLUMN IF NOT EXISTS person_type VARCHAR(20)")
            cur.execute("ALTER TABLE member_timestamp ADD COLUMN IF NOT EXISTS staff_id INTEGER REFERENCES staff_profiles(id)")
            cur.execute("ALTER TABLE member_timestamp ADD COLUMN IF NOT EXISTS confidence_score FLOAT")
        except Exception:
            conn.rollback()
            cur = conn.cursor()

        # Attendance Table (Restore Legacy & New)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                staff_id INTEGER REFERENCES staff_profiles(id) ON DELETE CASCADE,
                staff_name VARCHAR(100),
                attendance_date DATE NOT NULL,
                first_entry_time TIMESTAMP,
                last_exit_time TIMESTAMP,
                status VARCHAR(10),
                in_time TIMESTAMP,
                out_time TIMESTAMP,
                entry_image TEXT,
                in_image TEXT,
                out_image TEXT,
                camera_name VARCHAR(100),
                movement_count INTEGER DEFAULT 0,
                total_duration_minutes INTEGER DEFAULT 0,
                day_status VARCHAR(20) DEFAULT 'open',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS unique_staff_attendance ON attendance(staff_id, attendance_date)")

        # Backward-compatible schema alignment for existing attendance tables
        try:
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS staff_id INTEGER REFERENCES staff_profiles(id) ON DELETE RESTRICT")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS status VARCHAR(10)")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS in_time TIMESTAMP")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS out_time TIMESTAMP")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS staff_name VARCHAR(100)")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS entry_image TEXT")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS day_status VARCHAR(20) DEFAULT 'open'")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS movement_count INTEGER DEFAULT 0")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS total_duration_minutes INTEGER DEFAULT 0")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS camera_name VARCHAR(100)")
        except Exception:
            conn.rollback()
            cur = conn.cursor()

        # Helpful lookup index for attendance notifications
        cur.execute("CREATE INDEX IF NOT EXISTS idx_telegram_users_phone_number ON telegram_users(phone_number)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_staff_time ON attendance(staff_id, timestamp)")

        # Seed sample attendance rows from staff profiles (first-time bootstrap)
        cur.execute("""
            WITH staff_pool AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY id) AS rn,
                    COUNT(*) OVER () AS total_staff
                FROM staff_profiles
            ),
            slots AS (
                SELECT generate_series(1, 10) AS n
            ),
            picked_staff AS (
                SELECT
                    sp.id AS staff_id,
                    s.n
                FROM slots s
                JOIN staff_pool sp
                  ON sp.rn = ((s.n - 1) % sp.total_staff) + 1
            )
            INSERT INTO attendance (staff_id, attendance_date, status, in_time, out_time, timestamp)
            SELECT
                p.staff_id,
                (CURRENT_DATE - ((11 - p.n) * INTERVAL '1 day'))::DATE,
                'OUT',
                CURRENT_TIMESTAMP - ((11 - p.n) * INTERVAL '24 hours'),
                CURRENT_TIMESTAMP - ((11 - p.n) * INTERVAL '24 hours') + INTERVAL '8 hours 15 minutes',
                CURRENT_TIMESTAMP - ((11 - p.n) * INTERVAL '24 hours')
            FROM picked_staff p
            WHERE EXISTS (SELECT 1 FROM staff_profiles)
              AND NOT EXISTS (SELECT 1 FROM attendance)
        """)

        # Migrate existing settings to telegram_bots if first time
        cur.execute("SELECT COUNT(*) FROM telegram_bots")
        if cur.fetchone()['count'] == 0:
            cur.execute("SELECT value FROM system_settings WHERE key = 'TELEGRAM_BOT_TOKEN'")
            token_row = cur.fetchone()
            if token_row and token_row['value']:
                cur.execute("SELECT value FROM system_settings WHERE key = 'TELEGRAM_CHAT_ID'")
                chat_row = cur.fetchone()
                cur.execute("SELECT value FROM system_settings WHERE key = 'TELEGRAM_PHONE'")
                phone_row = cur.fetchone()
                
                cur.execute(
                    "INSERT INTO telegram_bots (bot_name, bot_token, chat_ids, phone_number) VALUES (%s, %s, %s, %s)",
                    ("Primary Bot", token_row['value'], chat_row['value'] if chat_row else "", phone_row['value'] if phone_row else "")
                )

        # Remove legacy/migrated settings from system_settings
        cur.execute("DELETE FROM system_settings WHERE key IN ('IMOU_APP_ID', 'IMOU_APP_SECRET', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'TELEGRAM_PHONE')")

        # Mandatory Schema Updates
        try:
            cur.execute("ALTER TABLE staff_profiles ADD COLUMN IF NOT EXISTS staff_id VARCHAR(100)")
            cur.execute("ALTER TABLE staff_profiles ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active'")
            cur.execute("ALTER TABLE staff_profiles ADD COLUMN IF NOT EXISTS communication TEXT")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS in_image TEXT")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS out_image TEXT")
            
            # Ensure attendance table has correct columns for the new logic
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS attendance_date DATE")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS first_entry_time TIMESTAMP")
            cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS last_exit_time TIMESTAMP")
            
            # Make staff_id nullable in attendance table for 'Unknown' person logging
            cur.execute("ALTER TABLE attendance ALTER COLUMN staff_id DROP NOT NULL")
            
            # Add unique constraint if not exists
            try:
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS unique_staff_attendance ON attendance(staff_id, attendance_date)")
            except Exception:
                conn.rollback()
                cur = conn.cursor()
            # Ensure staff_name exists in both tables
            for table in ["member_timestamp", "movement_log"]:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS staff_name VARCHAR(100)")
            
            # Remove person_id if it exists (cleanup)
            for table in ["member_timestamp", "movement_log"]:
                try: cur.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS person_id")
                except: conn.rollback(); cur = conn.cursor()

            conn.commit()
            log.info("Forced forensic schema updates applied.")
        except Exception as e:
            log.warning(f"Schema update minor error: {e}")

        conn.commit()
        log.info("PostgreSQL database initialized with RBAC support.")
    except Exception as e:
        log.error("Database initialization failed: %s", e)
        print(f"DATABASE ERROR: {e}")
    finally:
        if conn:
            conn.close()
