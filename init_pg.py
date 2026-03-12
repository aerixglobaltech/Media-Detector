import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
url = os.environ.get("DATABASE_URL")
if url and "postgres" in url:
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    print(f"Connecting to: {url}")
    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        
        # Roles Table (RBAC)
        print("Creating table: roles")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roles (
                id          SERIAL PRIMARY KEY,
                name        VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                permissions JSONB DEFAULT '{}',
                is_system   BOOLEAN DEFAULT FALSE
            )
        """)

        # User Table
        print("Creating/Updating table: users")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email         VARCHAR(255) PRIMARY KEY,
                name          VARCHAR(255) NOT NULL,
                company       VARCHAR(255) NOT NULL,
                password_hash TEXT NOT NULL,
                role_id       INTEGER REFERENCES roles(id),
                phone         VARCHAR(50),
                status        VARCHAR(50) DEFAULT 'active',
                avatar        TEXT,
                last_login    TIMESTAMP,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure Role columns exist (for migration)
        try:
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role_id INTEGER REFERENCES roles(id)")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(50)")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'active'")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP")
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except: pass

        # Seed Default Roles
        cur.execute("SELECT COUNT(*) FROM roles")
        if cur.fetchone()[0] == 0:
            print("Seeding default roles...")
            cur.execute("INSERT INTO roles (name, description, permissions, is_system) VALUES ('Administrator', 'Full system access', '{\"all\": true}', true)")
            cur.execute("INSERT INTO roles (name, description, permissions, is_system) VALUES ('Manager', 'Management access', '{\"cameras_view\": true, \"cameras_edit\": true, \"staff_view\": true, \"staff_edit\": true}', false)")
            cur.execute("INSERT INTO roles (name, description, permissions, is_system) VALUES ('User', 'Standard access', '{\"cameras_view\": true, \"staff_view\": true}', false)")

        # Assign Default Role (Administrator) to any user without a role
        cur.execute("SELECT id FROM roles WHERE name = 'Administrator'")
        res = cur.fetchone()
        if res:
            cur.execute("UPDATE users SET role_id = %s WHERE role_id IS NULL", (res[0],))

        print("Creating table: local_cameras")
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

        print("Creating table: staff_profiles")
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
        
        conn.commit()
        db_name = url.split('/')[-1].split('?')[0]
        print(f"SUCCESS: All tables created in {db_name} database.")
        conn.close()
    except Exception as e:
        print(f"FAILED: {e}")
else:
    print("DATABASE_URL not found or not Postgres.")
