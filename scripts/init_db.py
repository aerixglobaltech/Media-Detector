"""
scripts/init_db.py  –  Seed Default PostgreSQL Admin
──────────────────────────────────────────────────────────────────────────────
Creates a default admin user in the PostgreSQL database.
"""

from __future__ import annotations

import argparse
import sys
import psycopg2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from werkzeug.security import generate_password_hash
from app.db.session import get_db_connection, init_db

def main() -> int:
    parser = argparse.ArgumentParser(description="Seed default admin user to PostgreSQL.")
    parser.add_argument("--email", default="admin@growmax.com")
    parser.add_argument("--name", default="Aerix")
    parser.add_argument("--company", default="Aerixnova technologies")
    parser.add_argument("--password", default="Admin@123")
    args = parser.parse_args()

    # Ensure PostgreSQL tables exist
    init_db()

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Check if user exists
        cur.execute("SELECT 1 FROM users WHERE email = %s", (args.email,))
        if cur.fetchone():
            print(f"User already exists in PostgreSQL: {args.email}")
            return 0

        # Insert new user
        pw_hash = generate_password_hash(args.password)
        cur.execute(
            """
            INSERT INTO users (email, name, company, password_hash)
            VALUES (%s, %s, %s, %s)
            """,
            (args.email, args.name, args.company, pw_hash),
        )
        conn.commit()
        print(f"Seeded Admin User into PostgreSQL: {args.email}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
