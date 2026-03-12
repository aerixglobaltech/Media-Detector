"""
app/routes/auth.py  –  Authentication Routes Blueprint
──────────────────────────────────────────────────────────────────────────────
Handles user registration, login, and logout.

Routes:
  GET  /login   → show login form
  POST /login   → authenticate credentials
  GET  /signup  → show registration form
  POST /signup  → create new user account
  GET  /logout  → clear session and redirect to login
"""

from __future__ import annotations

import werkzeug.security
from flask import Blueprint, render_template, request, session, redirect, url_for

from app.db.session import get_db_connection

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def route_login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        user_record = cur.fetchone()

        if user_record and werkzeug.security.check_password_hash(
            user_record["password_hash"], password
        ):
            # Update last login
            from datetime import datetime
            cur.execute("UPDATE users SET last_login = %s WHERE email = %s", (datetime.now(), email))
            conn.commit()
            cur.close()
            conn.close()

            session["user"] = email
            return redirect(url_for("dashboard.index"))

        cur.close()
        conn.close()
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@auth_bp.route("/signup", methods=["GET", "POST"])
def route_signup():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        name     = request.form.get("username", "").strip()
        company  = request.form.get("company", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        cur  = conn.cursor()

        # Check if email already exists
        cur.execute("SELECT email FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return render_template("signup.html", error="Email already registered")

        # Get default user role
        cur.execute("SELECT id FROM roles WHERE name = 'User'")
        role_res = cur.fetchone()
        role_id = role_res['id'] if role_res else None

        # Insert new user
        pw_hash = werkzeug.security.generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (email, name, company, password_hash, role_id) VALUES (%s, %s, %s, %s, %s)",
            (email, name, company, pw_hash, role_id),
        )
        conn.commit()
        cur.close()
        conn.close()

        session["user"] = email
        return redirect(url_for("dashboard.index"))

    return render_template("signup.html")


@auth_bp.route("/logout")
def route_logout():
    session.pop("user", None)
    return redirect(url_for("auth.route_login"))
