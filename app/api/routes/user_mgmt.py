"""
app/api/routes/user_mgmt.py  –  User & RBAC API Routes
──────────────────────────────────────────────────────────────────────────────
Handles CRUD operations for users and roles.
"""

from __future__ import annotations

import werkzeug.security
from flask import Blueprint, request, jsonify, session
from app.db.session import get_db_connection
from app.core.security import login_required

user_mgmt_bp = Blueprint("user_mgmt", __name__)

# ── Users ────────────────────────────────────────────────────────────────────

@user_mgmt_bp.route("/api/settings/users", methods=["GET"])
@login_required
def get_users():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.email, u.name, u.company, u.phone, u.status, u.avatar, u.last_login, r.name as role_name, r.id as role_id
        FROM users u
        LEFT JOIN roles r ON u.role_id = r.id
        ORDER BY u.created_at DESC
    """)
    users = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(users)

@user_mgmt_bp.route("/api/settings/users", methods=["POST"])
@login_required
def create_user():
    data = request.json
    email = data.get("email", "").strip()
    name = data.get("name", "").strip()
    company = data.get("company", "").strip()
    password = data.get("password", "")
    role_id = data.get("role_id")
    phone = data.get("phone", "")

    if not email or not name or not password:
        return jsonify({"success": False, "error": "Email, Name and Password are required"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    
    # Check if exists
    cur.execute("SELECT email FROM users WHERE email = %s", (email,))
    if cur.fetchone():
        cur.close()
        conn.close()
        return jsonify({"success": False, "error": "Email already registered"}), 400

    pw_hash = werkzeug.security.generate_password_hash(password)
    cur.execute("""
        INSERT INTO users (email, name, company, password_hash, role_id, phone)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (email, name, company, pw_hash, role_id, phone))
    
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"success": True})

@user_mgmt_bp.route("/api/settings/users/<email>", methods=["PATCH"])
@login_required
def update_user(email):
    data = request.json
    name = data.get("name")
    company = data.get("company")
    role_id = data.get("role_id")
    phone = data.get("phone")
    status = data.get("status")
    password = data.get("password")

    conn = get_db_connection()
    cur = conn.cursor()

    updates = []
    params = []

    if name is not None:
        updates.append("name = %s")
        params.append(name)
    if company is not None:
        updates.append("company = %s")
        params.append(company)
    if role_id is not None:
        updates.append("role_id = %s")
        params.append(role_id)
    if phone is not None:
        updates.append("phone = %s")
        params.append(phone)
    if status is not None:
        updates.append("status = %s")
        params.append(status)
    if password:
        updates.append("password_hash = %s")
        params.append(werkzeug.security.generate_password_hash(password))

    if not updates:
        return jsonify({"success": False, "error": "No fields to update"}), 400

    params.append(email)
    cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE email = %s", tuple(params))
    
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"success": True})

@user_mgmt_bp.route("/api/settings/users/<email>", methods=["DELETE"])
@login_required
def delete_user(email):
    if email == session.get("user"):
        return jsonify({"success": False, "error": "Cannot delete your own account"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE email = %s", (email,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"success": True})

# ── Roles ────────────────────────────────────────────────────────────────────

@user_mgmt_bp.route("/api/settings/roles", methods=["GET"])
@login_required
def get_roles():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.*, (SELECT COUNT(*) FROM users WHERE role_id = r.id) as user_count
        FROM roles r
        ORDER BY r.id ASC
    """)
    roles = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(roles)

@user_mgmt_bp.route("/api/settings/roles", methods=["POST"])
@login_required
def create_role():
    data = request.json
    name = data.get("name", "").strip()
    description = data.get("description", "")
    permissions = data.get("permissions", {})

    if not name:
        return jsonify({"success": False, "error": "Role name is required"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        import json
        cur.execute("""
            INSERT INTO roles (name, description, permissions, status)
            VALUES (%s, %s, %s, 'active')
        """, (name, description, json.dumps(permissions)))
        conn.commit()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
    finally:
        cur.close()
        conn.close()
    return jsonify({"success": True})

@user_mgmt_bp.route("/api/settings/roles/<int:role_id>", methods=["PATCH"])
@login_required
def update_role(role_id):
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            # Fallback for form data if somehow sent that way
            data = request.form.to_dict()
        
        print(f"DEBUG: Role Update ID={role_id}, Data={data}")
        
        updates = []
        params = []

        if data.get("name") is not None:
            updates.append("name = %s")
            params.append(data.get("name"))
            
        if data.get("description") is not None:
            updates.append("description = %s")
            params.append(data.get("description"))
            
        if data.get("permissions") is not None:
            import json
            updates.append("permissions = %s")
            params.append(json.dumps(data.get("permissions")))
            
        if data.get("status") is not None:
            updates.append("status = %s")
            params.append(data.get("status"))

        if not updates:
            print(f"ERROR: No update fields found in request for role {role_id}. Data was: {data}")
            return jsonify({"success": False, "error": "No fields to update"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        
        params.append(role_id)
        query = f"UPDATE roles SET {', '.join(updates)} WHERE id = %s"
        print(f"DEBUG: Executing Query: {query} with Params: {params}")
        
        cur.execute(query, tuple(params))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        print(f"ERROR in update_role: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@user_mgmt_bp.route("/api/settings/roles/<int:role_id>", methods=["DELETE"])
@login_required
def delete_role(role_id):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Check if system role
    cur.execute("SELECT is_system FROM roles WHERE id = %s", (role_id,))
    role = cur.fetchone()
    if role and role['is_system']:
        cur.close()
        conn.close()
        return jsonify({"success": False, "error": "System roles cannot be deleted"}), 400

    # check if users assigned
    cur.execute("SELECT COUNT(*) FROM users WHERE role_id = %s", (role_id,))
    if cur.fetchone()['count'] > 0:
        cur.close()
        conn.close()
        return jsonify({"success": False, "error": "Cannot delete role with assigned users"}), 400

    cur.execute("DELETE FROM roles WHERE id = %s", (role_id,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"success": True})
