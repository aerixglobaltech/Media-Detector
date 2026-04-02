"""
app/routes/views.py  –  Page View Routes Blueprint
──────────────────────────────────────────────────────────────────────────────
Serves the HTML template pages of the MISSION CONTROL web UI.

Routes:
  GET /         → Dashboard overview (login required)
  GET /live     → Live camera feed page (login required)
  GET /settings → Settings page (login required)
"""

from __future__ import annotations

from flask import Blueprint, render_template, session, redirect, url_for

from app.core.security import login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def landing():
    """Landing/Splash page for unauthenticated users."""
    if "user" in session:
        return redirect(url_for("dashboard.index"))
    return render_template("landing.html")


@dashboard_bp.route("/dashboard")
@login_required
def index():
    from app.api.routes.camera import get_all_cameras
    user_email = session.get("user")
    cameras = get_all_cameras(user_email=user_email)
    return render_template("dashboard.html", cameras=cameras)


@dashboard_bp.route("/live")
@login_required
def route_live():
    return render_template("live.html")


@dashboard_bp.route("/attendance")
@login_required
def route_attendance():
    return render_template("attendance.html")


@dashboard_bp.route("/member-logs")
@login_required
def route_member_logs():
    return render_template("member_logs.html")


@dashboard_bp.route("/general-movement")
@login_required
def route_general_movement():
    return render_template("general_movement.html")


@dashboard_bp.route("/reports")
@login_required
def route_reports():
    return render_template("reports.html")


@dashboard_bp.route("/settings")
@login_required
def route_settings():
    return render_template("settings/index.html")


@dashboard_bp.route("/settings/profile")
@login_required
def route_settings_profile():
    return render_template("settings/profile.html")


@dashboard_bp.route("/settings/members")
@dashboard_bp.route("/settings/staff")
@login_required
def route_settings_members():
    return render_template("settings/staff.html")


@dashboard_bp.route("/settings/cameras")
@login_required
def route_settings_cameras():
    return render_template("settings/cameras.html")


@dashboard_bp.route("/settings/integrations")
@login_required
def route_settings_integrations():
    from flask import redirect, url_for
    return redirect(url_for('dashboard.route_telegram'))


@dashboard_bp.route("/settings/users")
@login_required
def route_settings_users():
    return render_template("settings/users.html")


@dashboard_bp.route("/settings/roles")
@login_required
def route_settings_roles():
    return render_template("settings/roles.html")

@dashboard_bp.route("/settings/appearance")
@login_required
def route_settings_appearance():
    return render_template("settings/appearance.html")

@dashboard_bp.route("/telegram")
@login_required
def route_telegram():
    return render_template("telegram.html")

@dashboard_bp.route("/telegram/notifications")
@login_required
def route_telegram_notifications():
    return render_template("telegram_notifications.html")
