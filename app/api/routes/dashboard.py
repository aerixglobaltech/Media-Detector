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

from flask import Blueprint, render_template

from app.core.security import login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
@login_required
def index():
    return render_template("dashboard.html")


@dashboard_bp.route("/live")
@login_required
def route_live():
    return render_template("live.html")


@dashboard_bp.route("/attendance")
@login_required
def route_attendance():
    return render_template("attendance.html")


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


@dashboard_bp.route("/settings/staff")
@login_required
def route_settings_staff():
    return render_template("settings/staff.html")


@dashboard_bp.route("/settings/cameras")
@login_required
def route_settings_cameras():
    return render_template("settings/cameras.html")


@dashboard_bp.route("/settings/integrations")
@login_required
def route_settings_integrations():
    return render_template("settings/integrations.html")
