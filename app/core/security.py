"""
app/auth.py  –  Authentication Decorator
──────────────────────────────────────────────────────────────────────────────
Provides the login_required decorator used by all protected routes.
"""

from __future__ import annotations

from functools import wraps
from flask import session, redirect, url_for


def login_required(f):
    """Redirect unauthenticated users to the login page."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("auth.route_login"))
        return f(*args, **kwargs)
    return decorated_function
