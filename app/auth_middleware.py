from flask import redirect, url_for, request
from flask_login import current_user, login_required
from functools import wraps

def protect_dash_views(dash_app):
    """Middleware to protect Dash routes with Flask-Login authentication."""
    
    # Protect the main Dash route
    @dash_app.server.before_request
    def check_dash_auth():
        if request.path.startswith(dash_app.config.url_base_pathname):
            if not current_user.is_authenticated:
                return redirect(url_for('auth.login', next=request.url))
        return None