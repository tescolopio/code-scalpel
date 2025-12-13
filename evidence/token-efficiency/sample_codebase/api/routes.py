"""API route handlers."""

from flask import Blueprint, request, jsonify, g
from functools import wraps
from typing import Callable

from ..services.auth_service import AuthService, PermissionService
from ..models.user import User, UserRepository


api = Blueprint('api', __name__)


def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing authentication"}), 401

        token = auth_header.split(' ')[1]
        user = g.auth_service.validate_access_token(token)

        if not user:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.current_user = user
        return f(*args, **kwargs)

    return decorated


def require_permission(permission: str) -> Callable:
    """Decorator to require specific permission."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            if not g.permission_service.has_permission(g.current_user, permission):
                return jsonify({"error": "Permission denied"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ============= Authentication Routes =============

@api.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    result = g.auth_service.authenticate(username, password)

    if not result.success:
        return jsonify({"error": result.error}), 401

    if result.requires_2fa:
        return jsonify({
            "requires_2fa": True,
            "user_id": result.user.id
        }), 200

    return jsonify({
        "access_token": result.token_pair.access_token,
        "refresh_token": result.token_pair.refresh_token,
        "expires_at": result.token_pair.expires_at.isoformat(),
        "user": result.user.to_dict()
    }), 200


@api.route('/auth/login/2fa', methods=['POST'])
def login_2fa():
    """Complete 2FA login."""
    data = request.get_json()
    user_id = data.get('user_id')
    code = data.get('code')

    if not user_id or not code:
        return jsonify({"error": "User ID and code required"}), 400

    user = g.user_repo.find_by_id(user_id)
    if not user:
        return jsonify({"error": "Invalid user"}), 400

    result = g.auth_service.authenticate_with_2fa(user, code)

    if not result.success:
        return jsonify({"error": result.error}), 401

    return jsonify({
        "access_token": result.token_pair.access_token,
        "refresh_token": result.token_pair.refresh_token,
        "expires_at": result.token_pair.expires_at.isoformat(),
        "user": result.user.to_dict()
    }), 200


@api.route('/auth/refresh', methods=['POST'])
def refresh_token():
    """Refresh access token."""
    data = request.get_json()
    refresh_token = data.get('refresh_token')

    if not refresh_token:
        return jsonify({"error": "Refresh token required"}), 400

    token_pair = g.auth_service.refresh_tokens(refresh_token)

    if not token_pair:
        return jsonify({"error": "Invalid or expired refresh token"}), 401

    return jsonify({
        "access_token": token_pair.access_token,
        "refresh_token": token_pair.refresh_token,
        "expires_at": token_pair.expires_at.isoformat()
    }), 200


@api.route('/auth/logout', methods=['POST'])
@require_auth
def logout():
    """User logout."""
    data = request.get_json()
    refresh_token = data.get('refresh_token')

    if refresh_token:
        g.auth_service.logout(refresh_token)

    return jsonify({"message": "Logged out successfully"}), 200


@api.route('/auth/password', methods=['PUT'])
@require_auth
def change_password():
    """Change user password."""
    data = request.get_json()
    old_password = data.get('old_password')
    new_password = data.get('new_password')

    if not old_password or not new_password:
        return jsonify({"error": "Old and new password required"}), 400

    success, message = g.auth_service.change_password(
        g.current_user, old_password, new_password
    )

    if not success:
        return jsonify({"error": message}), 400

    return jsonify({"message": message}), 200


# ============= User Routes =============

@api.route('/users/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current user profile."""
    return jsonify(g.current_user.to_dict(include_sensitive=True)), 200


@api.route('/users/me', methods=['PUT'])
@require_auth
def update_current_user():
    """Update current user profile."""
    data = request.get_json()

    allowed_fields = ['username', 'email']
    for field in allowed_fields:
        if field in data:
            setattr(g.current_user, field, data[field])

    g.user_repo.save(g.current_user)

    return jsonify(g.current_user.to_dict(include_sensitive=True)), 200


@api.route('/users/me/preferences', methods=['PUT'])
@require_auth
def update_preferences():
    """Update user preferences."""
    data = request.get_json()

    prefs = g.current_user.preferences
    for key, value in data.items():
        if hasattr(prefs, key):
            setattr(prefs, key, value)

    g.user_repo.save(g.current_user)

    return jsonify(prefs.to_dict()), 200


@api.route('/users', methods=['GET'])
@require_permission('manage_users')
def list_users():
    """List all users (admin only)."""
    users = g.user_repo.find_all_active()
    return jsonify([u.to_dict() for u in users]), 200


@api.route('/users/<int:user_id>', methods=['GET'])
@require_permission('manage_users')
def get_user(user_id: int):
    """Get user by ID (admin only)."""
    user = g.user_repo.find_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify(user.to_dict(include_sensitive=True)), 200


@api.route('/users/<int:user_id>', methods=['PUT'])
@require_permission('manage_users')
def update_user(user_id: int):
    """Update user (admin only)."""
    user = g.user_repo.find_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json()

    allowed_fields = ['username', 'email', 'is_active', 'is_admin', 'roles']
    for field in allowed_fields:
        if field in data:
            setattr(user, field, data[field])

    g.user_repo.save(user)

    return jsonify(user.to_dict(include_sensitive=True)), 200


@api.route('/users/<int:user_id>', methods=['DELETE'])
@require_permission('manage_users')
def delete_user(user_id: int):
    """Delete user (admin only)."""
    if user_id == g.current_user.id:
        return jsonify({"error": "Cannot delete yourself"}), 400

    user = g.user_repo.find_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    g.user_repo.delete(user_id)

    return jsonify({"message": "User deleted"}), 200


@api.route('/users/<int:user_id>/lock', methods=['POST'])
@require_permission('manage_users')
def lock_user(user_id: int):
    """Lock user account."""
    user = g.user_repo.find_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json()
    duration = data.get('duration_minutes', 30)

    user.lock_account(duration)
    g.user_repo.save(user)

    return jsonify({"message": f"User locked for {duration} minutes"}), 200


@api.route('/users/<int:user_id>/unlock', methods=['POST'])
@require_permission('manage_users')
def unlock_user(user_id: int):
    """Unlock user account."""
    user = g.user_repo.find_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    user.unlock_account()
    g.user_repo.save(user)

    return jsonify({"message": "User unlocked"}), 200


# ============= Health Check =============

@api.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    }), 200
