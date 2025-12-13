"""Authentication service layer."""

import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass

from ..models.user import User, UserRepository


@dataclass
class TokenPair:
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    expires_at: datetime
    token_type: str = "Bearer"


@dataclass
class AuthResult:
    """Result of authentication attempt."""
    success: bool
    user: Optional[User] = None
    token_pair: Optional[TokenPair] = None
    error: Optional[str] = None
    requires_2fa: bool = False


class AuthService:
    """Service for user authentication."""

    JWT_SECRET = "your-secret-key"  # Should be from config
    ACCESS_TOKEN_EXPIRY_HOURS = 1
    REFRESH_TOKEN_EXPIRY_DAYS = 30

    def __init__(self, user_repository: UserRepository):
        self.user_repo = user_repository
        self._refresh_tokens = {}  # In production, use Redis/DB

    def authenticate(self, username: str, password: str) -> AuthResult:
        """Authenticate user with username and password."""
        user = self.user_repo.find_by_username(username)

        if not user:
            return AuthResult(success=False, error="Invalid credentials")

        if not user.is_active:
            return AuthResult(success=False, error="Account is disabled")

        if user.is_locked():
            return AuthResult(success=False, error="Account is temporarily locked")

        if not user.verify_password(password):
            user.record_login_attempt(success=False)
            self.user_repo.save(user)
            return AuthResult(success=False, error="Invalid credentials")

        user.record_login_attempt(success=True)
        self.user_repo.save(user)

        if user.preferences.two_factor_enabled:
            return AuthResult(success=True, user=user, requires_2fa=True)

        token_pair = self._generate_tokens(user)
        return AuthResult(success=True, user=user, token_pair=token_pair)

    def authenticate_with_2fa(self, user: User, code: str) -> AuthResult:
        """Complete 2FA authentication."""
        if not self._verify_2fa_code(user, code):
            return AuthResult(success=False, error="Invalid 2FA code")

        token_pair = self._generate_tokens(user)
        return AuthResult(success=True, user=user, token_pair=token_pair)

    def refresh_tokens(self, refresh_token: str) -> Optional[TokenPair]:
        """Refresh access token using refresh token."""
        try:
            payload = jwt.decode(refresh_token, self.JWT_SECRET, algorithms=["HS256"])
            user_id = payload.get("user_id")
            token_id = payload.get("token_id")

            if token_id not in self._refresh_tokens:
                return None

            if self._refresh_tokens[token_id] != user_id:
                return None

            user = self.user_repo.find_by_id(user_id)
            if not user or not user.is_active:
                return None

            # Invalidate old refresh token
            del self._refresh_tokens[token_id]

            return self._generate_tokens(user)

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def validate_access_token(self, token: str) -> Optional[User]:
        """Validate access token and return user."""
        try:
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=["HS256"])
            user_id = payload.get("user_id")
            user = self.user_repo.find_by_id(user_id)

            if user and user.is_active:
                return user
            return None

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def logout(self, refresh_token: str) -> bool:
        """Invalidate refresh token on logout."""
        try:
            payload = jwt.decode(refresh_token, self.JWT_SECRET, algorithms=["HS256"])
            token_id = payload.get("token_id")
            if token_id in self._refresh_tokens:
                del self._refresh_tokens[token_id]
                return True
            return False
        except jwt.InvalidTokenError:
            return False

    def change_password(self, user: User, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password."""
        if not user.verify_password(old_password):
            return False, "Current password is incorrect"

        if len(new_password) < 8:
            return False, "Password must be at least 8 characters"

        user.password_hash = User.hash_password(new_password)
        self.user_repo.save(user)

        # Invalidate all refresh tokens for this user
        self._invalidate_user_tokens(user.id)

        return True, "Password changed successfully"

    def request_password_reset(self, email: str) -> Optional[str]:
        """Request password reset token."""
        user = self.user_repo.find_by_email(email)
        if not user:
            # Don't reveal if email exists
            return None

        reset_token = secrets.token_urlsafe(32)
        # In production, store token with expiry in DB
        return reset_token

    def reset_password(self, reset_token: str, new_password: str) -> Tuple[bool, str]:
        """Reset password using reset token."""
        # In production, validate token from DB
        if len(new_password) < 8:
            return False, "Password must be at least 8 characters"

        # Would find user from token and update password
        return True, "Password reset successfully"

    def _generate_tokens(self, user: User) -> TokenPair:
        """Generate access and refresh token pair."""
        now = datetime.utcnow()
        access_expiry = now + timedelta(hours=self.ACCESS_TOKEN_EXPIRY_HOURS)
        refresh_expiry = now + timedelta(days=self.REFRESH_TOKEN_EXPIRY_DAYS)

        access_payload = {
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.is_admin,
            "roles": user.roles,
            "exp": access_expiry,
            "iat": now,
            "type": "access"
        }

        token_id = secrets.token_hex(16)
        refresh_payload = {
            "user_id": user.id,
            "token_id": token_id,
            "exp": refresh_expiry,
            "iat": now,
            "type": "refresh"
        }

        access_token = jwt.encode(access_payload, self.JWT_SECRET, algorithm="HS256")
        refresh_token = jwt.encode(refresh_payload, self.JWT_SECRET, algorithm="HS256")

        self._refresh_tokens[token_id] = user.id

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=access_expiry
        )

    def _verify_2fa_code(self, user: User, code: str) -> bool:
        """Verify 2FA code."""
        # In production, use TOTP library
        return code == "123456"  # Placeholder

    def _invalidate_user_tokens(self, user_id: int):
        """Invalidate all refresh tokens for a user."""
        tokens_to_remove = [
            token_id for token_id, uid in self._refresh_tokens.items()
            if uid == user_id
        ]
        for token_id in tokens_to_remove:
            del self._refresh_tokens[token_id]


class PermissionService:
    """Service for checking user permissions."""

    PERMISSIONS = {
        "admin": ["read", "write", "delete", "manage_users"],
        "editor": ["read", "write"],
        "viewer": ["read"]
    }

    def __init__(self, user_repository: UserRepository):
        self.user_repo = user_repository

    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission."""
        if user.is_admin:
            return True

        for role in user.roles:
            role_permissions = self.PERMISSIONS.get(role, [])
            if permission in role_permissions:
                return True

        return False

    def get_user_permissions(self, user: User) -> set:
        """Get all permissions for a user."""
        if user.is_admin:
            return set(sum(self.PERMISSIONS.values(), []))

        permissions = set()
        for role in user.roles:
            permissions.update(self.PERMISSIONS.get(role, []))

        return permissions

    def require_permission(self, user: User, permission: str):
        """Raise error if user lacks permission."""
        if not self.has_permission(user, permission):
            raise PermissionError(f"User lacks '{permission}' permission")

    def can_access_resource(self, user: User, resource_owner_id: int, permission: str) -> bool:
        """Check if user can access a specific resource."""
        # Users can always access their own resources
        if user.id == resource_owner_id:
            return True

        # Otherwise check permission
        return self.has_permission(user, permission)
