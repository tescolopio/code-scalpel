"""User model and authentication logic."""

from datetime import datetime, timedelta
from typing import Optional, List
import hashlib
import secrets
from dataclasses import dataclass, field


@dataclass
class Address:
    """User address information."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"

    def format_full(self) -> str:
        """Format as full mailing address."""
        return f"{self.street}\n{self.city}, {self.state} {self.zip_code}\n{self.country}"

    def format_short(self) -> str:
        """Format as short address."""
        return f"{self.city}, {self.state}"


@dataclass
class UserPreferences:
    """User preference settings."""
    theme: str = "light"
    language: str = "en"
    timezone: str = "UTC"
    notifications_enabled: bool = True
    email_digest: str = "daily"
    two_factor_enabled: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "theme": self.theme,
            "language": self.language,
            "timezone": self.timezone,
            "notifications_enabled": self.notifications_enabled,
            "email_digest": self.email_digest,
            "two_factor_enabled": self.two_factor_enabled
        }


@dataclass
class User:
    """Main user model."""
    id: int
    username: str
    email: str
    password_hash: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_admin: bool = False
    last_login: Optional[datetime] = None
    login_count: int = 0
    failed_login_count: int = 0
    locked_until: Optional[datetime] = None
    address: Optional[Address] = None
    preferences: UserPreferences = field(default_factory=UserPreferences)
    roles: List[str] = field(default_factory=list)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """Hash a password with optional salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${hashed.hex()}"

    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        if '$' not in self.password_hash:
            return False
        salt, _ = self.password_hash.split('$', 1)
        return self.hash_password(password, salt) == self.password_hash

    def is_locked(self) -> bool:
        """Check if account is locked."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until

    def lock_account(self, duration_minutes: int = 30):
        """Lock account for specified duration."""
        self.locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)

    def unlock_account(self):
        """Unlock the account."""
        self.locked_until = None
        self.failed_login_count = 0

    def record_login_attempt(self, success: bool):
        """Record a login attempt."""
        if success:
            self.last_login = datetime.utcnow()
            self.login_count += 1
            self.failed_login_count = 0
            self.locked_until = None
        else:
            self.failed_login_count += 1
            if self.failed_login_count >= 5:
                self.lock_account()

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles or self.is_admin

    def add_role(self, role: str):
        """Add a role to the user."""
        if role not in self.roles:
            self.roles.append(role)

    def remove_role(self, role: str):
        """Remove a role from the user."""
        if role in self.roles:
            self.roles.remove(role)

    def get_display_name(self) -> str:
        """Get display name for UI."""
        return self.username

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email if include_sensitive else self._mask_email(),
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "roles": self.roles,
            "preferences": self.preferences.to_dict()
        }
        if self.address:
            data["address"] = self.address.format_short()
        return data

    def _mask_email(self) -> str:
        """Mask email for privacy."""
        local, domain = self.email.split('@')
        if len(local) <= 2:
            masked_local = local[0] + '*'
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        return f"{masked_local}@{domain}"


class UserRepository:
    """Repository for user data access."""

    def __init__(self, db_connection):
        self.db = db_connection
        self._cache = {}

    def find_by_id(self, user_id: int) -> Optional[User]:
        """Find user by ID."""
        if user_id in self._cache:
            return self._cache[user_id]

        result = self.db.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

        if result:
            user = self._row_to_user(result)
            self._cache[user_id] = user
            return user
        return None

    def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        result = self.db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        return self._row_to_user(result) if result else None

    def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        result = self.db.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        return self._row_to_user(result) if result else None

    def save(self, user: User) -> User:
        """Save or update user."""
        user.updated_at = datetime.utcnow()

        if user.id:
            self.db.execute(
                """UPDATE users SET username=?, email=?, password_hash=?,
                   is_active=?, is_admin=?, updated_at=? WHERE id=?""",
                (user.username, user.email, user.password_hash,
                 user.is_active, user.is_admin, user.updated_at, user.id)
            )
            self._cache[user.id] = user
        else:
            cursor = self.db.execute(
                """INSERT INTO users (username, email, password_hash, is_active, is_admin)
                   VALUES (?, ?, ?, ?, ?)""",
                (user.username, user.email, user.password_hash,
                 user.is_active, user.is_admin)
            )
            user.id = cursor.lastrowid

        self.db.commit()
        return user

    def delete(self, user_id: int) -> bool:
        """Delete user by ID."""
        self.db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.db.commit()
        self._cache.pop(user_id, None)
        return True

    def find_all_active(self) -> List[User]:
        """Find all active users."""
        results = self.db.execute(
            "SELECT * FROM users WHERE is_active = 1"
        ).fetchall()
        return [self._row_to_user(r) for r in results]

    def count_by_role(self, role: str) -> int:
        """Count users with specific role."""
        result = self.db.execute(
            "SELECT COUNT(*) FROM users WHERE roles LIKE ?",
            (f'%{role}%',)
        ).fetchone()
        return result[0] if result else 0

    def _row_to_user(self, row) -> User:
        """Convert database row to User object."""
        return User(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            password_hash=row['password_hash'],
            is_active=bool(row['is_active']),
            is_admin=bool(row['is_admin'])
        )
