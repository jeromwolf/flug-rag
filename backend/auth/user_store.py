"""In-memory user store backed by hashed passwords (passlib + bcrypt).

For production, replace with a database-backed store.  This module provides
enough functionality for demo / development use with the default users.
"""

from __future__ import annotations

from datetime import datetime, timezone

from passlib.context import CryptContext

from auth.models import Role, User, get_default_users

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Default password map  (username -> plain-text password for demo)
# ---------------------------------------------------------------------------
_DEFAULT_PASSWORDS: dict[str, str] = {
    "admin": "admin123",
    "manager": "manager123",
    "user": "user123",
    "viewer": "viewer123",
}


class UserStore:
    """Simple in-memory user store with bcrypt password hashing."""

    def __init__(self) -> None:
        # user_id -> User
        self._users: dict[str, User] = {}
        # username -> hashed password
        self._passwords: dict[str, str] = {}
        self._init_defaults()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _init_defaults(self) -> None:
        """Seed the store with built-in demo users."""
        for u in get_default_users():
            user = User(**u)
            self._users[user.id] = user
            plain = _DEFAULT_PASSWORDS.get(user.username, "changeme")
            self._passwords[user.username] = pwd_context.hash(plain)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def authenticate(self, username: str, password: str) -> User | None:
        """Verify credentials. Returns ``User`` on success, ``None`` on failure."""
        hashed = self._passwords.get(username)
        if hashed is None:
            return None
        if not pwd_context.verify(password, hashed):
            return None
        user = self.get_by_username(username)
        if user is None or not user.is_active:
            return None
        # Update last_login
        user.last_login = datetime.now(timezone.utc)
        return user

    def get_by_username(self, username: str) -> User | None:
        for u in self._users.values():
            if u.username == username:
                return u
        return None

    def get_by_id(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def list_users(self) -> list[User]:
        return list(self._users.values())

    def update_role(self, user_id: str, new_role: Role) -> User | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        user.role = new_role
        return user

    def set_active(self, user_id: str, active: bool) -> User | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        user.is_active = active
        return user


# Singleton instance
user_store = UserStore()
