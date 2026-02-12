"""SQLite-backed user store with bcrypt password hashing.

Persists user data across server restarts using aiosqlite.
Seeds default demo users on first database creation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from passlib.context import CryptContext

from auth.models import Role, User, get_default_users
from core.db.base import AsyncSQLiteManager, create_async_singleton

logger = logging.getLogger(__name__)

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


class UserStore(AsyncSQLiteManager):
    """SQLite-backed user store with bcrypt password hashing."""

    def __init__(self) -> None:
        super().__init__(Path("data/users.db"))

    # ------------------------------------------------------------------
    # Schema & Seeding
    # ------------------------------------------------------------------

    async def _create_tables(self, db: aiosqlite.Connection) -> None:
        """Create the users table and seed default demo users."""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                full_name TEXT,
                department TEXT,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                password_hash TEXT,
                created_at TEXT,
                last_login TEXT
            )
        """)
        await db.commit()

        # Seed default users if table is empty
        async with db.execute("SELECT COUNT(*) FROM users") as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0

        if count == 0:
            logger.info("Seeding default demo users into users.db")
            for u in get_default_users():
                user = User(**u)
                plain = _DEFAULT_PASSWORDS.get(user.username, "changeme")
                hashed = pwd_context.hash(plain)
                await db.execute(
                    """
                    INSERT OR IGNORE INTO users
                    (id, username, email, full_name, department, role, is_active, password_hash, created_at, last_login)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user.id,
                        user.username,
                        user.email,
                        user.full_name,
                        user.department,
                        user.role.value,
                        1 if user.is_active else 0,
                        hashed,
                        user.created_at.isoformat(),
                        None,
                    ),
                )
            await db.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_user(row: aiosqlite.Row) -> User:
        """Convert an aiosqlite.Row to a User model."""
        d = dict(row)
        return User(
            id=d["id"],
            username=d["username"],
            email=d["email"] or "",
            full_name=d["full_name"] or "",
            department=d["department"] or "",
            role=Role(d["role"]) if d["role"] else Role.USER,
            is_active=bool(d["is_active"]),
            created_at=datetime.fromisoformat(d["created_at"]) if d["created_at"] else datetime.now(timezone.utc),
            last_login=datetime.fromisoformat(d["last_login"]) if d.get("last_login") else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def authenticate(self, username: str, password: str) -> User | None:
        """Verify credentials. Returns ``User`` on success, ``None`` on failure."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        hashed = dict(row).get("password_hash", "")
        if not hashed or not pwd_context.verify(password, hashed):
            return None

        user = self._row_to_user(row)
        if not user.is_active:
            return None

        # Update last_login
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (now, user.id),
            )
            await db.commit()

        user.last_login = datetime.now(timezone.utc)
        return user

    async def get_by_username(self, username: str) -> User | None:
        """Look up a user by username."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None
        return self._row_to_user(row)

    async def get_by_id(self, user_id: str) -> User | None:
        """Look up a user by their unique ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None
        return self._row_to_user(row)

    async def list_users(self) -> list[User]:
        """Return all users."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM users ORDER BY username") as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_user(row) for row in rows]

    async def update_role(self, user_id: str, new_role: Role) -> User | None:
        """Update a user's role. Returns the updated user or ``None`` if not found."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE users SET role = ? WHERE id = ?",
                (new_role.value, user_id),
            )
            await db.commit()
            if cursor.rowcount == 0:
                return None

        return await self.get_by_id(user_id)

    async def set_active(self, user_id: str, active: bool) -> User | None:
        """Activate or deactivate a user. Returns the updated user or ``None``."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE users SET is_active = ? WHERE id = ?",
                (1 if active else 0, user_id),
            )
            await db.commit()
            if cursor.rowcount == 0:
                return None

        return await self.get_by_id(user_id)

    async def create_user(self, user: User, password: str | None = None) -> User:
        """Insert a new user into the database.

        Used for LDAP auto-creation and HR sync.  If *password* is ``None``
        the password_hash is set to an empty string (external auth only).
        """
        await self._ensure_initialized()

        hashed = pwd_context.hash(password) if password else ""

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO users
                (id, username, email, full_name, department, role, is_active, password_hash, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.id,
                    user.username,
                    user.email,
                    user.full_name,
                    user.department,
                    user.role.value,
                    1 if user.is_active else 0,
                    hashed,
                    user.created_at.isoformat(),
                    None,
                ),
            )
            await db.commit()

        return user

    async def update_department(self, user_id: str, department: str) -> None:
        """Update a user's department field."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET department = ? WHERE id = ?",
                (department, user_id),
            )
            await db.commit()


# Singleton factory
get_user_store = create_async_singleton(UserStore)
