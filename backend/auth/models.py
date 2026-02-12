"""User, Role, Permission, and Token models for authentication."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    """User roles with hierarchical access levels."""

    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    VIEWER = "viewer"


# ---------------------------------------------------------------------------
# Permission matrix: role -> set of allowed permission strings
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {
        "chat:read",
        "chat:write",
        "documents:read",
        "documents:write",
        "documents:delete",
        "admin:read",
        "admin:write",
        "monitor:read",
        "users:read",
        "users:write",
        "feedback:read",
        "feedback:write",
        "sessions:read",
        "sessions:write",
        "sessions:delete",
        "mcp:read",
        "mcp:execute",
        "workflows:read",
        "workflows:execute",
        "agent-builder:read",
        "agent-builder:write",
        "settings:read",
        "settings:write",
    },
    Role.MANAGER: {
        "chat:read",
        "chat:write",
        "documents:read",
        "documents:write",
        "admin:read",
        "monitor:read",
        "feedback:read",
        "feedback:write",
        "sessions:read",
        "sessions:write",
        "sessions:delete",
        "mcp:read",
        "workflows:read",
        "workflows:execute",
        "agent-builder:read",
    },
    Role.USER: {
        "chat:read",
        "chat:write",
        "documents:read",
        "sessions:read",
        "sessions:write",
        "sessions:delete",
        "feedback:write",
        "mcp:read",
        "workflows:read",
    },
    Role.VIEWER: {
        "chat:read",
        "sessions:read",
    },
}


class Permission(BaseModel):
    """A named permission that can be checked against a user's role."""

    name: str
    description: str = ""


class User(BaseModel):
    """Authenticated user representation."""

    id: str
    username: str
    email: str = ""
    full_name: str = ""
    department: str = ""
    role: Role = Role.USER
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: datetime | None = None

    def has_permission(self, permission: str) -> bool:
        """Check whether this user's role grants *permission*."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def has_role(self, roles: list[Role]) -> bool:
        """Return True if the user's role is in *roles*."""
        return self.role in roles


class Token(BaseModel):
    """JWT token pair returned on successful login."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: datetime


# ---------------------------------------------------------------------------
# Default local users (for demo / development)
# ---------------------------------------------------------------------------

_DEFAULT_USERS: list[dict] = [
    {
        "id": "admin-001",
        "username": "admin",
        "email": "admin@kogas-tech.co.kr",
        "full_name": "System Administrator",
        "department": "IT",
        "role": Role.ADMIN,
        "is_active": True,
    },
    {
        "id": "manager-001",
        "username": "manager",
        "email": "manager@kogas-tech.co.kr",
        "full_name": "Manager User",
        "department": "Engineering",
        "role": Role.MANAGER,
        "is_active": True,
    },
    {
        "id": "user-001",
        "username": "user",
        "email": "user@kogas-tech.co.kr",
        "full_name": "Regular User",
        "department": "Operations",
        "role": Role.USER,
        "is_active": True,
    },
    {
        "id": "viewer-001",
        "username": "viewer",
        "email": "viewer@kogas-tech.co.kr",
        "full_name": "Viewer User",
        "department": "Guest",
        "role": Role.VIEWER,
        "is_active": True,
    },
]


def get_default_users() -> list[dict]:
    """Return a copy of the built-in demo users."""
    return [u.copy() for u in _DEFAULT_USERS]
