"""Authentication and authorization module for flux-rag."""

from auth.models import Permission, Role, Token, User
from auth.jwt_handler import create_access_token, create_refresh_token, verify_token, verify_token_with_blacklist
from auth.dependencies import get_current_user, require_permission, require_role
from auth.audit import AuditAction, AuditLogger

__all__ = [
    "User",
    "Role",
    "Permission",
    "Token",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "verify_token_with_blacklist",
    "get_current_user",
    "require_role",
    "require_permission",
    "AuditLogger",
    "AuditAction",
]
