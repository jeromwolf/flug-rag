"""Authentication API routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from auth.audit import AuditAction, audit_logger
from auth.dependencies import get_current_user, require_role
from auth.jwt_handler import create_access_token, create_refresh_token, verify_token
from auth.ldap_provider import ldap_provider
from auth.models import Role, Token, User
from auth.rate_limiter import login_rate_limiter
from auth.user_store import user_store
from config.settings import settings

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: str
    user: UserResponse


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    department: str
    role: str
    is_active: bool


class RefreshRequest(BaseModel):
    refresh_token: str


class RoleUpdateRequest(BaseModel):
    role: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_to_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        department=user.department,
        role=user.role.value,
        is_active=user.is_active,
    )


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _build_token_pair(user: User) -> dict:
    user_data = {
        "id": user.id,
        "username": user.username,
        "role": user.role.value,
    }
    access = create_access_token(user_data)
    refresh = create_refresh_token(user_data)
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_at": expires_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest, request: Request):
    """Authenticate with username and password. Returns JWT token pair."""
    client_ip = _get_client_ip(request)
    rate_key = f"{client_ip}:{body.username}"

    # Rate limiting
    if not login_rate_limiter.is_allowed(rate_key):
        audit_logger.log_event(
            user_id=None,
            username=body.username,
            action=AuditAction.LOGIN_FAILED,
            resource="/api/auth/login",
            details="Rate limited",
            ip_address=client_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
        )

    login_rate_limiter.record(rate_key)

    # 1. Try LDAP first (if configured)
    user: User | None = None
    if ldap_provider.is_configured:
        ldap_info = ldap_provider.authenticate(body.username, body.password)
        if ldap_info is not None:
            # Map LDAP user to local user (create if missing)
            user = user_store.get_by_username(body.username)
            if user is None:
                # Auto-create from LDAP info with default USER role
                user = User(
                    id=f"ldap-{body.username}",
                    username=body.username,
                    email=ldap_info.email,
                    full_name=ldap_info.full_name,
                    department=ldap_info.department,
                    role=Role.USER,
                )

    # 2. Fall back to local store
    if user is None:
        user = user_store.authenticate(body.username, body.password)

    if user is None:
        audit_logger.log_event(
            user_id=None,
            username=body.username,
            action=AuditAction.LOGIN_FAILED,
            resource="/api/auth/login",
            details="Invalid credentials",
            ip_address=client_ip,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Success -- build tokens
    tokens = _build_token_pair(user)

    audit_logger.log_event(
        user_id=user.id,
        username=user.username,
        action=AuditAction.LOGIN,
        resource="/api/auth/login",
        details="Login successful",
        ip_address=client_ip,
    )

    # Reset rate limiter on success
    login_rate_limiter.reset(rate_key)

    return LoginResponse(
        **tokens,
        user=_user_to_response(user),
    )


@router.post("/auth/refresh")
async def refresh_token(body: RefreshRequest, request: Request):
    """Exchange a valid refresh token for a new access + refresh pair."""
    client_ip = _get_client_ip(request)
    try:
        payload = verify_token(body.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not a refresh token",
        )

    username = payload.get("sub")
    user = user_store.get_by_username(username) if username else None
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    tokens = _build_token_pair(user)

    audit_logger.log_event(
        user_id=user.id,
        username=user.username,
        action=AuditAction.TOKEN_REFRESH,
        resource="/api/auth/refresh",
        ip_address=client_ip,
    )

    return {**tokens, "user": _user_to_response(user).model_dump()}


@router.post("/auth/logout")
async def logout(
    request: Request,
    current_user: Annotated[User | None, Depends(get_current_user)] = None,
):
    """Logout (server-side audit only; client should discard tokens)."""
    client_ip = _get_client_ip(request)
    if current_user:
        audit_logger.log_event(
            user_id=current_user.id,
            username=current_user.username,
            action=AuditAction.LOGOUT,
            resource="/api/auth/logout",
            ip_address=client_ip,
        )
    return {"status": "logged_out"}


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Return the currently authenticated user's profile."""
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return _user_to_response(current_user)


@router.get("/auth/users", response_model=list[UserResponse])
async def list_users(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """List all users (admin only)."""
    return [_user_to_response(u) for u in user_store.list_users()]


@router.put("/auth/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    body: RoleUpdateRequest,
    request: Request,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """Change a user's role (admin only)."""
    try:
        new_role = Role(body.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {body.role}. Valid: {[r.value for r in Role]}",
        )

    updated = user_store.update_role(user_id, new_role)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    audit_logger.log_event(
        user_id=current_user.id,
        username=current_user.username,
        action=AuditAction.ROLE_CHANGE,
        resource=f"/api/auth/users/{user_id}/role",
        details=f"Changed role to {new_role.value}",
        ip_address=_get_client_ip(request),
    )

    return {"status": "updated", "user": _user_to_response(updated).model_dump()}


@router.get("/auth/audit")
async def get_audit_logs(
    limit: int = 100,
    user_id: str | None = None,
    action: str | None = None,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))] = None,
):
    """Retrieve audit log entries (admin only)."""
    events = audit_logger.get_events(limit=limit, user_id=user_id, action=action)
    return {"events": events, "total": len(events)}
