"""Authentication API routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from auth.audit import AuditAction, audit_logger
from auth.dependencies import get_current_user, require_role
from auth.jwt_handler import create_access_token, create_refresh_token, verify_token
from auth.ldap_provider import ldap_provider
from auth.models import Role, Token, User
from auth.rate_limiter import login_rate_limiter
from auth.user_store import get_user_store
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
    user_store = await get_user_store()
    user: User | None = None
    if ldap_provider.is_configured:
        ldap_info = ldap_provider.authenticate(body.username, body.password)
        if ldap_info is not None:
            # Map LDAP user to local user (create if missing)
            user = await user_store.get_by_username(body.username)
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
                await user_store.create_user(user)

    # 2. Fall back to local store
    if user is None:
        user = await user_store.authenticate(body.username, body.password)

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
    user_store = await get_user_store()
    user = await user_store.get_by_username(username) if username else None
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
    user_store = await get_user_store()
    return [_user_to_response(u) for u in await user_store.list_users()]


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

    user_store = await get_user_store()
    updated = await user_store.update_role(user_id, new_role)
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


# ===== Ethics Pledge Endpoints =====

@router.get("/auth/ethics-pledge")
async def get_ethics_pledge():
    """
    최신 윤리 서약 내용 조회
    """
    from auth.ethics import CURRENT_PLEDGE_VERSION, CURRENT_PLEDGE_CONTENT
    return {
        "version": CURRENT_PLEDGE_VERSION,
        "content": CURRENT_PLEDGE_CONTENT,
    }


@router.get("/auth/ethics-pledge/status")
async def get_pledge_status(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    현재 사용자의 서약 동의 상태 확인
    """
    from auth.ethics import get_pledge_manager, CURRENT_PLEDGE_VERSION
    manager = await get_pledge_manager()
    agreed = await manager.has_agreed(current_user.id)
    return {
        "agreed": agreed,
        "current_version": CURRENT_PLEDGE_VERSION,
        "user_id": current_user.id,
    }


@router.post("/auth/ethics-pledge/agree")
async def agree_ethics_pledge(
    current_user: Annotated[User, Depends(get_current_user)],
    request: Request,
):
    """
    윤리 서약 동의
    """
    from auth.ethics import get_pledge_manager
    manager = await get_pledge_manager()

    ip_address = request.client.host if request.client else ""
    record = await manager.agree(current_user.id, ip_address=ip_address)

    return {
        "status": "agreed",
        "version": record.version,
        "agreed_at": record.agreed_at,
    }


# ===== Access Request Endpoints =====

class AccessRequestCreate(BaseModel):
    requested_role: str = Field(..., description="요청 역할: admin, manager, user")
    reason: str = Field(..., min_length=10, description="신청 사유 (최소 10자)")


class AccessRequestReview(BaseModel):
    decision: str = Field(..., description="approved 또는 rejected")
    comment: str = Field("", description="검토 의견")


@router.post("/auth/access-request")
async def create_access_request(
    body: AccessRequestCreate,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    권한 신청 - 인증된 사용자가 역할 변경 요청
    """
    from auth.access_request import get_request_manager
    manager = await get_request_manager()

    try:
        req = await manager.create_request(
            user_id=current_user.id,
            username=current_user.username,
            current_role=current_user.role.value,
            requested_role=body.requested_role,
            reason=body.reason,
        )
        return {
            "status": "created",
            "request_id": req.id,
            "requested_role": req.requested_role,
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/auth/access-request/my")
async def get_my_access_requests(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    내 권한 신청 이력 조회
    """
    from auth.access_request import get_request_manager
    manager = await get_request_manager()
    requests = await manager.get_user_requests(current_user.id)

    return {
        "requests": [
            {
                "id": r.id,
                "requested_role": r.requested_role,
                "reason": r.reason,
                "status": r.status,
                "created_at": r.created_at,
                "reviewer_comment": r.reviewer_comment,
                "reviewed_at": r.reviewed_at,
            }
            for r in requests
        ]
    }


@router.get("/auth/admin/access-requests")
async def get_pending_access_requests(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """
    관리자: 대기 중인 권한 신청 목록
    """
    from auth.access_request import get_request_manager
    manager = await get_request_manager()
    requests = await manager.get_pending_requests()

    return {
        "requests": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "username": r.username,
                "current_role": r.current_role,
                "requested_role": r.requested_role,
                "reason": r.reason,
                "status": r.status,
                "created_at": r.created_at,
            }
            for r in requests
        ],
        "total": len(requests),
    }


@router.put("/auth/admin/access-requests/{request_id}")
async def review_access_request(
    request_id: str,
    body: AccessRequestReview,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """
    관리자: 권한 신청 승인/거절
    """
    from auth.access_request import get_request_manager

    if body.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")

    manager = await get_request_manager()
    try:
        req = await manager.review_request(
            request_id=request_id,
            reviewer_id=current_user.id,
            decision=body.decision,
            comment=body.comment,
        )
        return {
            "status": req.status,
            "request_id": req.id,
            "reviewer_comment": req.reviewer_comment,
            "reviewed_at": req.reviewed_at,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
