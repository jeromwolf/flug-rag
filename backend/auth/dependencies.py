"""FastAPI dependency functions for authentication and authorization."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from auth.jwt_handler import verify_token, verify_token_with_blacklist
from auth.models import ROLE_PERMISSIONS, Role, User
from auth.user_store import get_user_store
from config.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> User | None:
    """Return the current authenticated user, or ``None`` when auth is disabled.

    When ``settings.auth_enabled`` is ``False`` a synthetic admin user is
    returned so that all routes remain accessible during development.
    """
    if not settings.auth_enabled:
        # Auth disabled -- return a synthetic admin for dev/demo
        return User(
            id="dev-admin",
            username="dev",
            email="dev@localhost",
            full_name="Developer (auth disabled)",
            department="Development",
            role=Role.ADMIN,
        )

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = await verify_token_with_blacklist(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user_store = await get_user_store()
    user = await user_store.get_by_username(username)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


async def require_password_changed(
    current_user: Annotated[User | None, Depends(get_current_user)],
) -> User:
    """Ensure user has changed their default password.

    Blocks API access until the user calls ``/auth/change-password``.
    When ``auth_enabled`` is ``False`` the check is skipped.
    """
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    if not settings.auth_enabled:
        return current_user

    user_store = await get_user_store()
    must_change = await user_store.get_must_change_password(current_user.id)

    if must_change:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change required. Please change your default password before accessing this resource.",
            headers={"X-Password-Change-Required": "true"},
        )

    return current_user


def require_role(allowed_roles: list[Role]):
    """Return a dependency that ensures the user holds one of *allowed_roles*."""

    async def _check(
        current_user: Annotated[User | None, Depends(get_current_user)],
    ) -> User:
        if current_user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        if not current_user.has_role(allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {[r.value for r in allowed_roles]}",
            )
        return current_user

    return _check


def require_permission(permission: str):
    """Return a dependency that ensures the user has *permission*."""

    async def _check(
        current_user: Annotated[User | None, Depends(get_current_user)],
    ) -> User:
        if current_user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission}",
            )
        return current_user

    return _check
