"""JWT token creation and verification using python-jose."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from config.settings import settings


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(
    user_data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a short-lived access token.

    *user_data* is embedded in the JWT payload under the ``sub`` and custom
    claims.  If *expires_delta* is ``None`` the configured default is used.
    """
    expire = _now_utc() + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    to_encode: dict[str, Any] = {
        "sub": user_data.get("username", user_data.get("sub", "")),
        "user_id": user_data.get("id", ""),
        "role": user_data.get("role", "user"),
        "exp": expire,
        "iat": _now_utc(),
        "type": "access",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    user_data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a long-lived refresh token."""
    expire = _now_utc() + (
        expires_delta
        if expires_delta is not None
        else timedelta(days=settings.jwt_refresh_token_expire_days)
    )
    to_encode: dict[str, Any] = {
        "sub": user_data.get("username", user_data.get("sub", "")),
        "user_id": user_data.get("id", ""),
        "exp": expire,
        "iat": _now_utc(),
        "type": "refresh",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str, required_type: str = "access") -> dict[str, Any]:
    """Decode and verify a JWT.

    Returns the decoded payload dict on success.
    Raises ``JWTError`` on failure (expired, tampered, wrong type, etc.).
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        username: str | None = payload.get("sub")
        if username is None:
            raise JWTError("Token missing 'sub' claim")
        token_type = payload.get("type")
        if token_type != required_type:
            raise JWTError(f"Expected '{required_type}' token, got '{token_type}'")
        return payload
    except JWTError:
        raise


async def verify_token_with_blacklist(
    token: str,
    required_type: str = "access",
) -> dict[str, Any]:
    """Decode, verify, and check the blacklist for a JWT.

    This is the async counterpart of :func:`verify_token` that additionally
    rejects tokens whose ``jti`` has been revoked.
    """
    payload = verify_token(token, required_type)
    jti = payload.get("jti")
    if jti:
        from auth.token_blacklist import get_token_blacklist

        blacklist = await get_token_blacklist()
        if await blacklist.is_revoked(jti):
            raise JWTError("Token has been revoked")
    return payload


def get_token_expiry(token: str) -> datetime | None:
    """Return the ``exp`` datetime from *token* without full verification.

    Returns ``None`` if the token cannot be decoded or lacks an ``exp`` claim.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False},
        )
        exp = payload.get("exp")
        if exp is not None:
            return datetime.fromtimestamp(exp, tz=timezone.utc)
        return None
    except JWTError:
        return None
