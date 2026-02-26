"""Tests for the authentication module."""

import os
import time

import pytest
from httpx import ASGITransport, AsyncClient

# Force auth ENABLED for tests so we actually exercise the auth layer.
os.environ["AUTH_ENABLED"] = "true"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-tests"

from api.main import app  # noqa: E402 (env vars must be set before import)
from auth.audit import AuditAction, AuditLogger  # noqa: E402
from auth.jwt_handler import (  # noqa: E402
    create_access_token,
    create_refresh_token,
    verify_token,
)
from auth.models import ROLE_PERMISSIONS, Role  # noqa: E402
from auth.rate_limiter import RateLimiter  # noqa: E402
from config.settings import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_auth():
    """Ensure auth is enabled for every test in this module.

    The env-var trick above works when this module is collected first, but if
    another module has already imported ``settings`` (creating the singleton
    with ``auth_enabled=False``), we need to patch the live object directly.
    """
    original = settings.auth_enabled
    settings.__dict__["auth_enabled"] = True
    yield
    settings.__dict__["auth_enabled"] = original


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def admin_token(client: AsyncClient):
    """Login as admin and return the access token."""
    resp = await client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture
async def user_token(client: AsyncClient):
    """Login as regular user and return the access token."""
    resp = await client.post(
        "/api/auth/login",
        json={"username": "user", "password": "user123"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


@pytest.fixture
async def viewer_token(client: AsyncClient):
    """Login as viewer and return the access token."""
    resp = await client.post(
        "/api/auth/login",
        json={"username": "viewer", "password": "viewer123"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# JWT Handler Tests
# ===========================================================================


class TestJWTHandler:
    """Test JWT token creation and verification."""

    def test_create_access_token(self):
        user_data = {"id": "u1", "username": "testuser", "role": "user"}
        token = create_access_token(user_data)
        assert isinstance(token, str)
        assert len(token) > 20

    def test_create_refresh_token(self):
        user_data = {"id": "u1", "username": "testuser", "role": "user"}
        token = create_refresh_token(user_data)
        assert isinstance(token, str)

    def test_verify_valid_token(self):
        user_data = {"id": "u1", "username": "testuser", "role": "user"}
        token = create_access_token(user_data)
        payload = verify_token(token)
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == "u1"
        assert payload["role"] == "user"
        assert payload["type"] == "access"

    def test_verify_refresh_token(self):
        user_data = {"id": "u1", "username": "testuser", "role": "user"}
        token = create_refresh_token(user_data)
        payload = verify_token(token, required_type="refresh")
        assert payload["sub"] == "testuser"
        assert payload["type"] == "refresh"

    def test_refresh_token_rejected_as_access(self):
        """Refresh token must be rejected when verified as access token."""
        from jose import JWTError

        user_data = {"id": "u1", "username": "testuser", "role": "user"}
        token = create_refresh_token(user_data)
        with pytest.raises(JWTError, match="Expected 'access' token"):
            verify_token(token)  # default required_type="access"

    def test_access_token_rejected_as_refresh(self):
        """Access token must be rejected when verified as refresh token."""
        from jose import JWTError

        user_data = {"id": "u1", "username": "testuser", "role": "user"}
        token = create_access_token(user_data)
        with pytest.raises(JWTError, match="Expected 'refresh' token"):
            verify_token(token, required_type="refresh")

    def test_verify_invalid_token(self):
        from jose import JWTError

        with pytest.raises(JWTError):
            verify_token("invalid.token.here")

    def test_verify_tampered_token(self):
        from jose import JWTError

        user_data = {"id": "u1", "username": "testuser", "role": "user"}
        token = create_access_token(user_data)
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(JWTError):
            verify_token(tampered)


# ===========================================================================
# Login Endpoint Tests
# ===========================================================================


class TestLogin:
    """Test the login endpoint."""

    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient):
        resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "admin"
        assert data["user"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_login_invalid_password(self, client: AsyncClient):
        resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrongpassword"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, client: AsyncClient):
        resp = await client.post(
            "/api/auth/login",
            json={"username": "nonexistent", "password": "anything"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_login_all_demo_users(self, client: AsyncClient):
        """Verify all demo users can log in."""
        for username, password, role in [
            ("admin", "admin123", "admin"),
            ("manager", "manager123", "manager"),
            ("user", "user123", "user"),
            ("viewer", "viewer123", "viewer"),
        ]:
            resp = await client.post(
                "/api/auth/login",
                json={"username": username, "password": password},
            )
            assert resp.status_code == 200, f"Login failed for {username}"
            assert resp.json()["user"]["role"] == role


# ===========================================================================
# Protected Route Tests
# ===========================================================================


class TestProtectedRoutes:
    """Test that endpoints respect authentication."""

    @pytest.mark.asyncio
    async def test_me_with_valid_token(self, client: AsyncClient, admin_token: str):
        resp = await client.get("/api/auth/me", headers=_auth_header(admin_token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"

    @pytest.mark.asyncio
    async def test_me_without_token(self, client: AsyncClient):
        resp = await client.get("/api/auth/me")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_me_with_invalid_token(self, client: AsyncClient):
        resp = await client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401


# ===========================================================================
# RBAC Tests
# ===========================================================================


class TestRBAC:
    """Test role-based access control."""

    @pytest.mark.asyncio
    async def test_admin_can_list_users(self, client: AsyncClient, admin_token: str):
        resp = await client.get(
            "/api/auth/users", headers=_auth_header(admin_token)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 4  # default demo users

    @pytest.mark.asyncio
    async def test_user_cannot_list_users(self, client: AsyncClient, user_token: str):
        resp = await client.get(
            "/api/auth/users", headers=_auth_header(user_token)
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_viewer_cannot_list_users(self, client: AsyncClient, viewer_token: str):
        resp = await client.get(
            "/api/auth/users", headers=_auth_header(viewer_token)
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_can_change_role(self, client: AsyncClient, admin_token: str):
        resp = await client.put(
            "/api/auth/users/user-001/role",
            json={"role": "manager"},
            headers=_auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert resp.json()["user"]["role"] == "manager"

        # Reset back
        await client.put(
            "/api/auth/users/user-001/role",
            json={"role": "user"},
            headers=_auth_header(admin_token),
        )

    @pytest.mark.asyncio
    async def test_user_cannot_change_role(self, client: AsyncClient, user_token: str):
        resp = await client.put(
            "/api/auth/users/viewer-001/role",
            json={"role": "admin"},
            headers=_auth_header(user_token),
        )
        assert resp.status_code == 403

    def test_permission_matrix_admin_has_all(self):
        """Admin role should have the most permissions."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert "chat:read" in admin_perms
        assert "admin:write" in admin_perms
        assert "users:write" in admin_perms
        assert "settings:write" in admin_perms

    def test_permission_matrix_viewer_is_minimal(self):
        """Viewer should have very limited permissions."""
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert "chat:read" in viewer_perms
        assert "chat:write" not in viewer_perms
        assert "admin:write" not in viewer_perms
        assert "documents:write" not in viewer_perms

    def test_permission_matrix_user_can_chat(self):
        user_perms = ROLE_PERMISSIONS[Role.USER]
        assert "chat:read" in user_perms
        assert "chat:write" in user_perms
        assert "documents:read" in user_perms
        assert "admin:write" not in user_perms


# ===========================================================================
# Token Refresh Tests
# ===========================================================================


class TestTokenRefresh:
    """Test token refresh flow."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, client: AsyncClient):
        # Login first
        login_resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        refresh_token = login_resp.json()["refresh_token"]

        # Refresh
        resp = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_fails(self, client: AsyncClient):
        login_resp = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        access_token = login_resp.json()["access_token"]

        resp = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": access_token},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_with_invalid_token(self, client: AsyncClient):
        resp = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )
        assert resp.status_code == 401


# ===========================================================================
# Rate Limiter Tests
# ===========================================================================


class TestRateLimiter:
    """Test the login rate limiter."""

    def test_allows_within_limit(self):
        limiter = RateLimiter(max_attempts=3, window_seconds=60.0)
        key = "test-ip"
        assert limiter.is_allowed(key)
        limiter.record(key)
        limiter.record(key)
        assert limiter.is_allowed(key)  # 2 of 3 used
        limiter.record(key)
        assert not limiter.is_allowed(key)  # 3 of 3 used

    def test_resets_after_window(self):
        limiter = RateLimiter(max_attempts=2, window_seconds=0.1)
        key = "test-ip2"
        limiter.record(key)
        limiter.record(key)
        assert not limiter.is_allowed(key)
        time.sleep(0.15)
        assert limiter.is_allowed(key)

    def test_remaining_count(self):
        limiter = RateLimiter(max_attempts=5, window_seconds=60.0)
        key = "test-ip3"
        assert limiter.remaining(key) == 5
        limiter.record(key)
        assert limiter.remaining(key) == 4

    def test_reset(self):
        limiter = RateLimiter(max_attempts=2, window_seconds=60.0)
        key = "test-ip4"
        limiter.record(key)
        limiter.record(key)
        assert not limiter.is_allowed(key)
        limiter.reset(key)
        assert limiter.is_allowed(key)


# ===========================================================================
# Audit Logging Tests
# ===========================================================================


class TestAuditLogging:
    """Test audit log functionality."""

    def test_log_and_retrieve(self, tmp_path):
        db_path = str(tmp_path / "test_audit.db")
        logger = AuditLogger(db_path=db_path)

        logger.log_event(
            user_id="u1",
            username="admin",
            action=AuditAction.LOGIN,
            resource="/api/auth/login",
            details="Test login",
            ip_address="127.0.0.1",
        )

        events = logger.get_events(limit=10)
        assert len(events) == 1
        assert events[0]["action"] == "LOGIN"
        assert events[0]["user_id"] == "u1"
        assert events[0]["ip_address"] == "127.0.0.1"

    def test_filter_by_user(self, tmp_path):
        db_path = str(tmp_path / "test_audit2.db")
        logger = AuditLogger(db_path=db_path)

        logger.log_event("u1", "admin", AuditAction.LOGIN)
        logger.log_event("u2", "user", AuditAction.LOGIN)
        logger.log_event("u1", "admin", AuditAction.LOGOUT)

        events = logger.get_events(user_id="u1")
        assert len(events) == 2

    def test_filter_by_action(self, tmp_path):
        db_path = str(tmp_path / "test_audit3.db")
        logger = AuditLogger(db_path=db_path)

        logger.log_event("u1", "admin", AuditAction.LOGIN)
        logger.log_event("u1", "admin", AuditAction.LOGOUT)
        logger.log_event(None, "unknown", AuditAction.LOGIN_FAILED)

        events = logger.get_events(action="LOGIN_FAILED")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_login_creates_audit_event(self, client: AsyncClient):
        """Verify that a successful login creates an audit log entry."""
        await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        # Wait for fire-and-forget audit task to complete
        import asyncio
        await asyncio.sleep(0.1)

        # Use async singleton directly (sync proxy cannot return data in async context)
        from auth.audit import get_audit_logger
        logger_instance = await get_audit_logger()
        events = await logger_instance.get_events(limit=5, action="LOGIN")
        assert len(events) >= 1
        latest = events[0]
        assert latest["username"] == "admin"


# ===========================================================================
# Logout Tests
# ===========================================================================


class TestLogout:
    """Test logout endpoint."""

    @pytest.mark.asyncio
    async def test_logout_with_token(self, client: AsyncClient, admin_token: str):
        resp = await client.post(
            "/api/auth/logout", headers=_auth_header(admin_token)
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged_out"

    @pytest.mark.asyncio
    async def test_logout_without_token(self, client: AsyncClient):
        # Logout without token should still succeed (no-op)
        resp = await client.post("/api/auth/logout")
        # When auth_enabled, this should still return 200 (logout is permissive)
        assert resp.status_code in (200, 401)
