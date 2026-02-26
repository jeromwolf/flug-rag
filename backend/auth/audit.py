"""Audit logging for authentication and authorization events.

Events are persisted to a SQLite database (``data/audit.db`` by default) for
later review by administrators.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import aiosqlite

from config.settings import settings
from core.db.base import AsyncSQLiteManager, create_async_singleton

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Recognised audit event types."""

    LOGIN = "LOGIN"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    TOKEN_REFRESH = "TOKEN_REFRESH"
    ACCESS_DENIED = "ACCESS_DENIED"
    ROLE_CHANGE = "ROLE_CHANGE"
    DOCUMENT_ACCESS = "DOCUMENT_ACCESS"
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    DOCUMENT_DELETE = "DOCUMENT_DELETE"
    SETTINGS_CHANGE = "SETTINGS_CHANGE"
    USER_CREATED = "USER_CREATED"
    USER_DEACTIVATED = "USER_DEACTIVATED"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"


class AuditLogger(AsyncSQLiteManager):
    """Async audit logger backed by SQLite."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            data_dir = Path(settings.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "audit.db")
        super().__init__(Path(db_path))

    async def _create_tables(self, db: aiosqlite.Connection):
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                user_id     TEXT,
                username    TEXT,
                action      TEXT NOT NULL,
                resource    TEXT,
                details     TEXT,
                ip_address  TEXT
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log (user_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log (action)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (timestamp)"
        )
        await db.commit()

    async def log_event(
        self,
        user_id: str | None,
        username: str | None,
        action: AuditAction | str,
        resource: str = "",
        details: str = "",
        ip_address: str = "",
    ) -> None:
        """Record an audit event (async)."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        action_str = action.value if isinstance(action, AuditAction) else str(action)

        try:
            async with self.get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO audit_log (id, timestamp, user_id, username, action, resource, details, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, timestamp, user_id, username, action_str, resource, details, ip_address),
                )
                await db.commit()
        except Exception:
            logger.exception("Failed to write audit event %s", action_str)

    async def get_events(
        self,
        limit: int = 100,
        user_id: str | None = None,
        action: str | None = None,
    ) -> list[dict]:
        """Retrieve recent audit events with optional filters (async)."""
        try:
            async with self.get_connection() as db:
                db.row_factory = aiosqlite.Row
                query = "SELECT * FROM audit_log WHERE 1=1"
                params: list = []
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                if action:
                    query += " AND action = ?"
                    params.append(action)
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to read audit events")
            return []


# Async singleton factory
get_audit_logger = create_async_singleton(AuditLogger)


# Backward-compatible sync-looking wrapper that schedules in event loop
class _AuditLoggerProxy:
    """Proxy that makes audit_logger.log_event() work from sync contexts
    by scheduling the async call via fire-and-forget."""

    def log_event(self, **kwargs) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_log_event(**kwargs))
        except RuntimeError:
            # No running loop - skip audit (shouldn't happen in FastAPI)
            logger.warning("No event loop for audit logging")

    async def _async_log_event(self, **kwargs) -> None:
        instance = await get_audit_logger()
        await instance.log_event(**kwargs)

    def get_events(self, **kwargs) -> list[dict]:
        """Sync get_events - callers should migrate to async."""
        try:
            asyncio.get_running_loop()
            # Can't await in sync context, return empty
            logger.warning("Sync get_events called; migrate caller to async")
            return []
        except RuntimeError:
            return []


# Module-level backward-compatible singleton
audit_logger = _AuditLoggerProxy()
