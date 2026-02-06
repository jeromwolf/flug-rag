"""Audit logging for authentication and authorization events.

Events are persisted to a SQLite database (``data/audit.db`` by default) for
later review by administrators.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from config.settings import settings

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


class AuditLogger:
    """Lightweight synchronous audit logger backed by SQLite."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            data_dir = Path(settings.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "audit.db")
        self._db_path = db_path
        self._ensure_table()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log (action)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (timestamp)"
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to initialise audit database")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(
        self,
        user_id: str | None,
        username: str | None,
        action: AuditAction | str,
        resource: str = "",
        details: str = "",
        ip_address: str = "",
    ) -> None:
        """Record an audit event."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        action_str = action.value if isinstance(action, AuditAction) else str(action)

        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """
                INSERT INTO audit_log (id, timestamp, user_id, username, action, resource, details, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, timestamp, user_id, username, action_str, resource, details, ip_address),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("Failed to write audit event %s", action_str)

    def get_events(
        self,
        limit: int = 100,
        user_id: str | None = None,
        action: str | None = None,
    ) -> list[dict]:
        """Retrieve recent audit events with optional filters."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
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

            rows = conn.execute(query, params).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to read audit events")
            return []


# Module-level singleton
audit_logger = AuditLogger()
