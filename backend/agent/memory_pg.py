"""PostgreSQL-based conversation memory and session management."""

import json
import uuid
from datetime import datetime, timezone

import asyncpg

from config.settings import settings


class PostgresConversationMemory:
    """Manages conversation history with PostgreSQL storage."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.postgres_dsn
        self._pool = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Create connection pool and tables if they don't exist."""
        if self._initialized:
            return

        self._pool = await asyncpg.create_pool(
            dsn=self.dsn, min_size=2, max_size=10
        )

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id VARCHAR(36) PRIMARY KEY,
                    title TEXT DEFAULT '',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id VARCHAR(36) PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at)
            """)

        self._initialized = True

    async def create_session(self, title: str = "", metadata: dict | None = None) -> str:
        """Create a new conversation session. Returns session_id."""
        await self._ensure_initialized()
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, metadata) VALUES ($1, $2, $3, $4, $5)",
                session_id, title, now, now, json.dumps(metadata or {}),
            )

        return session_id

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """Add a message to a session. Returns message_id."""
        await self._ensure_initialized()
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages (id, session_id, role, content, metadata, created_at) VALUES ($1, $2, $3, $4, $5, $6)",
                msg_id, session_id, role, content, json.dumps(metadata or {}), now,
            )
            await conn.execute(
                "UPDATE sessions SET updated_at = $1 WHERE id = $2",
                now, session_id,
            )

        return msg_id

    async def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Get conversation history for a session.

        Args:
            session_id: Session to retrieve.
            limit: Max messages (most recent). Default from settings.max_history * 2.

        Returns:
            List of message dicts with role, content, metadata, created_at.
        """
        await self._ensure_initialized()
        max_msgs = limit or settings.max_history * 2

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, role, content, metadata, created_at
                   FROM messages
                   WHERE session_id = $1
                   ORDER BY created_at DESC
                   LIMIT $2""",
                session_id, max_msgs,
            )

        messages = []
        for row in reversed(rows):  # Reverse to chronological order
            messages.append({
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": row["metadata"] if row["metadata"] else {},
                "created_at": row["created_at"].isoformat(),
            })

        return messages

    async def get_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get recent sessions."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT s.id, s.title, s.created_at, s.updated_at, s.metadata,
                          COUNT(m.id) as message_count
                   FROM sessions s
                   LEFT JOIN messages m ON m.session_id = s.id
                   GROUP BY s.id
                   ORDER BY s.updated_at DESC
                   LIMIT $1 OFFSET $2""",
                limit, offset,
            )

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "metadata": row["metadata"] if row["metadata"] else {},
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    async def get_session(self, session_id: str) -> dict | None:
        """Get a single session by ID."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, title, created_at, updated_at, metadata FROM sessions WHERE id = $1",
                session_id,
            )

        if not row:
            return None

        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "metadata": row["metadata"] if row["metadata"] else {},
        }

    async def update_session_title(self, session_id: str, title: str):
        """Update session title."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET title = $1, updated_at = $2 WHERE id = $3",
                title, datetime.now(timezone.utc), session_id,
            )

    async def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM messages WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)

    async def clear_all(self):
        """Delete all sessions and messages."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM messages")
            await conn.execute("DELETE FROM sessions")

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
