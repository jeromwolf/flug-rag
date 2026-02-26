"""SQLite-based conversation memory and session management."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from config.settings import settings
from core.db.base import AsyncSQLiteManager


class ConversationMemory(AsyncSQLiteManager):
    """Manages conversation history with SQLite storage."""

    def __init__(self, db_path: str | None = None):
        path = Path(db_path) if db_path else settings.data_dir / "memory.db"
        super().__init__(path)

    async def _create_tables(self, db: aiosqlite.Connection) -> None:
        """Create sessions and messages tables."""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at)
        """)
        await db.commit()

    async def create_session(self, title: str = "", metadata: dict | None = None) -> str:
        """Create a new conversation session. Returns session_id."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with self.get_connection() as db:
            await db.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, title, now, now, json.dumps(metadata or {})),
            )
            await db.commit()

        return session_id

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """Add a message to a session. Returns message_id."""
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with self.get_connection() as db:
            await db.execute(
                "INSERT INTO messages (id, session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, session_id, role, content, json.dumps(metadata or {}), now),
            )
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            await db.commit()

        return msg_id

    async def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Get conversation history for a session."""
        max_msgs = limit or settings.max_history * 2

        async with self.get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT id, role, content, metadata, created_at
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, max_msgs),
            )
            rows = await cursor.fetchall()

        messages = []
        for row in reversed(rows):
            messages.append({
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"],
            })

        return messages

    async def get_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get recent sessions."""
        async with self.get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at, s.metadata,
                          COUNT(m.id) as message_count
                   FROM sessions s
                   LEFT JOIN messages m ON m.session_id = s.id
                   GROUP BY s.id
                   ORDER BY s.updated_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            )
            rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    async def get_session(self, session_id: str) -> dict | None:
        """Get a single session by ID."""
        async with self.get_connection() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, title, created_at, updated_at, metadata FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }

    async def update_session_title(self, session_id: str, title: str):
        """Update session title."""
        async with self.get_connection() as db:
            await db.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.now(timezone.utc).isoformat(), session_id),
            )
            await db.commit()

    async def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        async with self.get_connection() as db:
            await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()

    async def count_sessions(self) -> int:
        """Get total session count."""
        async with self.get_connection() as db:
            cursor = await db.execute("SELECT COUNT(*) FROM sessions")
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def clear_all(self):
        """Delete all sessions and messages."""
        async with self.get_connection() as db:
            await db.execute("DELETE FROM messages")
            await db.execute("DELETE FROM sessions")
            await db.commit()


# Singleton factory
_shared_memory: ConversationMemory | None = None


def get_memory() -> ConversationMemory:
    """Get the shared ConversationMemory singleton."""
    global _shared_memory
    if _shared_memory is None:
        _shared_memory = ConversationMemory()
    return _shared_memory
