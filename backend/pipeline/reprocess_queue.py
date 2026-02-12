"""
Reprocessing Queue for Failed Document Ingestion (SFR-008)

Manages a queue of failed document ingestion attempts with retry logic.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite


@dataclass
class QueueItem:
    """Represents a single item in the reprocessing queue."""

    id: str
    document_id: str
    filename: str
    file_path: str
    error_message: str
    status: str
    retry_count: int
    max_retries: int
    created_at: str
    updated_at: str
    completed_at: Optional[str]


@dataclass
class QueueStats:
    """Aggregated statistics for the reprocessing queue."""

    total: int
    pending: int
    processing: int
    completed: int
    failed: int


class ReprocessQueue:
    """Manages reprocessing queue for failed document ingestion."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False

    async def init_db(self) -> None:
        """Initialize database and create table if not exists."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS reprocess_queue (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            await db.commit()

        self._initialized = True

    async def enqueue(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        error_message: str,
        max_retries: int = 3,
    ) -> str:
        """
        Add a failed document to the reprocessing queue.

        Args:
            document_id: Document identifier
            filename: Original filename
            file_path: Path to document file
            error_message: Error message from failed ingestion
            max_retries: Maximum number of retry attempts

        Returns:
            Queue item ID
        """
        await self.init_db()

        queue_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO reprocess_queue
                (id, document_id, filename, file_path, error_message, status,
                 retry_count, max_retries, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (
                    queue_id,
                    document_id,
                    filename,
                    file_path,
                    error_message,
                    max_retries,
                    now,
                    now,
                ),
            )
            await db.commit()

        return queue_id

    async def get_queue(
        self, status: Optional[str] = None, limit: int = 50
    ) -> list[QueueItem]:
        """
        Get queue items, optionally filtered by status.

        Args:
            status: Filter by status (pending/processing/completed/failed)
            limit: Maximum number of items to return

        Returns:
            List of queue items
        """
        await self.init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if status:
                cursor = await db.execute(
                    """
                    SELECT * FROM reprocess_queue
                    WHERE status = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT * FROM reprocess_queue
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )

            rows = await cursor.fetchall()
            return [
                QueueItem(
                    id=row["id"],
                    document_id=row["document_id"],
                    filename=row["filename"],
                    file_path=row["file_path"],
                    error_message=row["error_message"],
                    status=row["status"],
                    retry_count=row["retry_count"],
                    max_retries=row["max_retries"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    completed_at=row["completed_at"],
                )
                for row in rows
            ]

    async def get_stats(self) -> QueueStats:
        """
        Get aggregated statistics for the queue.

        Returns:
            Queue statistics
        """
        await self.init_db()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM reprocess_queue
                """
            )
            row = await cursor.fetchone()

            return QueueStats(
                total=row[0] or 0,
                pending=row[1] or 0,
                processing=row[2] or 0,
                completed=row[3] or 0,
                failed=row[4] or 0,
            )

    async def mark_processing(self, queue_id: str) -> None:
        """
        Mark a queue item as processing and increment retry count.

        Args:
            queue_id: Queue item ID
        """
        await self.init_db()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE reprocess_queue
                SET status = 'processing',
                    retry_count = retry_count + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, queue_id),
            )
            await db.commit()

    async def mark_completed(self, queue_id: str) -> None:
        """
        Mark a queue item as completed.

        Args:
            queue_id: Queue item ID
        """
        await self.init_db()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE reprocess_queue
                SET status = 'completed',
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, queue_id),
            )
            await db.commit()

    async def mark_failed(self, queue_id: str, error_message: str) -> None:
        """
        Mark a queue item as failed or reset to pending based on retry count.

        Args:
            queue_id: Queue item ID
            error_message: Latest error message
        """
        await self.init_db()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Check if max retries exceeded
            cursor = await db.execute(
                """
                SELECT retry_count, max_retries
                FROM reprocess_queue
                WHERE id = ?
                """,
                (queue_id,),
            )
            row = await cursor.fetchone()

            if row and row[0] >= row[1]:
                # Max retries exceeded, mark as failed
                await db.execute(
                    """
                    UPDATE reprocess_queue
                    SET status = 'failed',
                        error_message = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (error_message, now, queue_id),
                )
            else:
                # Reset to pending for retry
                await db.execute(
                    """
                    UPDATE reprocess_queue
                    SET status = 'pending',
                        error_message = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (error_message, now, queue_id),
                )

            await db.commit()

    async def retry(self, queue_id: str) -> bool:
        """
        Reset a queue item to pending for manual retry.

        Args:
            queue_id: Queue item ID

        Returns:
            True if reset successful, False if max retries exceeded
        """
        await self.init_db()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Check if max retries exceeded
            cursor = await db.execute(
                """
                SELECT retry_count, max_retries
                FROM reprocess_queue
                WHERE id = ?
                """,
                (queue_id,),
            )
            row = await cursor.fetchone()

            if not row or row[0] >= row[1]:
                return False

            await db.execute(
                """
                UPDATE reprocess_queue
                SET status = 'pending',
                    updated_at = ?
                WHERE id = ?
                """,
                (now, queue_id),
            )
            await db.commit()

        return True

    async def retry_all_failed(self) -> None:
        """Reset all failed items to pending status (only if retries remain)."""
        await self.init_db()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE reprocess_queue
                SET status = 'pending',
                    updated_at = ?
                WHERE status = 'failed'
                    AND retry_count < max_retries
                """,
                (now,),
            )
            await db.commit()

    async def delete(self, queue_id: str) -> None:
        """
        Remove a queue item.

        Args:
            queue_id: Queue item ID
        """
        await self.init_db()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM reprocess_queue WHERE id = ?",
                (queue_id,),
            )
            await db.commit()

    async def process_next(self) -> Optional[QueueItem]:
        """
        Get the oldest pending item and mark it as processing atomically.

        Returns:
            Next queue item to process, or None if queue is empty
        """
        await self.init_db()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Use BEGIN IMMEDIATE to lock database for atomic SELECT + UPDATE
            await db.execute("BEGIN IMMEDIATE")

            try:
                # Get oldest pending item
                cursor = await db.execute(
                    """
                    SELECT * FROM reprocess_queue
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT 1
                    """
                )
                row = await cursor.fetchone()

                if not row:
                    await db.rollback()
                    return None

                queue_id = row["id"]

                # Mark as processing within same transaction
                await db.execute(
                    """
                    UPDATE reprocess_queue
                    SET status = 'processing',
                        retry_count = retry_count + 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, queue_id),
                )
                await db.commit()

                # Return queue item with updated status
                return QueueItem(
                    id=row["id"],
                    document_id=row["document_id"],
                    filename=row["filename"],
                    file_path=row["file_path"],
                    error_message=row["error_message"],
                    status="processing",  # Updated status
                    retry_count=row["retry_count"] + 1,  # Incremented
                    max_retries=row["max_retries"],
                    created_at=row["created_at"],
                    updated_at=now,  # Updated timestamp
                    completed_at=row["completed_at"],
                )

            except Exception:
                await db.rollback()
                raise


# Singleton instance
_queue: Optional[ReprocessQueue] = None


async def get_reprocess_queue() -> ReprocessQueue:
    """
    Get the singleton reprocessing queue instance.

    Returns:
        Reprocessing queue instance
    """
    global _queue

    if _queue is None:
        db_path = Path(__file__).resolve().parent.parent / "data" / "reprocess_queue.db"
        _queue = ReprocessQueue(db_path)
        await _queue.init_db()

    return _queue
