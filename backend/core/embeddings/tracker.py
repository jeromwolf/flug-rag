"""Embedding processing tracker for RAG pipeline quality management (SFR-008)."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


@dataclass
class EmbeddingJob:
    """Represents an embedding job."""

    id: str
    document_id: str
    filename: str
    total_chunks: int
    success_count: int
    failure_count: int
    error_message: str | None
    status: str
    started_at: str
    completed_at: str | None


@dataclass
class EmbeddingStatus:
    """Aggregate statistics for embedding operations."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    total_chunks_processed: int
    total_chunks_failed: int
    success_rate: float  # 0.0 ~ 1.0
    recent_failures: list[EmbeddingJob]


class EmbeddingTracker:
    """Tracks embedding processing status for quality management."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path(__file__).resolve().parent.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "embedding_tracker.db")

        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def init_db(self):
        """Create table if not exists."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS embedding_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status
                ON embedding_jobs(status, created_at)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_jobs_document
                ON embedding_jobs(document_id)
            """)
            await db.commit()

        self._initialized = True

    async def start_job(
        self, document_id: str, filename: str, total_chunks: int
    ) -> str:
        """Create new job, return job_id (uuid)."""
        await self.init_db()
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO embedding_jobs
                   (id, document_id, filename, total_chunks, success_count,
                    failure_count, error_message, status, started_at, completed_at, created_at)
                   VALUES (?, ?, ?, ?, 0, 0, NULL, 'processing', ?, NULL, ?)""",
                (job_id, document_id, filename, total_chunks, now, now),
            )
            await db.commit()

        return job_id

    async def update_progress(
        self, job_id: str, success_count: int, failure_count: int
    ):
        """Update counts for a job."""
        await self.init_db()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE embedding_jobs
                   SET success_count = ?, failure_count = ?
                   WHERE id = ?""",
                (success_count, failure_count, job_id),
            )
            await db.commit()

    async def complete_job(
        self, job_id: str, status: str = "completed", error_message: str | None = None
    ):
        """Mark job done."""
        await self.init_db()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE embedding_jobs
                   SET status = ?, error_message = ?, completed_at = ?
                   WHERE id = ?""",
                (status, error_message, now, job_id),
            )
            await db.commit()

    async def get_status(self) -> EmbeddingStatus:
        """Get aggregate stats."""
        await self.init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get counts by status
            cursor = await db.execute("""
                SELECT
                    COUNT(*) as total_jobs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status IN ('pending', 'processing') THEN 1 ELSE 0 END) as pending,
                    SUM(success_count) as total_success,
                    SUM(failure_count) as total_failures
                FROM embedding_jobs
            """)
            stats_row = await cursor.fetchone()

            # Get recent failures
            cursor = await db.execute(
                """SELECT * FROM embedding_jobs
                   WHERE status = 'failed'
                   ORDER BY created_at DESC
                   LIMIT 10"""
            )
            failure_rows = await cursor.fetchall()

        recent_failures = [self._row_to_job(row) for row in failure_rows]

        total_jobs = stats_row["total_jobs"] or 0
        completed = stats_row["completed"] or 0
        failed = stats_row["failed"] or 0
        pending = stats_row["pending"] or 0
        total_success = stats_row["total_success"] or 0
        total_failures = stats_row["total_failures"] or 0

        # Calculate success rate
        total_chunks = total_success + total_failures
        success_rate = total_success / total_chunks if total_chunks > 0 else 0.0

        return EmbeddingStatus(
            total_jobs=total_jobs,
            completed_jobs=completed,
            failed_jobs=failed,
            pending_jobs=pending,
            total_chunks_processed=total_success,
            total_chunks_failed=total_failures,
            success_rate=success_rate,
            recent_failures=recent_failures,
        )

    async def get_job_history(self, limit: int = 50) -> list[EmbeddingJob]:
        """Recent jobs."""
        await self.init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM embedding_jobs
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            )
            rows = await cursor.fetchall()

        return [self._row_to_job(row) for row in rows]

    async def get_failed_jobs(self) -> list[EmbeddingJob]:
        """All failed jobs."""
        await self.init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM embedding_jobs
                   WHERE status = 'failed'
                   ORDER BY created_at DESC"""
            )
            rows = await cursor.fetchall()

        return [self._row_to_job(row) for row in rows]

    def _row_to_job(self, row: aiosqlite.Row) -> EmbeddingJob:
        """Convert database row to EmbeddingJob."""
        return EmbeddingJob(
            id=row["id"],
            document_id=row["document_id"],
            filename=row["filename"],
            total_chunks=row["total_chunks"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            error_message=row["error_message"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )


# Singleton instance
_tracker: EmbeddingTracker | None = None


async def get_tracker() -> EmbeddingTracker:
    """Get or create the singleton EmbeddingTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = EmbeddingTracker()
        await _tracker.init_db()
    return _tracker
