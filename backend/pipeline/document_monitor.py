"""
Document processing monitoring module (SFR-008).

Tracks document ingestion status, detects file changes, and provides
processing statistics via SQLite storage.
"""

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from config.settings import settings


@dataclass
class DocumentStatus:
    """Document processing status record."""

    id: str
    filename: str
    file_type: str
    file_path: str
    file_hash: str
    status: str
    chunk_count: int
    error_message: Optional[str]
    processed_at: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class DocumentStatusSummary:
    """Aggregate document processing statistics."""

    total: int
    by_status: dict[str, int]  # {"completed": 10, "failed": 2, ...}
    by_file_type: dict[str, int]  # {".pdf": 5, ".hwp": 3, ...}


@dataclass
class DocumentChange:
    """File change detection result."""

    filename: str
    file_path: str
    change_type: str  # "new", "modified", "deleted"
    old_hash: Optional[str]
    new_hash: Optional[str]


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of file content.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal MD5 hash string
    """
    hasher = hashlib.md5()
    path = Path(file_path)

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


class DocumentMonitor:
    """Document processing status monitor with SQLite backend."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize monitor.

        Args:
            db_path: SQLite database path (defaults to data/document_monitor.db)
        """
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data" / "document_monitor.db"
        self.db_path = db_path
        self._initialized = False

    async def init_db(self) -> None:
        """Create document_status table if not exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS document_status (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    processed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.commit()

        self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized."""
        if not self._initialized:
            await self.init_db()

    async def register_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        file_path: str,
        file_hash: str,
    ) -> None:
        """Register or update a document.

        Args:
            document_id: Unique document identifier
            filename: Original filename
            file_type: File extension (e.g., ".pdf")
            file_path: Full path to file
            file_hash: MD5 hash of file content
        """
        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Check if document exists
            cursor = await db.execute(
                "SELECT id FROM document_status WHERE id = ?", (document_id,)
            )
            exists = await cursor.fetchone()

            if exists:
                # Update existing record
                await db.execute(
                    """
                    UPDATE document_status
                    SET filename = ?,
                        file_type = ?,
                        file_path = ?,
                        file_hash = ?,
                        status = 'pending',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (filename, file_type, file_path, file_hash, now, document_id),
                )
            else:
                # Insert new record
                await db.execute(
                    """
                    INSERT INTO document_status
                    (id, filename, file_type, file_path, file_hash, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (document_id, filename, file_type, file_path, file_hash, now, now),
                )

            await db.commit()

    async def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update document processing status.

        Args:
            document_id: Document identifier
            status: New status (pending/processing/completed/failed)
            chunk_count: Number of chunks created (optional)
            error_message: Error message if failed (optional)
        """
        await self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            updates = ["status = ?", "updated_at = ?"]
            params = [status, now]

            if chunk_count is not None:
                updates.append("chunk_count = ?")
                params.append(chunk_count)

            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)

            if status in ("completed", "failed"):
                updates.append("processed_at = ?")
                params.append(now)

            params.append(document_id)

            query = f"UPDATE document_status SET {', '.join(updates)} WHERE id = ?"
            await db.execute(query, params)
            await db.commit()

    async def get_document(self, document_id: str) -> Optional[DocumentStatus]:
        """Get single document status.

        Args:
            document_id: Document identifier

        Returns:
            DocumentStatus or None if not found
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM document_status WHERE id = ?", (document_id,)
            )
            row = await cursor.fetchone()

            if row:
                return DocumentStatus(
                    id=row["id"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    status=row["status"],
                    chunk_count=row["chunk_count"],
                    error_message=row["error_message"],
                    processed_at=row["processed_at"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            return None

    async def get_all_status(
        self, status_filter: Optional[str] = None
    ) -> list[DocumentStatus]:
        """List all document statuses.

        Args:
            status_filter: Optional status filter (e.g., "completed")

        Returns:
            List of DocumentStatus objects
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if status_filter:
                cursor = await db.execute(
                    "SELECT * FROM document_status WHERE status = ? ORDER BY created_at DESC",
                    (status_filter,),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM document_status ORDER BY created_at DESC"
                )

            rows = await cursor.fetchall()

            return [
                DocumentStatus(
                    id=row["id"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    status=row["status"],
                    chunk_count=row["chunk_count"],
                    error_message=row["error_message"],
                    processed_at=row["processed_at"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def get_status_summary(self) -> DocumentStatusSummary:
        """Get aggregate document processing statistics.

        Returns:
            DocumentStatusSummary with counts by status and file type
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            # Total count
            cursor = await db.execute("SELECT COUNT(*) FROM document_status")
            total = (await cursor.fetchone())[0]

            # By status
            cursor = await db.execute(
                "SELECT status, COUNT(*) FROM document_status GROUP BY status"
            )
            by_status = {row[0]: row[1] for row in await cursor.fetchall()}

            # By file type
            cursor = await db.execute(
                "SELECT file_type, COUNT(*) FROM document_status GROUP BY file_type"
            )
            by_file_type = {row[0]: row[1] for row in await cursor.fetchall()}

            return DocumentStatusSummary(
                total=total, by_status=by_status, by_file_type=by_file_type
            )

    async def detect_changes(self, upload_dir: str) -> list[DocumentChange]:
        """Detect file changes in upload directory.

        Compares current files with stored records to identify:
        - New files (not in database)
        - Modified files (hash differs)
        - Deleted files (in database but not on disk)

        Args:
            upload_dir: Directory to scan for files

        Returns:
            List of DocumentChange objects
        """
        await self._ensure_initialized()

        changes: list[DocumentChange] = []
        upload_path = Path(upload_dir)

        # Get all stored documents
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT file_path, file_hash FROM document_status")
            stored_docs = {row["file_path"]: row["file_hash"] for row in await cursor.fetchall()}

        # Scan current files
        current_files = set()
        if upload_path.exists():
            for file_path in upload_path.rglob("*"):
                if file_path.is_file():
                    file_path_str = str(file_path)
                    current_files.add(file_path_str)

                    # Compute current hash
                    new_hash = compute_file_hash(file_path_str)

                    if file_path_str not in stored_docs:
                        # New file
                        changes.append(
                            DocumentChange(
                                filename=file_path.name,
                                file_path=file_path_str,
                                change_type="new",
                                old_hash=None,
                                new_hash=new_hash,
                            )
                        )
                    elif stored_docs[file_path_str] != new_hash:
                        # Modified file
                        changes.append(
                            DocumentChange(
                                filename=file_path.name,
                                file_path=file_path_str,
                                change_type="modified",
                                old_hash=stored_docs[file_path_str],
                                new_hash=new_hash,
                            )
                        )

        # Check for deleted files
        for stored_path in stored_docs:
            if stored_path not in current_files:
                changes.append(
                    DocumentChange(
                        filename=Path(stored_path).name,
                        file_path=stored_path,
                        change_type="deleted",
                        old_hash=stored_docs[stored_path],
                        new_hash=None,
                    )
                )

        return changes


# Singleton instance
_monitor: Optional[DocumentMonitor] = None


async def get_document_monitor() -> DocumentMonitor:
    """Get singleton DocumentMonitor instance.

    Returns:
        Initialized DocumentMonitor
    """
    global _monitor
    if _monitor is None:
        _monitor = DocumentMonitor()
        await _monitor.init_db()
    return _monitor
