"""
Document synchronization engine (SFR-005).

Detects file changes in upload directory and synchronizes with vector store:
- New files: ingest
- Modified files: delete old vectors + re-ingest
- Deleted files: remove vectors
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from core.vectorstore import create_vectorstore

from .document_monitor import DocumentChange, compute_file_hash, get_document_monitor
from .ingest import IngestPipeline

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a synchronization run."""

    started_at: str
    completed_at: str
    new_files: int
    modified_files: int
    deleted_files: int
    failed_files: int
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class SyncEngine:
    """Document synchronization engine."""

    def __init__(self):
        """Initialize sync engine with lazy loading."""
        self._ingest: IngestPipeline | None = None
        self._monitor = None
        self._vectorstore = None
        self._last_sync: str | None = None
        self._sync_history: list[SyncResult] = []

    async def _get_ingest_pipeline(self) -> IngestPipeline:
        """Get or create IngestPipeline instance."""
        if self._ingest is None:
            self._ingest = IngestPipeline()
        return self._ingest

    async def _get_monitor(self):
        """Get or create DocumentMonitor instance."""
        if self._monitor is None:
            self._monitor = await get_document_monitor()
        return self._monitor

    async def _get_vectorstore(self):
        """Get or create vectorstore instance."""
        if self._vectorstore is None:
            self._vectorstore = create_vectorstore()
        return self._vectorstore

    async def run_sync(self, source_dir: str | None = None) -> SyncResult:
        """Run document synchronization.

        Detects changes in source directory and synchronizes with vector store:
        - New files: ingest via IngestPipeline
        - Modified files: delete old vectors, re-ingest
        - Deleted files: delete vectors, update monitor status

        Args:
            source_dir: Directory to sync (defaults to settings.upload_dir)

        Returns:
            SyncResult with statistics and errors
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc).isoformat()

        source_dir = source_dir or settings.upload_dir
        logger.info("Starting document sync from %s", source_dir)

        # Initialize counters
        new_files = 0
        modified_files = 0
        deleted_files = 0
        failed_files = 0
        errors: list[str] = []

        try:
            # Get dependencies
            monitor = await self._get_monitor()
            ingest = await self._get_ingest_pipeline()
            vectorstore = await self._get_vectorstore()

            # Detect changes
            changes = await monitor.detect_changes(source_dir)
            logger.info("Detected %d changes", len(changes))

            # Process each change
            for change in changes:
                try:
                    await self._process_change(
                        change, monitor, ingest, vectorstore
                    )

                    # Update counters based on change type
                    if change.change_type == "new":
                        new_files += 1
                    elif change.change_type == "modified":
                        modified_files += 1
                    elif change.change_type == "deleted":
                        deleted_files += 1

                except Exception as e:
                    failed_files += 1
                    error_msg = f"{change.filename} ({change.change_type}): {str(e)}"
                    errors.append(error_msg)
                    logger.error("Failed to process %s: %s", change.filename, e)

            # Calculate duration
            end_time = time.time()
            completed_at = datetime.now(timezone.utc).isoformat()
            duration = end_time - start_time

            # Create result
            result = SyncResult(
                started_at=started_at,
                completed_at=completed_at,
                new_files=new_files,
                modified_files=modified_files,
                deleted_files=deleted_files,
                failed_files=failed_files,
                errors=errors,
                duration_seconds=round(duration, 2),
            )

            # Update sync state
            self._last_sync = completed_at
            self._sync_history.append(result)

            # Keep only last 100 sync results
            if len(self._sync_history) > 100:
                self._sync_history = self._sync_history[-100:]

            logger.info(
                "Sync completed: new=%d, modified=%d, deleted=%d, failed=%d, duration=%.2fs",
                new_files,
                modified_files,
                deleted_files,
                failed_files,
                duration,
            )

            return result

        except Exception as e:
            end_time = time.time()
            completed_at = datetime.now(timezone.utc).isoformat()
            duration = end_time - start_time

            error_msg = f"Sync failed: {str(e)}"
            errors.append(error_msg)
            logger.error("Sync failed: %s", e)

            result = SyncResult(
                started_at=started_at,
                completed_at=completed_at,
                new_files=new_files,
                modified_files=modified_files,
                deleted_files=deleted_files,
                failed_files=failed_files,
                errors=errors,
                duration_seconds=round(duration, 2),
            )

            self._sync_history.append(result)
            if len(self._sync_history) > 100:
                self._sync_history = self._sync_history[-100:]

            return result

    async def _process_change(
        self, change: DocumentChange, monitor, ingest, vectorstore
    ) -> None:
        """Process a single document change.

        Args:
            change: DocumentChange object
            monitor: DocumentMonitor instance
            ingest: IngestPipeline instance
            vectorstore: VectorStore instance
        """
        if change.change_type == "new":
            logger.info("Processing new file: %s", change.filename)
            await self._process_new_file(change, ingest)

        elif change.change_type == "modified":
            logger.info("Processing modified file: %s", change.filename)
            await self._process_modified_file(change, monitor, ingest, vectorstore)

        elif change.change_type == "deleted":
            logger.info("Processing deleted file: %s", change.filename)
            await self._process_deleted_file(change, monitor, vectorstore)

    async def _process_new_file(self, change: DocumentChange, ingest) -> None:
        """Process new file by ingesting it.

        Args:
            change: DocumentChange object
            ingest: IngestPipeline instance
        """
        result = await ingest.ingest(change.file_path)

        if result.status != "completed":
            raise Exception(result.error or "Ingest failed")

        logger.info(
            "Ingested new file %s: %d chunks", change.filename, result.chunk_count
        )

    async def _process_modified_file(
        self, change: DocumentChange, monitor, ingest, vectorstore
    ) -> None:
        """Process modified file by deleting old vectors and re-ingesting.

        Args:
            change: DocumentChange object
            monitor: DocumentMonitor instance
            ingest: IngestPipeline instance
            vectorstore: VectorStore instance
        """
        # Find document_id from monitor DB
        doc_status = await self._find_document_by_path(change.file_path, monitor)

        if doc_status:
            document_id = doc_status.id
            logger.info("Deleting old vectors for document_id: %s", document_id)

            # Delete old vectors by document_id metadata filter
            await self._delete_vectors_by_document_id(document_id, vectorstore)

            # Re-ingest with same document_id
            result = await ingest.ingest(change.file_path, document_id=document_id)

            if result.status != "completed":
                raise Exception(result.error or "Re-ingest failed")

            logger.info(
                "Re-ingested modified file %s: %d chunks",
                change.filename,
                result.chunk_count,
            )
        else:
            # No existing record, treat as new
            logger.warning(
                "Modified file %s not found in monitor, treating as new",
                change.filename,
            )
            await self._process_new_file(change, ingest)

    async def _process_deleted_file(
        self, change: DocumentChange, monitor, vectorstore
    ) -> None:
        """Process deleted file by removing vectors.

        Args:
            change: DocumentChange object
            monitor: DocumentMonitor instance
            vectorstore: VectorStore instance
        """
        # Find document_id from monitor DB
        doc_status = await self._find_document_by_path(change.file_path, monitor)

        if doc_status:
            document_id = doc_status.id
            logger.info("Deleting vectors for deleted file: %s", change.filename)

            # Delete vectors
            await self._delete_vectors_by_document_id(document_id, vectorstore)

            # Update monitor status to "deleted"
            await monitor.update_status(
                document_id, "deleted", error_message="File deleted from disk"
            )

            logger.info("Removed vectors for deleted file: %s", change.filename)
        else:
            logger.warning(
                "Deleted file %s not found in monitor DB, skipping", change.filename
            )

    async def _find_document_by_path(self, file_path: str, monitor):
        """Find document status by file path.

        Args:
            file_path: File path to search for
            monitor: DocumentMonitor instance

        Returns:
            DocumentStatus or None if not found
        """
        all_docs = await monitor.get_all_status()
        for doc in all_docs:
            if doc.file_path == file_path:
                return doc
        return None

    async def _delete_vectors_by_document_id(self, document_id: str, vectorstore) -> None:
        """Delete all vectors with given document_id metadata.

        Note: Since the base vectorstore delete() method takes a list of IDs,
        we need to query for all chunks with this document_id first.
        This implementation assumes we can search/filter by metadata.

        For ChromaDB and Milvus, we'll need to use their specific query APIs.
        As a workaround, we use the vectorstore's get() method if available,
        or we can add a query_by_metadata method to the base class later.

        For now, we'll use a simple approach: query the collection directly.

        Args:
            document_id: Document ID to delete
            vectorstore: VectorStore instance
        """
        # Get all chunks with this document_id
        # This is a placeholder - actual implementation depends on vectorstore type
        try:
            # Try ChromaDB-specific approach
            if hasattr(vectorstore, "_collection"):
                collection = vectorstore._collection
                results = collection.get(where={"document_id": document_id})
                chunk_ids = results.get("ids", [])

                if chunk_ids:
                    logger.info(
                        "Deleting %d chunks for document_id: %s",
                        len(chunk_ids),
                        document_id,
                    )
                    await vectorstore.delete(chunk_ids)
                else:
                    logger.warning(
                        "No chunks found for document_id: %s", document_id
                    )
            else:
                logger.warning(
                    "Cannot delete by document_id - vectorstore type not supported"
                )
        except Exception as e:
            logger.error("Failed to delete vectors for document_id %s: %s", document_id, e)
            raise

    async def get_history(self, limit: int = 20) -> list[SyncResult]:
        """Get recent sync history.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of SyncResult objects (most recent first)
        """
        return self._sync_history[-limit:][::-1]

    async def get_last_sync(self) -> str | None:
        """Get timestamp of last sync.

        Returns:
            ISO timestamp string or None if never synced
        """
        return self._last_sync


# Singleton instance
_engine: SyncEngine | None = None


async def get_sync_engine() -> SyncEngine:
    """Get singleton SyncEngine instance.

    Returns:
        SyncEngine instance
    """
    global _engine
    if _engine is None:
        _engine = SyncEngine()
    return _engine
