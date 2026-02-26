"""Document ingest orchestrator: load → chunk → embed → store."""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from core.embeddings import BaseEmbedding, create_embedder
from core.vectorstore import BaseVectorStore, create_vectorstore

from .chunker import Chunk, SemanticChunker
from .loader import DocumentLoader
from .metadata import MetadataExtractor

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of a document ingestion."""
    document_id: str
    filename: str
    chunk_count: int
    status: str  # completed, failed
    error: str | None = None
    pii_warnings: list[str] | None = None
    metadata: dict | None = None


class IngestPipeline:
    """Orchestrates the full document ingestion flow."""

    def __init__(
        self,
        vectorstore: BaseVectorStore | None = None,
        embedder: BaseEmbedding | None = None,
        chunker: SemanticChunker | None = None,
    ):
        self.loader = DocumentLoader()
        self.chunker = chunker or SemanticChunker()
        self.metadata_extractor = MetadataExtractor()
        self.embedder = embedder or create_embedder()
        self.vectorstore = vectorstore or create_vectorstore()

    async def ingest(
        self,
        file_path: str | Path,
        document_id: str | None = None,
        extra_metadata: dict | None = None,
        apply_ocr: bool = False,
        dp_mode: str = "auto",
    ) -> IngestResult:
        """Ingest a single document through the full pipeline.

        Steps:
        1. Load document (text extraction)
        2. Extract metadata
        3. Chunk text
        4. Generate embeddings
        5. Store in vector DB

        Args:
            file_path: Path to the document file.
            document_id: Pre-assigned document ID (or auto-generate).
            extra_metadata: Additional metadata to attach.
            apply_ocr: Force OCR processing (backward compat).
            dp_mode: Document Parse mode - "auto", "force_dp", or "local_only".

        Returns:
            IngestResult with status and chunk count.
        """
        path = Path(file_path)
        doc_id = document_id or str(uuid.uuid4())

        # 문서 모니터링 등록
        try:
            from .document_monitor import compute_file_hash, get_document_monitor
            monitor = await get_document_monitor()
            file_hash = compute_file_hash(str(path))
            await monitor.register_document(
                doc_id, path.name, path.suffix.lstrip("."), str(path), file_hash,
            )
            await monitor.update_status(doc_id, "processing")
        except Exception as e:
            logger.debug("Document monitor unavailable: %s", e)
            monitor = None

        # 임베딩 트래킹 준비
        embed_job_id = None

        try:
            # Step 1: Load document
            loaded = await self.loader.load(path, apply_ocr=apply_ocr, dp_mode=dp_mode)

            if not loaded.content.strip():
                error_msg = "No text extracted from document"
                if monitor:
                    await monitor.update_status(doc_id, "failed", error_message=error_msg)
                await self._enqueue_on_failure(doc_id, path, error_msg)
                return IngestResult(
                    document_id=doc_id,
                    filename=path.name,
                    chunk_count=0,
                    status="failed",
                    error=error_msg,
                )

            # Step 1.5: PII scan (best-effort, non-blocking)
            pii_warnings = []
            try:
                from .pii_detector import get_pii_detector
                pii_result = get_pii_detector().scan(loaded.content[:50000])
                pii_warnings = pii_result.warnings
                if pii_result.has_pii:
                    logger.warning(
                        "PII detected in %s: %s", path.name, ", ".join(pii_warnings)
                    )
            except Exception as e:
                logger.debug("PII scan skipped: %s", e)

            # Step 2: Extract metadata
            auto_metadata = self.metadata_extractor.extract(
                text=loaded.content,
                filename=path.name,
                file_path=str(path),
            )
            base_metadata = {
                "document_id": doc_id,
                "filename": path.name,
                "file_type": path.suffix.lstrip("."),
                **auto_metadata,
                **(extra_metadata or {}),
            }

            # Step 3: Chunk
            if loaded.pages:
                chunks = self.chunker.chunk_with_pages(loaded.pages, base_metadata)
            else:
                chunks = self.chunker.chunk(loaded.content, base_metadata)

            if not chunks:
                error_msg = "No chunks generated"
                if monitor:
                    await monitor.update_status(doc_id, "failed", error_message=error_msg)
                await self._enqueue_on_failure(doc_id, path, error_msg)
                return IngestResult(
                    document_id=doc_id,
                    filename=path.name,
                    chunk_count=0,
                    status="failed",
                    error=error_msg,
                )

            # Step 4: Generate embeddings (batch) + 트래킹
            try:
                from core.embeddings.tracker import get_tracker
                tracker = await get_tracker()
                embed_job_id = await tracker.start_job(doc_id, path.name, len(chunks))
            except Exception:
                tracker = None

            texts = [c.content for c in chunks]
            embeddings = await self.embedder.embed_texts(texts)

            if tracker and embed_job_id:
                await tracker.update_progress(embed_job_id, len(embeddings), 0)

            # Step 5: Store in vector DB
            ids = [c.id for c in chunks]
            metadatas = [c.metadata for c in chunks]

            await self.vectorstore.add(
                ids=ids,
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            # 성공 트래킹
            if tracker and embed_job_id:
                await tracker.complete_job(embed_job_id, "completed")
            if monitor:
                await monitor.update_status(doc_id, "completed", chunk_count=len(chunks))

            return IngestResult(
                document_id=doc_id,
                filename=path.name,
                chunk_count=len(chunks),
                status="completed",
                pii_warnings=pii_warnings or None,
                metadata=base_metadata,
            )

        except Exception as e:
            # 실패 트래킹
            if embed_job_id:
                try:
                    from core.embeddings.tracker import get_tracker
                    tracker = await get_tracker()
                    await tracker.complete_job(embed_job_id, "failed", str(e))
                except Exception:
                    pass
            if monitor:
                try:
                    await monitor.update_status(doc_id, "failed", error_message=str(e))
                except Exception:
                    pass
            await self._enqueue_on_failure(doc_id, path, str(e))
            return IngestResult(
                document_id=doc_id,
                filename=path.name,
                chunk_count=0,
                status="failed",
                error=str(e),
            )

    async def _enqueue_on_failure(self, doc_id: str, path: Path, error: str) -> None:
        """실패한 문서를 재처리 큐에 추가."""
        try:
            from .reprocess_queue import get_reprocess_queue
            queue = await get_reprocess_queue()
            await queue.enqueue(doc_id, path.name, str(path), error)
        except Exception as e:
            logger.debug("Reprocess queue unavailable: %s", e)

    async def ingest_batch(
        self,
        file_paths: list[str | Path],
        extra_metadata: dict | None = None,
        dp_mode: str = "auto",
        max_concurrent: int = 5,
    ) -> list[IngestResult]:
        """Ingest multiple documents with bounded concurrency.

        Args:
            file_paths: List of file paths.
            extra_metadata: Shared metadata for all documents.
            dp_mode: Document Parse mode - "auto", "force_dp", or "local_only".
            max_concurrent: Maximum parallel ingestion tasks (default 5).

        Returns:
            List of IngestResult for each document (order preserved).
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _ingest_with_limit(path: str | Path) -> IngestResult:
            async with semaphore:
                return await self.ingest(path, extra_metadata=extra_metadata, dp_mode=dp_mode)

        results = await asyncio.gather(
            *[_ingest_with_limit(p) for p in file_paths],
            return_exceptions=False,
        )
        return list(results)
