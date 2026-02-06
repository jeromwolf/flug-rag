"""Document ingest orchestrator: load → chunk → embed → store."""

import uuid
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from core.embeddings import BaseEmbedding, create_embedder
from core.vectorstore import BaseVectorStore, create_vectorstore

from .chunker import Chunk, SemanticChunker
from .loader import DocumentLoader
from .metadata import MetadataExtractor


@dataclass
class IngestResult:
    """Result of a document ingestion."""
    document_id: str
    filename: str
    chunk_count: int
    status: str  # completed, failed
    error: str | None = None


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
            apply_ocr: Force OCR processing.

        Returns:
            IngestResult with status and chunk count.
        """
        path = Path(file_path)
        doc_id = document_id or str(uuid.uuid4())

        try:
            # Step 1: Load document
            loaded = await self.loader.load(path, apply_ocr=apply_ocr)

            if not loaded.content.strip():
                return IngestResult(
                    document_id=doc_id,
                    filename=path.name,
                    chunk_count=0,
                    status="failed",
                    error="No text extracted from document",
                )

            # Step 2: Extract metadata
            auto_metadata = self.metadata_extractor.extract(
                text=loaded.content,
                filename=path.name,
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
                return IngestResult(
                    document_id=doc_id,
                    filename=path.name,
                    chunk_count=0,
                    status="failed",
                    error="No chunks generated",
                )

            # Step 4: Generate embeddings (batch)
            texts = [c.content for c in chunks]
            embeddings = await self.embedder.embed_texts(texts)

            # Step 5: Store in vector DB
            ids = [c.id for c in chunks]
            metadatas = [c.metadata for c in chunks]

            await self.vectorstore.add(
                ids=ids,
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            return IngestResult(
                document_id=doc_id,
                filename=path.name,
                chunk_count=len(chunks),
                status="completed",
            )

        except Exception as e:
            return IngestResult(
                document_id=doc_id,
                filename=path.name,
                chunk_count=0,
                status="failed",
                error=str(e),
            )

    async def ingest_batch(
        self,
        file_paths: list[str | Path],
        extra_metadata: dict | None = None,
    ) -> list[IngestResult]:
        """Ingest multiple documents sequentially.

        Args:
            file_paths: List of file paths.
            extra_metadata: Shared metadata for all documents.

        Returns:
            List of IngestResult for each document.
        """
        results = []
        for path in file_paths:
            result = await self.ingest(path, extra_metadata=extra_metadata)
            results.append(result)
        return results
