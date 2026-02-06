"""Semantic chunking for document text."""

import re
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


@dataclass
class Chunk:
    """A text chunk with metadata."""
    id: str
    content: str
    index: int  # Position in original document
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.token_count == 0:
            # Rough estimate: ~1.5 chars per token for Korean
            self.token_count = len(self.content) * 2 // 3


class SemanticChunker:
    """Split documents into semantically meaningful chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",      # Major section breaks
                "\n\n",        # Paragraph breaks
                "\n",          # Line breaks
                "。",          # Korean/CJK sentence end
                ". ",          # English sentence end
                ".\n",
                " ",           # Word boundary
                "",            # Character level (last resort)
            ],
            length_function=len,
        )

    def chunk(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into chunks with metadata.

        Args:
            text: Full document text.
            base_metadata: Metadata inherited by all chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        # Pre-process: normalize whitespace
        text = self._normalize(text)

        # Split using LangChain splitter
        raw_chunks = self._splitter.split_text(text)

        # Create Chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            metadata = dict(base_metadata or {})
            metadata["chunk_index"] = idx

            # Detect if chunk contains a table
            if self._is_table_chunk(chunk_text):
                metadata["has_table"] = True

            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    content=chunk_text,
                    index=idx,
                    metadata=metadata,
                )
            )

        return chunks

    def chunk_with_pages(
        self,
        pages: list[dict],
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Chunk page-by-page, preserving page numbers in metadata.

        Args:
            pages: List of {page_num, content} dicts.
            base_metadata: Base metadata for all chunks.

        Returns:
            List of Chunks with page_number in metadata.
        """
        all_chunks = []
        global_idx = 0

        for page in pages:
            page_num = page.get("page_num", 0)
            content = page.get("content", "")

            if not content.strip():
                continue

            page_meta = dict(base_metadata or {})
            page_meta["page_number"] = page_num

            page_chunks = self.chunk(content, base_metadata=page_meta)

            # Update global indices
            for chunk in page_chunks:
                chunk.index = global_idx
                chunk.metadata["chunk_index"] = global_idx
                global_idx += 1

            all_chunks.extend(page_chunks)

        return all_chunks

    def _normalize(self, text: str) -> str:
        """Normalize whitespace and clean text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Remove null bytes and control chars (except newline, tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()

    def _is_table_chunk(self, text: str) -> bool:
        """Detect if chunk contains tabular data."""
        lines = text.split('\n')
        pipe_lines = sum(1 for line in lines if '|' in line)
        return pipe_lines >= 2
