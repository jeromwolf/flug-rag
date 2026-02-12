"""Document chunking with multiple strategies."""

import asyncio
import concurrent.futures
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
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


class RecursiveChunker:
    """Split documents using recursive character text splitting.

    Uses a hierarchy of separators (section > paragraph > sentence > word)
    to split at the most meaningful boundary possible.
    """

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
                "\u3002",          # Korean/CJK sentence end
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
        """Split text into chunks with metadata."""
        if not text or not text.strip():
            return []

        text = self._normalize(text)
        raw_chunks = self._splitter.split_text(text)

        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            metadata = dict(base_metadata or {})
            metadata["chunk_index"] = idx
            metadata["chunk_strategy"] = "recursive"

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
        """Chunk page-by-page, preserving page numbers in metadata."""
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

            for chunk in page_chunks:
                chunk.index = global_idx
                chunk.metadata["chunk_index"] = global_idx
                global_idx += 1

            all_chunks.extend(page_chunks)

        return all_chunks

    def _normalize(self, text: str) -> str:
        """Normalize whitespace and clean text."""
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()

    def _is_table_chunk(self, text: str) -> bool:
        """Detect if chunk contains tabular data."""
        lines = text.split('\n')
        pipe_lines = sum(1 for line in lines if '|' in line)
        return pipe_lines >= 2


class EmbeddingSemanticChunker:
    """Split documents based on embedding similarity between sentences.

    Uses bge-m3 embeddings to identify semantic breakpoints where the
    topic changes significantly (cosine similarity drops below threshold).
    """

    # Korean sentence-ending patterns
    _SENTENCE_SPLIT_RE = re.compile(
        r'(?<=[.!?\u3002])\s+|(?<=\ub2e4\.)\s*\n|(?<=\uc694\.)\s*\n|(?<=\ub2c8\ub2e4\.)\s*\n|\n{2,}'
    )

    def __init__(
        self,
        breakpoint_threshold: float | None = None,
        min_chunk_size: int | None = None,
        max_chunk_size: int | None = None,
    ):
        self.breakpoint_threshold = breakpoint_threshold or settings.semantic_breakpoint_threshold
        self.min_chunk_size = min_chunk_size or settings.semantic_min_chunk_size
        self.max_chunk_size = max_chunk_size or settings.chunk_size * 2  # 2x recursive chunk_size
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from core.embeddings.local import LocalEmbedding
            self._embedder = LocalEmbedding()
        return self._embedder

    @staticmethod
    def _embed_sync(embedder, texts: list[str]) -> list:
        """Run async embedding synchronously, safe from async context.

        Uses a separate thread with its own event loop to avoid
        'RuntimeError: This event loop is already running'.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, embedder.embed_texts(texts))
            return future.result()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using Korean-aware patterns."""
        # First normalize
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Split by sentence boundaries
        segments = self._SENTENCE_SPLIT_RE.split(text)

        # Filter empty and merge very short segments
        sentences = []
        buffer = ""
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            buffer = f"{buffer} {seg}".strip() if buffer else seg
            # Only emit if buffer is long enough to be a meaningful unit
            if len(buffer) >= 30:
                sentences.append(buffer)
                buffer = ""

        # Don't lose trailing text
        if buffer:
            if sentences:
                sentences[-1] = f"{sentences[-1]} {buffer}"
            else:
                sentences.append(buffer)

        return sentences

    def chunk(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into semantically coherent chunks using embeddings."""
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # For very short texts, return as single chunk
        if len(sentences) <= 2:
            return [
                Chunk(
                    id=str(uuid.uuid4()),
                    content=text.strip(),
                    index=0,
                    metadata={**(base_metadata or {}), "chunk_index": 0, "chunk_strategy": "semantic"},
                )
            ]

        # Embed all sentences (run async embedding in separate thread to avoid
        # "event loop already running" error when called from async context)
        embedder = self._get_embedder()
        embeddings = self._embed_sync(embedder, sentences)
        emb_array = np.array(embeddings)

        # Calculate cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(emb_array) - 1):
            sim = float(np.dot(emb_array[i], emb_array[i + 1]))
            sim = max(0.0, min(1.0, sim))
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                breakpoints.append(i + 1)  # Break AFTER sentence i

        # Build chunks from breakpoints
        chunk_groups = []
        start = 0
        for bp in breakpoints:
            group = sentences[start:bp]
            chunk_groups.append(group)
            start = bp
        # Last group
        if start < len(sentences):
            chunk_groups.append(sentences[start:])

        # Merge small groups and enforce max size
        chunks = self._merge_and_limit(chunk_groups)

        # Create Chunk objects
        result = []
        for idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            metadata = dict(base_metadata or {})
            metadata["chunk_index"] = idx
            metadata["chunk_strategy"] = "semantic"

            if self._is_table_chunk(chunk_text):
                metadata["has_table"] = True

            result.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    content=chunk_text,
                    index=idx,
                    metadata=metadata,
                )
            )

        return result

    def _merge_and_limit(self, groups: list[list[str]]) -> list[str]:
        """Merge small groups and split oversized ones."""
        merged = []
        current = ""

        for group in groups:
            group_text = " ".join(group)

            if not current:
                current = group_text
            elif len(current) + len(group_text) + 1 < self.min_chunk_size:
                # Too small, merge with current
                current = f"{current} {group_text}"
            else:
                # Current is big enough, emit it
                if len(current) >= self.min_chunk_size:
                    merged.append(current)
                    current = group_text
                else:
                    current = f"{current} {group_text}"

        if current:
            merged.append(current)

        # Split any chunks that exceed max size using simple splitting
        final = []
        for chunk_text in merged:
            if len(chunk_text) <= self.max_chunk_size:
                final.append(chunk_text)
            else:
                # Fall back to simple splitting for oversized chunks
                for i in range(0, len(chunk_text), self.max_chunk_size):
                    part = chunk_text[i:i + self.max_chunk_size].strip()
                    if part:
                        final.append(part)

        return final

    def chunk_with_pages(
        self,
        pages: list[dict],
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Chunk page-by-page, preserving page numbers in metadata."""
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

            for chunk in page_chunks:
                chunk.index = global_idx
                chunk.metadata["chunk_index"] = global_idx
                global_idx += 1

            all_chunks.extend(page_chunks)

        return all_chunks

    def _is_table_chunk(self, text: str) -> bool:
        """Detect if chunk contains tabular data."""
        lines = text.split('\n')
        pipe_lines = sum(1 for line in lines if '|' in line)
        return pipe_lines >= 2


# Backward-compatible alias
SemanticChunker = RecursiveChunker


def create_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveChunker | EmbeddingSemanticChunker:
    """Factory function to create chunker based on settings.

    Args:
        strategy: "recursive" or "semantic". Defaults to settings.chunk_strategy.
        chunk_size: Override chunk size.
        chunk_overlap: Override chunk overlap.

    Returns:
        Configured chunker instance.
    """
    strategy = strategy or settings.chunk_strategy

    if strategy == "semantic":
        return EmbeddingSemanticChunker(
            max_chunk_size=chunk_size,
        )
    else:
        return RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
