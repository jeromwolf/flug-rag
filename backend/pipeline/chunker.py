"""Document chunking with multiple strategies."""

from __future__ import annotations

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


class TableChunker:
    """Split table content row-by-row while preserving headers.

    Designed for tabular documents such as spec sheets and Korean regulatory
    tables commonly found in HWP/PDF files. Non-table regions are delegated
    to RecursiveChunker.

    Detection patterns:
    - Markdown-style pipe-delimited tables (``|col|col|``)
    - Tab-separated tables (3+ consecutive lines with equal tab count)
    - Separator rows like ``|---|---|`` or ``+---+---+``
    """

    # Separator row that sits between header and body
    _SEPARATOR_RE = re.compile(r'^[\s|+\-:]+$')

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.base_chunker = RecursiveChunker(chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split *text* into chunks, treating table regions specially."""
        if not text or not text.strip():
            return []

        regions = self._split_regions(text)
        chunks: list[Chunk] = []
        global_idx = 0

        for kind, content in regions:
            if kind == "table":
                table_chunks = self._chunk_table(content, base_metadata, global_idx)
                chunks.extend(table_chunks)
                global_idx += len(table_chunks)
            else:
                text_chunks = self.base_chunker.chunk(content, base_metadata)
                for c in text_chunks:
                    c.index = global_idx
                    c.metadata["chunk_index"] = global_idx
                    global_idx += 1
                chunks.extend(text_chunks)

        return chunks

    def chunk_with_pages(
        self,
        pages: list[dict],
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Chunk page-by-page, preserving page numbers."""
        all_chunks: list[Chunk] = []
        global_idx = 0

        for page in pages:
            page_num = page.get("page_num", 0)
            content = page.get("content", "")
            if not content.strip():
                continue

            page_meta = dict(base_metadata or {})
            page_meta["page_number"] = page_num

            page_chunks = self.chunk(content, base_metadata=page_meta)
            for c in page_chunks:
                c.index = global_idx
                c.metadata["chunk_index"] = global_idx
                global_idx += 1
            all_chunks.extend(page_chunks)

        return all_chunks

    # ------------------------------------------------------------------
    # Region detection
    # ------------------------------------------------------------------

    def _split_regions(self, text: str) -> list[tuple[str, str]]:
        """Split *text* into alternating ("text", ...) and ("table", ...) regions."""
        lines = text.split('\n')
        regions: list[tuple[str, str]] = []
        buf: list[str] = []
        in_table = False

        i = 0
        while i < len(lines):
            # Look-ahead: does a table start here?
            table_end = self._detect_table_end(lines, i)
            if table_end is not None and not in_table:
                # Flush preceding text
                if buf:
                    regions.append(("text", "\n".join(buf)))
                    buf = []
                # Collect table lines
                regions.append(("table", "\n".join(lines[i:table_end])))
                i = table_end
                continue

            buf.append(lines[i])
            i += 1

        if buf:
            regions.append(("text", "\n".join(buf)))

        return regions

    def _detect_table_end(self, lines: list[str], start: int) -> int | None:
        """If a table begins at *start*, return the index past its last row.

        Returns ``None`` when no table is detected.
        """
        # --- Pipe-delimited tables ---
        if self._is_pipe_line(lines[start]):
            end = start + 1
            while end < len(lines) and (self._is_pipe_line(lines[end]) or self._SEPARATOR_RE.match(lines[end])):
                end += 1
            if end - start >= 3:
                return end

        # --- Tab-separated tables ---
        tab_count = lines[start].count('\t')
        if tab_count >= 2:
            end = start + 1
            while end < len(lines) and lines[end].count('\t') == tab_count:
                end += 1
            if end - start >= 3:
                return end

        return None

    @staticmethod
    def _is_pipe_line(line: str) -> bool:
        """Return True if *line* looks like a pipe-delimited table row."""
        return line.count('|') >= 2

    # ------------------------------------------------------------------
    # Table chunking
    # ------------------------------------------------------------------

    def _chunk_table(
        self,
        table_text: str,
        base_metadata: dict | None,
        start_idx: int,
    ) -> list[Chunk]:
        """Chunk a single table region row-by-row with header context."""
        lines = [l for l in table_text.split('\n') if l.strip()]
        if not lines:
            return []

        # Identify header and data rows
        header_line = lines[0]
        data_lines: list[str] = []
        for line in lines[1:]:
            if self._SEPARATOR_RE.match(line):
                continue  # skip separator rows
            data_lines.append(line)

        if not data_lines:
            # Table with only a header – return as single chunk
            meta = dict(base_metadata or {})
            meta.update(chunk_index=start_idx, chunk_strategy="table", has_table=True, table_header=header_line)
            return [Chunk(id=str(uuid.uuid4()), content=header_line, index=start_idx, metadata=meta)]

        # Group small rows together to avoid tiny chunks
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_len = 0

        for row in data_lines:
            row_with_header = f"{header_line}\n{row}"
            row_len = len(row_with_header)

            if current_group and current_len + len(row) + 1 > self.chunk_size:
                groups.append(current_group)
                current_group = []
                current_len = 0

            current_group.append(row)
            current_len += len(row) + 1  # +1 for newline

            # Also enforce: if individual rows are very small (< 50 chars),
            # keep accumulating unless we would exceed chunk_size
            if len(row) >= 50 and current_len + len(header_line) + 1 >= 50:
                # Row is substantial enough on its own – flush
                if current_len + len(header_line) + 1 <= self.chunk_size:
                    continue  # still fits, keep accumulating
                groups.append(current_group)
                current_group = []
                current_len = 0

        if current_group:
            groups.append(current_group)

        chunks: list[Chunk] = []
        for group in groups:
            body = "\n".join(group)
            content = f"{header_line}\n{body}"
            meta = dict(base_metadata or {})
            idx = start_idx + len(chunks)
            meta.update(
                chunk_index=idx,
                chunk_strategy="table",
                has_table=True,
                table_header=header_line,
            )
            chunks.append(Chunk(id=str(uuid.uuid4()), content=content, index=idx, metadata=meta))

        return chunks


class HierarchicalChunker:
    """Create parent-child chunk structures for hierarchical documents.

    Targets Korean legal texts (법령), company regulations (사규), and
    ISO-style numbered sections. Each child chunk gets the parent title
    prepended so it can stand alone during retrieval.

    Hierarchy levels:
        0 – Article  (제N조)
        1 – Paragraph (①~⑳ or 제N항)
        2 – Item      (N. or 제N호)
        3 – Sub-item  (가. 나. 다. …)
    """

    # --- Korean legal patterns ---
    ARTICLE_RE = re.compile(r'^제\d+조(?:의\d+)?\s*[\(\(]?')
    PARAGRAPH_RE = re.compile(r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|^제?\d+항\s')
    ITEM_RE = re.compile(r'^\d+\.\s|^제?\d+호\s')
    SUBITEM_RE = re.compile(r'^[가나다라마바사아자차카타파하]\.\s')

    # --- ISO / numbered section patterns ---
    ISO_SECTION_RE = re.compile(r'^(\d+(?:\.\d+)*)\s')

    # Circled numbers for quick lookup
    _CIRCLE_NUMS = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        include_parent_context: bool = True,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.include_parent_context = include_parent_context
        self.base_chunker = RecursiveChunker(chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split *text* using hierarchical structure when detected."""
        if not text or not text.strip():
            return []

        if not self._has_legal_structure(text):
            return self.base_chunker.chunk(text, base_metadata)

        sections = self._parse_sections(text)
        return self._sections_to_chunks(sections, base_metadata)

    def chunk_with_pages(
        self,
        pages: list[dict],
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Combine all pages and apply hierarchical chunking.

        Hierarchical structures (articles, paragraphs) commonly span page
        boundaries, so we combine first and chunk the whole text.
        """
        combined = "\n\n".join(
            p.get("content", "") for p in pages if p.get("content", "").strip()
        )
        if not combined.strip():
            return []

        return self.chunk(combined, base_metadata)

    # ------------------------------------------------------------------
    # Structure detection
    # ------------------------------------------------------------------

    def _has_legal_structure(self, text: str) -> bool:
        """Return True when *text* contains >= 2 article headings."""
        return len(self.ARTICLE_RE.findall(text, re.MULTILINE if False else 0)) >= 2 or \
               len(re.findall(r'^제\d+조', text, re.MULTILINE)) >= 2

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _line_level(self, line: str) -> int | None:
        """Return the hierarchy level of *line*, or None for plain text."""
        stripped = line.strip()
        if not stripped:
            return None
        if self.ARTICLE_RE.match(stripped):
            return 0
        if self.PARAGRAPH_RE.match(stripped):
            return 1
        if self.ITEM_RE.match(stripped):
            return 2
        if self.SUBITEM_RE.match(stripped):
            return 3
        return None

    def _extract_title(self, line: str) -> str:
        """Extract a short title from a heading line.

        For ``제1조(목적) 이 법은 ...`` returns ``제1조(목적)``.
        """
        stripped = line.strip()
        # 제N조(title) ...
        m = re.match(r'(제\d+조(?:의\d+)?\s*[\(\(][^)\)]*[\)\)])', stripped)
        if m:
            return m.group(1)
        # 제N조 ...
        m = re.match(r'(제\d+조(?:의\d+)?)', stripped)
        if m:
            return m.group(1)
        # Circled number
        if stripped and stripped[0] in self._CIRCLE_NUMS:
            return stripped[0]
        # Numbered item
        m = re.match(r'(\d+\.)\s', stripped)
        if m:
            return m.group(1)
        # Korean sub-item
        m = re.match(r'([가나다라마바사아자차카타파하]\.)\s', stripped)
        if m:
            return m.group(1)
        # Fallback – first 30 chars
        return stripped[:30]

    def _parse_sections(self, text: str) -> list[dict]:
        """Parse *text* into a flat list of section dicts.

        Each dict: ``{level, title, content, children: []}``
        """
        lines = text.split('\n')
        sections: list[dict] = []
        current: dict | None = None
        buf: list[str] = []

        def _flush():
            nonlocal current, buf
            if current is not None:
                current["content"] = "\n".join(buf).strip()
                sections.append(current)
            elif buf:
                # Text before the first heading
                sections.append({"level": -1, "title": "", "content": "\n".join(buf).strip(), "children": []})
            buf = []

        for line in lines:
            level = self._line_level(line)
            if level is not None and level == 0:
                _flush()
                current = {
                    "level": level,
                    "title": self._extract_title(line),
                    "content": "",
                    "children": [],
                }
                buf = [line]
            else:
                buf.append(line)

        _flush()

        # Now split each article-level section into sub-sections (paragraphs, items)
        for section in sections:
            if section["level"] != 0:
                continue
            section["children"] = self._parse_children(section["content"])

        return sections

    def _parse_children(self, article_text: str) -> list[dict]:
        """Parse paragraphs / items within a single article's text."""
        lines = article_text.split('\n')
        children: list[dict] = []
        current_child: dict | None = None
        buf: list[str] = []

        def _flush_child():
            nonlocal current_child, buf
            if current_child is not None:
                current_child["content"] = "\n".join(buf).strip()
                children.append(current_child)
            buf = []

        for line in lines:
            level = self._line_level(line)
            if level is not None and level >= 1:
                _flush_child()
                current_child = {
                    "level": level,
                    "title": self._extract_title(line),
                    "content": "",
                }
                buf = [line]
            else:
                buf.append(line)

        _flush_child()
        return children

    # ------------------------------------------------------------------
    # Chunk creation
    # ------------------------------------------------------------------

    def _sections_to_chunks(
        self,
        sections: list[dict],
        base_metadata: dict | None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0

        for section in sections:
            parent_title = section["title"]
            parent_id = str(uuid.uuid4())
            full_content = section["content"]
            children = section.get("children", [])

            # Pre-heading text or sections without hierarchy
            if section["level"] < 0 or not children:
                # Small enough – emit as single chunk
                if len(full_content) <= self.chunk_size:
                    meta = dict(base_metadata or {})
                    meta.update(
                        chunk_index=idx,
                        chunk_strategy="hierarchical",
                        hierarchy_level=max(section["level"], 0),
                        parent_title=parent_title,
                    )
                    chunks.append(Chunk(id=parent_id, content=full_content, index=idx, metadata=meta))
                    idx += 1
                else:
                    # Too large – fallback to base chunker
                    fallback = self.base_chunker.chunk(full_content, base_metadata)
                    for c in fallback:
                        c.index = idx
                        c.metadata["chunk_index"] = idx
                        c.metadata["chunk_strategy"] = "hierarchical"
                        c.metadata["parent_title"] = parent_title
                        idx += 1
                    chunks.extend(fallback)
                continue

            # --- Article with children ---
            # Emit parent chunk (full article text) if it fits
            if len(full_content) <= self.chunk_size:
                meta = dict(base_metadata or {})
                meta.update(
                    chunk_index=idx,
                    chunk_strategy="hierarchical",
                    hierarchy_level=0,
                    parent_title=parent_title,
                )
                chunks.append(Chunk(id=parent_id, content=full_content, index=idx, metadata=meta))
                idx += 1
            else:
                # Parent too large – emit only child chunks
                pass

            # Emit child chunks
            for child in children:
                child_content = child["content"]
                if self.include_parent_context and parent_title:
                    child_content = f"[{parent_title}] {child_content}"

                if len(child_content) <= self.chunk_size:
                    meta = dict(base_metadata or {})
                    meta.update(
                        chunk_index=idx,
                        chunk_strategy="hierarchical",
                        hierarchy_level=child["level"],
                        parent_id=parent_id,
                        parent_title=parent_title,
                    )
                    chunks.append(Chunk(id=str(uuid.uuid4()), content=child_content, index=idx, metadata=meta))
                    idx += 1
                else:
                    # Child too large – split with base chunker, keep parent context
                    fallback = self.base_chunker.chunk(child_content, base_metadata)
                    for c in fallback:
                        c.index = idx
                        c.metadata["chunk_index"] = idx
                        c.metadata["chunk_strategy"] = "hierarchical"
                        c.metadata["hierarchy_level"] = child["level"]
                        c.metadata["parent_id"] = parent_id
                        c.metadata["parent_title"] = parent_title
                        idx += 1
                    chunks.extend(fallback)

        return chunks


class ISOChunker:
    """Chunk ISO/KS standard documents using section boundaries.

    Preserves section numbers and titles as context for each chunk.
    Falls back to RecursiveChunker for non-ISO content.
    """

    # Section heading pattern
    SECTION_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.base_chunker = RecursiveChunker(chunk_size, chunk_overlap)

    def chunk(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into chunks using ISO section boundaries."""
        if not text or not text.strip():
            return []

        # Check if this is ISO-structured content
        if not self._has_iso_structure(text):
            return self.base_chunker.chunk(text, base_metadata)

        sections = self._parse_sections(text)
        return self._sections_to_chunks(sections, base_metadata)

    def chunk_with_pages(
        self,
        pages: list[dict],
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """Chunk page-by-page or use sections if available."""
        # If pages already have section metadata, use that
        if pages and "section_number" in pages[0]:
            return self._chunk_structured_pages(pages, base_metadata)

        # Otherwise combine and parse
        combined = "\n\n".join(
            p.get("content", "") for p in pages if p.get("content", "").strip()
        )
        return self.chunk(combined, base_metadata)

    def _has_iso_structure(self, text: str) -> bool:
        """Check if text has ISO-style numbered sections."""
        matches = self.SECTION_RE.findall(text[:3000], re.MULTILINE)
        return len(matches) >= 2

    def _parse_sections(self, text: str) -> list[dict]:
        """Parse text into sections."""
        lines = text.split('\n')
        sections = []
        current_section = None
        buffer = []

        for line in lines:
            match = self.SECTION_RE.match(line.strip())
            if match:
                # Save previous section
                if current_section and buffer:
                    current_section["content"] = "\n".join(buffer).strip()
                    if current_section["content"]:
                        sections.append(current_section)

                # Start new section
                section_num, section_title = match.groups()
                current_section = {
                    "section_number": section_num,
                    "section_title": section_title.strip(),
                    "content": "",
                }
                buffer = [line]
            else:
                buffer.append(line)

        # Save last section
        if current_section and buffer:
            current_section["content"] = "\n".join(buffer).strip()
            if current_section["content"]:
                sections.append(current_section)

        return sections

    def _sections_to_chunks(
        self,
        sections: list[dict],
        base_metadata: dict | None,
    ) -> list[Chunk]:
        """Convert sections to chunks."""
        chunks = []
        global_idx = 0

        for section in sections:
            section_num = section["section_number"]
            section_title = section["section_title"]
            content = section["content"]

            # Prepend section context
            content_with_context = f"[{section_num} {section_title}]\n{content}"

            if len(content_with_context) <= self.chunk_size:
                # Single chunk
                meta = dict(base_metadata or {})
                meta.update({
                    "chunk_index": global_idx,
                    "chunk_strategy": "iso",
                    "section_number": section_num,
                    "section_title": section_title,
                })
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        content=content_with_context,
                        index=global_idx,
                        metadata=meta,
                    )
                )
                global_idx += 1
            else:
                # Section too large, split with base chunker
                sub_chunks = self.base_chunker.chunk(content_with_context, base_metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.index = global_idx
                    sub_chunk.metadata["chunk_index"] = global_idx
                    sub_chunk.metadata["chunk_strategy"] = "iso"
                    sub_chunk.metadata["section_number"] = section_num
                    sub_chunk.metadata["section_title"] = section_title
                    global_idx += 1
                chunks.extend(sub_chunks)

        return chunks

    def _chunk_structured_pages(
        self,
        pages: list[dict],
        base_metadata: dict | None,
    ) -> list[Chunk]:
        """Chunk pages that already have section metadata."""
        chunks = []
        global_idx = 0

        for page in pages:
            section_num = page.get("section_number", "")
            section_title = page.get("section_title", "")
            content = page.get("content", "")

            if not content.strip():
                continue

            meta = dict(base_metadata or {})
            meta.update({
                "chunk_index": global_idx,
                "chunk_strategy": "iso",
                "section_number": section_num,
                "section_title": section_title,
            })

            if len(content) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        content=content,
                        index=global_idx,
                        metadata=meta,
                    )
                )
                global_idx += 1
            else:
                # Split large sections
                sub_chunks = self.base_chunker.chunk(content, base_metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.index = global_idx
                    sub_chunk.metadata.update(meta)
                    sub_chunk.metadata["chunk_index"] = global_idx
                    global_idx += 1
                chunks.extend(sub_chunks)

        return chunks


class AdaptiveChunker:
    """Auto-detect document structure and apply the best chunking strategy.

    Strategy selection order:
    1. Hierarchical – if the text contains Korean legal article patterns
       (제N조 appearing >= 2 times).
    2. Table – if > 50 % of lines are pipe- or tab-delimited.
    3. Recursive – general-purpose fallback.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.recursive = RecursiveChunker(chunk_size, chunk_overlap)
        self.table = TableChunker(chunk_size, chunk_overlap)
        self.hierarchical = HierarchicalChunker(chunk_size, chunk_overlap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []

        if self._has_legal_structure(text):
            return self.hierarchical.chunk(text, base_metadata)
        if self._is_mostly_table(text):
            return self.table.chunk(text, base_metadata)
        return self.recursive.chunk(text, base_metadata)

    def chunk_with_pages(
        self,
        pages: list[dict],
        base_metadata: dict | None = None,
    ) -> list[Chunk]:
        """For paged documents, choose strategy based on combined content."""
        combined = "\n\n".join(
            p.get("content", "") for p in pages if p.get("content", "").strip()
        )
        if not combined.strip():
            return []

        # Hierarchical structures span pages – process as single text
        if self._has_legal_structure(combined):
            return self.hierarchical.chunk(combined, base_metadata)

        # Otherwise process page-by-page, choosing table vs recursive per page
        all_chunks: list[Chunk] = []
        global_idx = 0

        for page in pages:
            page_num = page.get("page_num", 0)
            content = page.get("content", "")
            if not content.strip():
                continue

            page_meta = dict(base_metadata or {})
            page_meta["page_number"] = page_num

            if self._is_mostly_table(content):
                page_chunks = self.table.chunk(content, base_metadata=page_meta)
            else:
                page_chunks = self.recursive.chunk(content, base_metadata=page_meta)

            for c in page_chunks:
                c.index = global_idx
                c.metadata["chunk_index"] = global_idx
                global_idx += 1
            all_chunks.extend(page_chunks)

        return all_chunks

    # ------------------------------------------------------------------
    # Detection heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _has_legal_structure(text: str) -> bool:
        """Check if text contains Korean legal article patterns (>= 2)."""
        return len(re.findall(r'^제\d+조', text, re.MULTILINE)) >= 2

    @staticmethod
    def _is_mostly_table(text: str) -> bool:
        """Return True when > 50 % of non-empty lines look tabular."""
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return False
        table_lines = sum(1 for l in lines if l.count('|') >= 2 or l.count('\t') >= 2)
        return table_lines / len(lines) > 0.5


# Backward-compatible alias – now uses adaptive strategy
SemanticChunker = AdaptiveChunker


def create_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveChunker | EmbeddingSemanticChunker | TableChunker | HierarchicalChunker | ISOChunker | AdaptiveChunker:
    """Factory function to create chunker based on settings.

    Args:
        strategy: One of "recursive", "semantic", "table", "hierarchical",
                  "iso", or "adaptive". Defaults to ``settings.chunk_strategy``.
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
    elif strategy == "table":
        return TableChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "hierarchical":
        return HierarchicalChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "iso":
        return ISOChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "adaptive":
        return AdaptiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:  # "recursive" default
        return RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
