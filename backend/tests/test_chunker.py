"""Tests for semantic chunker."""

import pytest

from pipeline.chunker import SemanticChunker, Chunk
from pipeline.metadata import MetadataExtractor


class TestSemanticChunker:

    def setup_method(self):
        self.chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)

    def test_basic_chunking(self):
        text = "가스 안전 관리에 대한 문서입니다. " * 50
        chunks = self.chunker.chunk(text)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_text(self):
        chunks = self.chunker.chunk("")
        assert chunks == []

    def test_short_text(self):
        chunks = self.chunker.chunk("짧은 텍스트")
        assert len(chunks) == 1

    def test_metadata_propagation(self):
        text = "테스트 문서 내용입니다. " * 50
        chunks = self.chunker.chunk(text, base_metadata={"department": "안전팀"})
        for chunk in chunks:
            assert chunk.metadata["department"] == "안전팀"
            assert "chunk_index" in chunk.metadata

    def test_chunk_ids_unique(self):
        text = "반복 텍스트입니다. " * 100
        chunks = self.chunker.chunk(text)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_with_pages(self):
        pages = [
            {"page_num": 1, "content": "첫째 페이지 내용 " * 30},
            {"page_num": 2, "content": "둘째 페이지 내용 " * 30},
        ]
        chunks = self.chunker.chunk_with_pages(pages)
        assert len(chunks) > 0
        # Check page numbers in metadata
        page_nums = {c.metadata.get("page_number") for c in chunks}
        assert 1 in page_nums or 2 in page_nums

    def test_table_detection(self):
        text = "이름 | 나이 | 부서\n홍길동 | 30 | 안전팀\n김철수 | 25 | 기술팀"
        chunks = self.chunker.chunk(text)
        assert chunks[0].metadata.get("has_table") is True


class TestMetadataExtractor:

    def setup_method(self):
        self.extractor = MetadataExtractor()

    def test_department_from_filename(self):
        meta = self.extractor.extract("", filename="안전팀_점검보고서_2024.pdf")
        assert meta.get("department") == "안전팀"

    def test_category_from_filename(self):
        meta = self.extractor.extract("", filename="가스배관_매뉴얼_v3.pdf")
        assert meta.get("category") == "매뉴얼"

    def test_date_from_filename(self):
        meta = self.extractor.extract("", filename="월간보고서_2024-01-15.pdf")
        assert meta.get("document_date") == "2024-01-15"

    def test_department_from_text(self):
        text = "본 문서는 기술연구소에서 작성한 점검 보고서입니다."
        meta = self.extractor.extract(text)
        assert meta.get("department") == "기술연구소"

    def test_tags_from_text(self):
        text = "가스 배관 안전 점검을 실시하였습니다."
        meta = self.extractor.extract(text)
        tags = meta.get("tags", [])
        assert "가스" in tags or "안전" in tags or "점검" in tags
