"""Unit tests for vector store."""

import pytest

from core.vectorstore.chroma import ChromaStore
from core.vectorstore.base import SearchResult


@pytest.fixture
def chroma_store(tmp_path):
    """Create a temporary ChromaDB store for testing."""
    return ChromaStore(
        persist_dir=str(tmp_path / "test_chroma"),
        collection_name="test_collection",
    )


class TestChromaStore:
    """Tests for ChromaDB vector store."""

    @pytest.mark.asyncio
    async def test_add_and_count(self, chroma_store):
        await chroma_store.add(
            ids=["doc1", "doc2"],
            texts=["Hello world", "Test document"],
            embeddings=[[0.1] * 1024, [0.2] * 1024],
            metadatas=[{"source": "test1"}, {"source": "test2"}],
        )
        count = await chroma_store.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_search(self, chroma_store):
        await chroma_store.add(
            ids=["doc1", "doc2", "doc3"],
            texts=["가스 안전 규정", "배관 점검 매뉴얼", "예산 보고서"],
            embeddings=[[0.9, 0.1] + [0.0] * 1022, [0.8, 0.2] + [0.0] * 1022, [0.1, 0.9] + [0.0] * 1022],
            metadatas=[
                {"department": "안전팀"},
                {"department": "기술팀"},
                {"department": "재무팀"},
            ],
        )

        results = await chroma_store.search(
            query_embedding=[0.85, 0.15] + [0.0] * 1022,
            top_k=2,
        )
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_search_with_filter(self, chroma_store):
        await chroma_store.add(
            ids=["doc1", "doc2"],
            texts=["가스 안전", "예산 보고"],
            embeddings=[[0.5] * 1024, [0.5] * 1024],
            metadatas=[{"department": "안전팀"}, {"department": "재무팀"}],
        )

        results = await chroma_store.search(
            query_embedding=[0.5] * 1024,
            top_k=10,
            filters={"department": "안전팀"},
        )
        assert len(results) == 1
        assert results[0].metadata["department"] == "안전팀"

    @pytest.mark.asyncio
    async def test_delete(self, chroma_store):
        await chroma_store.add(
            ids=["doc1", "doc2"],
            texts=["A", "B"],
            embeddings=[[0.1] * 1024, [0.2] * 1024],
        )
        await chroma_store.delete(ids=["doc1"])
        count = await chroma_store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_get(self, chroma_store):
        await chroma_store.add(
            ids=["doc1"],
            texts=["Test content"],
            embeddings=[[0.1] * 1024],
            metadatas=[{"key": "value"}],
        )
        items = await chroma_store.get(ids=["doc1"])
        assert len(items) == 1
        assert items[0]["content"] == "Test content"
        assert items[0]["metadata"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_clear(self, chroma_store):
        await chroma_store.add(
            ids=["doc1", "doc2"],
            texts=["A", "B"],
            embeddings=[[0.1] * 1024, [0.2] * 1024],
        )
        await chroma_store.clear()
        count = await chroma_store.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_collection_info(self, chroma_store):
        info = chroma_store.get_collection_info()
        assert info["name"] == "test_collection"
        assert info["count"] == 0
