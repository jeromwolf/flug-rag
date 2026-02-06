"""Unit tests for hybrid retriever."""

import pytest

from core.vectorstore.chroma import ChromaStore
from rag.retriever import HybridRetriever, RetrievalResult


@pytest.fixture
def chroma_store(tmp_path):
    return ChromaStore(
        persist_dir=str(tmp_path / "test_chroma"),
        collection_name="test_retriever",
    )


@pytest.fixture
def retriever(chroma_store):
    return HybridRetriever(
        vectorstore=chroma_store,
        vector_weight=0.7,
        bm25_weight=0.3,
        top_k=10,
        rerank_top_n=5,
    )


class TestHybridRetriever:

    @pytest.mark.asyncio
    async def test_vector_search_empty_store(self, retriever):
        """Should return empty when no documents."""
        # Mock embedder to avoid loading model
        class MockEmbedder:
            async def embed_query(self, text):
                return [0.1] * 1024
        retriever.embedder = MockEmbedder()

        results = await retriever.retrieve("test query", use_rerank=False)
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_with_docs(self, retriever, chroma_store):
        """Should find relevant documents."""
        # Add documents
        await chroma_store.add(
            ids=["doc1", "doc2", "doc3"],
            texts=[
                "가스 배관 안전 점검 매뉴얼",
                "예산 편성 기준 문서",
                "비상 대응 절차서",
            ],
            embeddings=[
                [0.9, 0.1] + [0.0] * 1022,
                [0.1, 0.9] + [0.0] * 1022,
                [0.5, 0.5] + [0.0] * 1022,
            ],
            metadatas=[
                {"department": "안전팀"},
                {"department": "재무팀"},
                {"department": "안전팀"},
            ],
        )

        class MockEmbedder:
            async def embed_query(self, text):
                return [0.85, 0.15] + [0.0] * 1022

        retriever.embedder = MockEmbedder()
        results = await retriever.retrieve("가스 안전 점검", use_rerank=False)

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    @pytest.mark.asyncio
    async def test_merge_results(self, retriever):
        """Should merge vector and BM25 results correctly."""
        from core.vectorstore.base import SearchResult

        vector = [
            SearchResult(id="a", content="text a", score=0.9, metadata={}),
            SearchResult(id="b", content="text b", score=0.7, metadata={}),
        ]
        bm25 = [
            {"id": "b", "content": "text b", "score": 2.0, "metadata": {}},
            {"id": "c", "content": "text c", "score": 1.5, "metadata": {}},
        ]

        merged = retriever._merge_results(vector, bm25)
        ids = [r.id for r in merged]

        # "b" should rank high (appears in both)
        assert "b" in ids
        assert "a" in ids
        assert "c" in ids

    def test_tokenize(self, retriever):
        text = "가스 배관 안전 점검 2024년"
        tokens = retriever._tokenize(text)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_invalidate_cache(self, retriever):
        retriever._bm25_index = "something"
        retriever._bm25_corpus = ["data"]
        retriever.invalidate_bm25_cache()
        assert retriever._bm25_index is None
        assert retriever._bm25_corpus is None
