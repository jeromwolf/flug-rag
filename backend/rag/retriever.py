"""Hybrid retriever combining vector search and BM25 keyword search with re-ranking."""

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field

from rank_bm25 import BM25Okapi

from config.settings import settings
from core.embeddings import BaseEmbedding, create_embedder
from core.vectorstore import BaseVectorStore, SearchResult, create_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Enhanced search result with combined scoring."""
    id: str
    content: str
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """Combines vector similarity search with BM25 keyword search."""

    def __init__(
        self,
        vectorstore: BaseVectorStore | None = None,
        embedder: BaseEmbedding | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ):
        self.vectorstore = vectorstore or create_vectorstore()
        self.embedder = embedder or create_embedder()
        self.vector_weight = vector_weight or settings.vector_weight
        self.bm25_weight = bm25_weight or settings.bm25_weight
        self.top_k = top_k or settings.retrieval_top_k
        self.rerank_top_n = rerank_top_n or settings.rerank_top_n
        self._reranker = None
        self._bm25_corpus: list[dict] | None = None
        self._bm25_index: BM25Okapi | None = None

    @staticmethod
    def _retrieval_cache_key(query: str, top_k: int | None, filters: dict | None) -> str:
        """Build a cache key for retrieval results."""
        raw = json.dumps({"q": query, "k": top_k, "f": filters}, sort_keys=True, ensure_ascii=False)
        return "retriever:search:" + hashlib.md5(raw.encode()).hexdigest()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
        use_rerank: bool = True,
    ) -> list[RetrievalResult]:
        """Execute hybrid search and optional re-ranking.

        Args:
            query: Search query text.
            top_k: Number of results to return (before rerank).
            filters: Metadata filters for vector search.
            use_rerank: Whether to apply re-ranking.

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        top_k = top_k or self.top_k

        # Check cache for identical query+filters (TTL: 120s)
        cache_key = self._retrieval_cache_key(query, top_k, filters)
        try:
            from core.cache import get_cache

            cache = await get_cache()
            cached = await cache.get(cache_key)
            if cached is not None:
                logger.debug("Retriever cache hit for query: %s", query[:50])
                return [RetrievalResult(**r) for r in cached]
        except Exception:
            pass

        # Run vector search and BM25 in parallel
        vector_results, bm25_results = await asyncio.gather(
            self._vector_search(query, top_k, filters),
            self._bm25_search(query, top_k),
        )

        # Combine results
        combined = self._merge_results(vector_results, bm25_results)

        # Re-rank if enabled
        if use_rerank and combined:
            combined = await self._rerank(query, combined)

        # Return top N
        results = combined[: self.rerank_top_n if use_rerank else top_k]

        # Store in cache (TTL: 120s)
        try:
            await cache.set(cache_key, [asdict(r) for r in results], ttl=120)
        except Exception:
            pass

        return results

    async def _vector_search(
        self, query: str, top_k: int, filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        query_embedding = await self.embedder.embed_query(query)
        return await self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

    async def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Search by BM25 keyword matching."""
        if self._bm25_index is None:
            await self._build_bm25_index()

        if self._bm25_index is None or self._bm25_corpus is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._bm25_corpus[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "score": float(scores[idx]),
                    "metadata": doc.get("metadata", {}),
                })

        return results

    async def _build_bm25_index(self):
        """Build BM25 index from all documents in vectorstore."""
        # Get all documents from ChromaDB
        count = await self.vectorstore.count()
        if count == 0:
            return

        # ChromaDB peek to get all docs (for BM25 index)
        collection = self.vectorstore._collection
        all_docs = await asyncio.to_thread(
            collection.get,
            include=["documents", "metadatas"],
        )

        if not all_docs["ids"]:
            return

        self._bm25_corpus = []
        tokenized_corpus = []

        for i, doc_id in enumerate(all_docs["ids"]):
            content = all_docs["documents"][i] if all_docs["documents"] else ""
            metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
            self._bm25_corpus.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
            })
            tokenized_corpus.append(self._tokenize(content))

        self._bm25_index = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + character tokenization for Korean text."""
        # Basic tokenization: split on whitespace and punctuation
        import re
        tokens = re.findall(r'[\w]+', text.lower())
        return tokens

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[dict],
    ) -> list[RetrievalResult]:
        """Merge vector and BM25 results with weighted scoring."""
        merged: dict[str, RetrievalResult] = {}

        # Normalize vector scores
        max_vector = max((r.score for r in vector_results), default=1.0) or 1.0
        for r in vector_results:
            norm_score = r.score / max_vector
            merged[r.id] = RetrievalResult(
                id=r.id,
                content=r.content,
                score=norm_score * self.vector_weight,
                vector_score=norm_score,
                metadata=r.metadata,
            )

        # Normalize BM25 scores
        max_bm25 = max((r["score"] for r in bm25_results), default=1.0) or 1.0
        for r in bm25_results:
            norm_score = r["score"] / max_bm25
            if r["id"] in merged:
                merged[r["id"]].bm25_score = norm_score
                merged[r["id"]].score += norm_score * self.bm25_weight
            else:
                merged[r["id"]] = RetrievalResult(
                    id=r["id"],
                    content=r["content"],
                    score=norm_score * self.bm25_weight,
                    bm25_score=norm_score,
                    metadata=r.get("metadata", {}),
                )

        # Sort by combined score
        results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return results

    async def _rerank(
        self, query: str, results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-rank results using cross-encoder model."""
        if not results:
            return results

        reranker = self._get_reranker()
        if reranker is None:
            return results

        # Prepare pairs for re-ranking
        passages = [{"id": r.id, "text": r.content} for r in results]

        # flashrank uses RerankRequest object
        from flashrank import RerankRequest
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = await asyncio.to_thread(
            reranker.rerank,
            rerank_request,
        )

        # Map back to RetrievalResult
        # flashrank returns: [{'id': ..., 'text': ..., 'score': ...}, ...]
        result_map = {r.id: r for r in results}
        reranked_results = []

        for item in reranked:
            doc_id = item["id"]
            if doc_id in result_map:
                result = result_map[doc_id]
                result.rerank_score = float(item["score"])
                result.score = float(item["score"])  # Override with rerank score
                reranked_results.append(result)

        return reranked_results

    def _get_reranker(self):
        """Lazy-load flashrank reranker."""
        if self._reranker is None:
            try:
                from flashrank import Ranker
                self._reranker = Ranker(model_name="ms-marco-MultiBERT-L-12")
            except ImportError:
                return None
        return self._reranker

    def invalidate_bm25_cache(self):
        """Invalidate BM25 index (call after document changes)."""
        self._bm25_index = None
        self._bm25_corpus = None
