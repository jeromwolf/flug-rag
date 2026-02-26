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
        score_threshold: float | None = None,
        use_rerank: bool | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
    ):
        self.vectorstore = vectorstore or create_vectorstore()
        self.embedder = embedder or create_embedder()
        self.vector_weight = vector_weight if vector_weight is not None else settings.vector_weight
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
        self.top_k = top_k or settings.retrieval_top_k
        self.rerank_top_n = rerank_top_n or settings.rerank_top_n
        self.score_threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
        self.use_rerank = use_rerank if use_rerank is not None else settings.use_rerank
        self.bm25_k1 = bm25_k1 if bm25_k1 is not None else settings.bm25_k1
        self.bm25_b = bm25_b if bm25_b is not None else settings.bm25_b
        self._reranker = None
        self._kiwi = None

    def _retrieval_cache_key(self, query: str, top_k: int | None, filters: dict | None, hyde: bool = False) -> str:
        """Build a cache key for retrieval results."""
        raw = json.dumps({
            "q": query, "k": top_k, "f": filters,
            "st": self.score_threshold, "rr": self.use_rerank,
            "k1": self.bm25_k1, "b": self.bm25_b, "hyde": hyde,
        }, sort_keys=True, ensure_ascii=False)
        return "retriever:search:" + hashlib.md5(raw.encode()).hexdigest()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
        use_rerank: bool | None = None,
        hyde_embedding: list[float] | None = None,
        rerank_top_n: int | None = None,
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
        use_rerank = use_rerank if use_rerank is not None else self.use_rerank

        # Check cache for identical query+filters (TTL: 120s)
        cache = None
        cache_key = self._retrieval_cache_key(query, top_k, filters, hyde=hyde_embedding is not None)
        try:
            from core.cache import get_cache

            cache = await get_cache()
            cached = await cache.get(cache_key)
            if cached is not None:
                logger.debug("Retriever cache hit for query: %s", query[:50])
                return [RetrievalResult(**r) for r in cached]
        except Exception:
            pass

        # Stage 1: Vector search with expanded pool for BM25 re-scoring
        expanded_k = top_k * 3  # Fetch 3x candidates for BM25 scoring
        vector_results = await self._vector_search(query, expanded_k, filters, hyde_embedding=hyde_embedding)

        # Stage 2: BM25 score the vector candidates (no full-corpus load)
        bm25_results = await asyncio.to_thread(self._bm25_score_candidates, query, vector_results)

        # Combine results
        combined = self._merge_results(vector_results[:top_k], bm25_results)

        # Re-rank if enabled
        if use_rerank and combined:
            combined = await self._rerank(query, combined)

        # Return top N
        effective_rerank_n = rerank_top_n or self.rerank_top_n
        results = combined[: effective_rerank_n if use_rerank else top_k]

        # Apply score threshold filtering
        if self.score_threshold > 0:
            results = [r for r in results if r.score >= self.score_threshold]

        # Store in cache (TTL: 120s)
        if cache is not None:
            try:
                await cache.set(cache_key, [asdict(r) for r in results], ttl=120)
            except Exception:
                pass

        return results

    async def _vector_search(
        self, query: str, top_k: int, filters: dict | None = None,
        hyde_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        if hyde_embedding is not None:
            query_embedding = hyde_embedding
        else:
            query_embedding = await self.embedder.embed_query(query)
        return await self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

    def _bm25_score_candidates(self, query: str, candidates: list[SearchResult]) -> list[dict]:
        """Score pre-fetched candidates with BM25 (no full-corpus index needed)."""
        if not candidates:
            return []

        tokenized_query = self._tokenize(query)

        # Build a small BM25 index from candidates only
        corpus_tokens = [self._tokenize(c.content) for c in candidates]
        if not corpus_tokens or not tokenized_query:
            return []

        bm25 = BM25Okapi(corpus_tokens, k1=self.bm25_k1, b=self.bm25_b)
        scores = bm25.get_scores(tokenized_query)

        results = []
        for i, candidate in enumerate(candidates):
            if scores[i] > 0:
                results.append({
                    "id": candidate.id,
                    "content": candidate.content,
                    "score": float(scores[i]),
                    "metadata": candidate.metadata,
                })

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Korean morphological tokenization using kiwipiepy."""
        if self._kiwi is None:
            try:
                from kiwipiepy import Kiwi
                self._kiwi = Kiwi()
            except ImportError:
                import re
                return re.findall(r'[\w]+', text.lower())
        tokens = []
        for token in self._kiwi.tokenize(text):
            # 의미 있는 형태소만: 명사, 동사, 형용사, 어근, 숫자, 외국어
            if token.tag.startswith(('NN', 'VV', 'VA', 'XR', 'SN', 'SL')):
                tokens.append(token.form.lower())
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

        # Prepare (query, passage) pairs for re-ranking
        pairs = [(query, r.content) for r in results]

        # Get scores from CrossEncoder
        scores = await asyncio.to_thread(reranker.predict, pairs)

        # Update results with rerank scores and sort by score descending
        for i, result in enumerate(results):
            score = float(scores[i])
            result.rerank_score = score
            result.score = score  # Override with rerank score

        # Sort by rerank score descending
        reranked_results = sorted(results, key=lambda r: r.score, reverse=True)

        return reranked_results

    def _get_reranker(self):
        """Lazy-load sentence-transformers CrossEncoder reranker."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            except ImportError:
                return None
        return self._reranker

