"""Multi-Query retrieval for improved recall.

Generates alternative queries from different perspectives,
runs retrieval for each, and merges/deduplicates results.
"""

import json
import logging
from typing import Any

from core.llm import BaseLLM, create_llm
from rag.prompt import PromptManager
from rag.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """Generate multiple query variants and merge retrieval results."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: BaseLLM | None = None,
        prompt_manager: PromptManager | None = None,
        query_count: int = 3,
    ):
        self._retriever = retriever
        self._llm = llm
        self._prompt_manager = prompt_manager
        self.query_count = query_count

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager()
        return self._prompt_manager

    async def generate_queries(self, query: str) -> list[str]:
        """Generate alternative queries from different perspectives.

        Args:
            query: Original user question.

        Returns:
            List of alternative queries (original not included; caller prepends it).
        """
        prompt = self.prompt_manager.get_system_prompt("multi_query_system").format(
            query=query,
            count=self.query_count,
        )

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=512,
            )

            # Parse JSON array from response
            content = response.content.strip()
            # Extract JSON array if wrapped in other text
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                queries = json.loads(content[start:end])
                if isinstance(queries, list):
                    # Filter valid strings and limit count
                    alt_queries = [q for q in queries if isinstance(q, str) and q.strip()]
                    alt_queries = alt_queries[:self.query_count]
                    logger.info(
                        "Multi-query generated %d alternatives for: %s",
                        len(alt_queries), query[:50],
                    )
                    return alt_queries
        except Exception as e:
            logger.warning("Multi-query generation failed: %s", e)

        return []

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
        hyde_embedding: list[float] | None = None,
        rerank_top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using multiple query variants.

        Process:
        1. Generate alternative queries
        2. Run retrieval for original + alternatives
        3. Merge and deduplicate by chunk ID
        4. Re-score by frequency and max score

        Args:
            query: Original user question.
            filters: Metadata filters.
            hyde_embedding: Optional HyDE embedding for original query.

        Returns:
            Merged and re-ranked results.
        """
        import asyncio

        # Generate alternative queries
        alt_queries = await self.generate_queries(query)

        # Always include original query
        all_queries = [query] + alt_queries

        # Run retrieval for all queries in parallel
        tasks = []
        for i, q in enumerate(all_queries):
            # Only use hyde_embedding for the original query
            emb = hyde_embedding if i == 0 else None
            tasks.append(self.retriever.retrieve(query=q, top_k=top_k, filters=filters, hyde_embedding=emb, rerank_top_n=rerank_top_n))

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge and deduplicate
        merged = self._merge_multi_results(all_results)

        logger.info(
            "Multi-query retrieved %d unique chunks from %d queries",
            len(merged), len(all_queries),
        )

        return merged

    @staticmethod
    def _merge_multi_results(
        result_sets: list[list[RetrievalResult] | BaseException],
    ) -> list[RetrievalResult]:
        """Merge results from multiple queries.

        Scoring strategy:
        - Each chunk gets the maximum score across all queries
        - Chunks found by multiple queries get a frequency bonus
        """
        chunk_map: dict[str, RetrievalResult] = {}
        chunk_frequency: dict[str, int] = {}

        for result_set in result_sets:
            if isinstance(result_set, BaseException):
                continue
            for r in result_set:
                chunk_frequency[r.id] = chunk_frequency.get(r.id, 0) + 1
                if r.id not in chunk_map or r.score > chunk_map[r.id].score:
                    chunk_map[r.id] = r

        # Apply frequency bonus (chunks found by multiple queries are boosted)
        results = []
        for chunk_id, result in chunk_map.items():
            freq = chunk_frequency.get(chunk_id, 1)
            # Frequency bonus: +5% for each additional query that found this chunk
            bonus = (freq - 1) * 0.05
            result.score = min(1.0, result.score + bonus)
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results
