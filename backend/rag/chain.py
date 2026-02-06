"""RAG chain: orchestrates retrieval → prompt → LLM → response."""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator

from core.llm import BaseLLM, create_llm
from rag.prompt import PromptManager
from rag.quality import QualityController
from rag.retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG response with sources and metadata."""
    content: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    confidence_level: str = "low"
    safety_warning: str | None = None
    metadata: dict = field(default_factory=dict)


class RAGChain:
    """Full RAG pipeline: retrieve → augment → generate."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: BaseLLM | None = None,
        prompt_manager: PromptManager | None = None,
        quality: QualityController | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or create_llm()
        self.prompt_manager = prompt_manager or PromptManager()
        self.quality = quality or QualityController()

    @staticmethod
    def _make_cache_key(question: str, mode: str, filters: dict | None, provider: str | None, model: str | None) -> str:
        """Build a deterministic cache key for a RAG query."""
        raw = json.dumps(
            {"q": question, "m": mode, "f": filters, "p": provider, "mdl": model},
            sort_keys=True,
            ensure_ascii=False,
        )
        return "rag:query:" + hashlib.md5(raw.encode()).hexdigest()

    async def query(
        self,
        question: str,
        mode: str = "rag",
        filters: dict | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> RAGResponse:
        """Execute a full RAG query.

        Args:
            question: User's question.
            mode: "rag" (search+generate), "direct" (LLM only).
            filters: Metadata filters for retrieval.
            provider: Override LLM provider.
            model: Override LLM model.
            temperature: Override temperature.

        Returns:
            RAGResponse with content, sources, and confidence.
        """
        # Check cache for identical queries (TTL: 60s)
        cache_key = self._make_cache_key(question, mode, filters, provider, model)
        try:
            from core.cache import get_cache

            cache = await get_cache()
            cached = await cache.get(cache_key)
            if cached is not None:
                logger.debug("RAG cache hit for query: %s", question[:50])
                return RAGResponse(**cached)
        except Exception:
            pass

        start_time = time.time()

        # Use override LLM if specified
        llm = self.llm
        if provider or model:
            llm = create_llm(provider=provider, model=model, temperature=temperature or 0.7)

        if mode == "direct":
            response = await self._direct_query(question, llm, start_time)
        else:
            response = await self._rag_query(question, llm, filters, start_time)

        # Store in cache (TTL: 60s)
        try:
            await cache.set(cache_key, asdict(response), ttl=60)
        except Exception:
            pass

        return response

    async def _rag_query(
        self,
        question: str,
        llm: BaseLLM,
        filters: dict | None,
        start_time: float,
    ) -> RAGResponse:
        """RAG mode: retrieve documents then generate."""
        # Step 1: Retrieve
        retrieval_results = await self.retriever.retrieve(
            query=question,
            filters=filters,
        )

        # Step 2: Calculate confidence
        chunk_scores = [r.score for r in retrieval_results]
        confidence = self.quality.calculate_confidence(chunk_scores)

        # Step 3: Build sources
        sources = [
            {
                "chunk_id": r.id,
                "content": r.content,
                "score": round(r.score, 3),
                "metadata": r.metadata,
            }
            for r in retrieval_results
        ]

        # Step 4: Build prompt
        context_chunks = [
            {"content": r.content, "metadata": r.metadata}
            for r in retrieval_results
        ]
        system_prompt, user_prompt = self.prompt_manager.build_rag_prompt(
            query=question,
            context_chunks=context_chunks,
        )

        # Step 5: Generate
        response = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Step 6: Quality assessment
        safety_warning = self.quality.get_safety_message(confidence)
        content = response.content
        if safety_warning:
            content = f"{safety_warning}\n\n제한적인 정보로 추정한 답변:\n{content}"

        metadata = self.quality.build_response_metadata(
            confidence=confidence,
            sources=sources,
            model_used=f"{llm.provider_name}/{llm.model}",
            latency_ms=latency_ms,
            response_mode="rag",
        )

        return RAGResponse(
            content=content,
            sources=sources,
            confidence=confidence,
            confidence_level=self.quality.get_confidence_level(confidence),
            safety_warning=safety_warning,
            metadata=metadata,
        )

    async def _direct_query(
        self,
        question: str,
        llm: BaseLLM,
        start_time: float,
    ) -> RAGResponse:
        """Direct mode: LLM only, no retrieval."""
        system_prompt, user_prompt = self.prompt_manager.build_direct_prompt(question)

        response = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return RAGResponse(
            content=response.content,
            sources=[],
            confidence=1.0,
            confidence_level="high",
            metadata={
                "confidence_score": 1.0,
                "confidence_level": "high",
                "model_used": f"{llm.provider_name}/{llm.model}",
                "latency_ms": latency_ms,
                "response_mode": "direct",
            },
        )

    async def stream_query(
        self,
        question: str,
        mode: str = "rag",
        filters: dict | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[dict]:
        """Stream a RAG query response.

        Yields SSE-compatible events:
        - {"event": "start", "data": {"message_id": ...}}
        - {"event": "routing", "data": {"category": ..., "confidence": ...}}
        - {"event": "source", "data": {source info}}
        - {"event": "chunk", "data": {"content": token}}
        - {"event": "end", "data": {metadata}}
        """
        import uuid
        start_time = time.time()
        message_id = str(uuid.uuid4())

        # Use override LLM if specified
        llm = self.llm
        if provider or model:
            llm = create_llm(provider=provider, model=model)

        yield {"event": "start", "data": {"message_id": message_id}}

        if mode == "direct":
            system, user_prompt = self.prompt_manager.build_direct_prompt(question)
            async for token in llm.stream(prompt=user_prompt, system=system):
                yield {"event": "chunk", "data": {"content": token}}
            latency_ms = int((time.time() - start_time) * 1000)
            yield {"event": "end", "data": {"confidence_score": 1.0, "latency_ms": latency_ms}}
            return

        # RAG mode with streaming
        # Step 1: Retrieve
        results = await self.retriever.retrieve(query=question, filters=filters)
        chunk_scores = [r.score for r in results]
        confidence = self.quality.calculate_confidence(chunk_scores)

        # Emit sources
        for r in results:
            yield {
                "event": "source",
                "data": {
                    "chunk_id": r.id,
                    "filename": r.metadata.get("filename", ""),
                    "page": r.metadata.get("page_number"),
                    "score": round(r.score, 3),
                },
            }

        # Step 2: Build prompt and stream
        context_chunks = [{"content": r.content, "metadata": r.metadata} for r in results]
        system, user_prompt = self.prompt_manager.build_rag_prompt(
            query=question, context_chunks=context_chunks,
        )

        # Safety warning first
        if self.quality.should_add_safety_warning(confidence):
            warning = self.quality.get_safety_message(confidence)
            yield {"event": "chunk", "data": {"content": warning + "\n\n"}}

        # Stream LLM response
        async for token in llm.stream(prompt=user_prompt, system=system):
            yield {"event": "chunk", "data": {"content": token}}

        latency_ms = int((time.time() - start_time) * 1000)
        yield {
            "event": "end",
            "data": {
                "confidence_score": round(confidence, 3),
                "confidence_level": self.quality.get_confidence_level(confidence),
                "latency_ms": latency_ms,
                "source_count": len(results),
            },
        }
