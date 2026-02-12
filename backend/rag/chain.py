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
from config.settings import settings

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
    def _auto_detect_filters(question: str, filters: dict | None) -> dict | None:
        """Auto-detect source_type filter from question keywords.

        Returns enhanced filters dict if a specific source is detected,
        otherwise returns the original filters unchanged.
        """
        if filters and filters.get("source_type"):
            return filters  # User already specified, don't override

        q = question.lower()

        # Law/statute detection patterns
        law_keywords = [
            "한국가스공사법", "가스공사법", "시행령", "시행규칙",
            "법 제", "법률 제", "조문", "부칙",
        ]

        # Internal regulation detection patterns
        rule_keywords = [
            "내부규정", "규정에", "사규", "지침에",
            "인사규정", "감사규정", "보수규정", "여비규정",
            "복무규정", "퇴직금", "경조금", "당직",
            "출자회사", "정관",
        ]

        detected_source = None

        for kw in law_keywords:
            if kw in q:
                detected_source = "법률"
                break

        if detected_source is None:
            for kw in rule_keywords:
                if kw in q:
                    detected_source = "내부규정"
                    break

        if detected_source:
            new_filters = dict(filters) if filters else {}
            new_filters["source_type"] = detected_source
            return new_filters

        return filters

    @staticmethod
    def _make_cache_key(question: str, mode: str, filters: dict | None, provider: str | None, model: str | None) -> str:
        """Build a deterministic cache key for a RAG query."""
        raw = json.dumps(
            {"q": question, "m": mode, "f": filters, "p": provider, "mdl": model},
            sort_keys=True,
            ensure_ascii=False,
        )
        return "rag:query:" + hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _postprocess_answer(text: str) -> str:
        """Strip common LLM output artifacts."""
        import re
        text = text.strip()
        # Strip "Q: ... A:" pattern (LLM repeating the question)
        text = re.sub(r'^Q[:：].*?(?:A[:：])\s*', '', text, flags=re.DOTALL)
        # Strip "A:" or "답변:" prefix
        text = re.sub(r'^(A[:：]|답변[:：])\s*', '', text.strip())
        # Strip [출처: ...] or [출처] tags
        text = re.sub(r'\[출[처처][^\]]*\]', '', text)
        # Strip trailing Chinese/Japanese text (qwen2.5 leakage)
        text = re.sub(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+[^\n]*$', '', text, flags=re.MULTILINE)
        return text.strip()

    async def query(
        self,
        question: str,
        mode: str = "rag",
        filters: dict | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        user_id: str = "",
    ) -> RAGResponse:
        """Execute a full RAG query.

        Args:
            question: User's question.
            mode: "rag" (search+generate), "direct" (LLM only).
            filters: Metadata filters for retrieval.
            provider: Override LLM provider.
            model: Override LLM model.
            temperature: Override temperature.
            user_id: User identifier for guardrails.

        Returns:
            RAGResponse with content, sources, and confidence.
        """
        # Check cache for identical queries (TTL: 60s)
        cache_key = self._make_cache_key(question, mode, filters, provider, model)
        cache = None
        try:
            from core.cache import get_cache

            cache = await get_cache()
            cached = await cache.get(cache_key)
            if cached is not None:
                logger.debug("RAG cache hit for query: %s", question[:50])
                return RAGResponse(**cached)
        except Exception:
            cache = None

        start_time = time.time()

        # Guardrails: input check (applies to both rag and direct modes)
        try:
            from .guardrails import get_guardrails_manager
            guard = await get_guardrails_manager()
            guard_result = await guard.check_input(question, user_id=user_id)
            if not guard_result.passed and guard_result.action == "block":
                return RAGResponse(
                    content=guard_result.message or "입력이 필터링되었습니다.",
                    sources=[],
                    confidence=0.0,
                    confidence_level="blocked",
                    safety_warning=guard_result.message,
                    metadata={"blocked_by": "guardrails", "triggered_rules": guard_result.triggered_rules},
                )
        except Exception as e:
            logger.debug("Guardrails input check skipped: %s", e)

        # Query correction (SFR-006)
        correction_info = None
        try:
            from .query_corrector import get_query_corrector
            corrector = get_query_corrector()
            correction = corrector.correct(question)
            if correction.was_corrected:
                correction_info = {
                    "original": correction.original,
                    "corrected": correction.corrected,
                    "corrections": correction.corrections,
                }
                question = correction.corrected
                logger.info("Query corrected: '%s' -> '%s'", correction.original[:50], correction.corrected[:50])
        except Exception as e:
            logger.debug("Query correction skipped: %s", e)

        # Terminology expansion for search (SFR: 기술용어 사전)
        terminology_info = None
        search_query = question  # Default: use corrected question for search
        try:
            from .terminology import get_terminology_service
            ts = get_terminology_service()
            expansion = ts.expand_query(question)
            if expansion.was_expanded:
                terminology_info = {
                    "matched_terms": expansion.matched_terms,
                    "expansions": expansion.expansions,
                }
                # Use expanded query for retrieval, keep original for LLM
                search_query = expansion.expanded_query
                logger.info("Query expanded with terminology: %d terms matched", len(expansion.matched_terms))
        except Exception as e:
            logger.debug("Terminology expansion skipped: %s", e)

        # Use override LLM if specified
        llm = self.llm
        temp_llm = None
        if provider or model:
            temp_llm = create_llm(provider=provider, model=model, temperature=temperature or settings.llm_temperature)
            llm = temp_llm

        try:
            # Agentic RAG: dynamic strategy routing
            agentic_meta = None
            if settings.agentic_rag_enabled and mode == "rag":
                try:
                    from .agentic import AgenticRAGRouter
                    router = AgenticRAGRouter(llm=llm)
                    decision = await router.route(question)
                    agentic_meta = {
                        "strategy": decision.strategy,
                        "routing_confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "_params": decision.params,
                    }
                    logger.info("Agentic RAG: strategy=%s (confidence=%.2f)", decision.strategy, decision.confidence)

                    # Apply routing decision
                    if decision.params.get("mode") == "direct":
                        mode = "direct"
                    if decision.params.get("temperature") is not None:
                        temperature = temperature if temperature is not None else decision.params["temperature"]
                except Exception as e:
                    logger.warning("Agentic RAG routing failed: %s", e)

            # Determine per-request overrides from agentic routing
            agentic_use_multi_query = False
            agentic_top_k = None
            if agentic_meta:
                agentic_use_multi_query = bool(agentic_meta.get("strategy") == "multi_query_rag")
                params = agentic_meta.get("_params", {})
                if params.get("top_k"):
                    agentic_top_k = params["top_k"]

            # Auto-detect source_type filter from question
            if mode != "direct":
                filters = self._auto_detect_filters(question, filters)
                if filters and filters.get("source_type"):
                    logger.info("Auto-detected source_type filter: %s", filters["source_type"])

            if mode == "direct":
                response = await self._direct_query(question, llm, start_time, user_id, correction_info, temperature)
            else:
                response = await self._rag_query(
                    question, llm, filters, start_time, user_id, correction_info, temperature,
                    force_multi_query=agentic_use_multi_query, override_top_k=agentic_top_k,
                    search_query=search_query, terminology_info=terminology_info,
                )

            # Add agentic routing metadata (exclude internal _params)
            if agentic_meta:
                response.metadata["agentic_routing"] = {
                    k: v for k, v in agentic_meta.items() if not k.startswith("_")
                }

            # Store in cache (TTL: 60s)
            if cache is not None:
                try:
                    await cache.set(cache_key, asdict(response), ttl=60)
                except Exception:
                    pass

            return response
        finally:
            if temp_llm is not None:
                await temp_llm.close()

    async def _rag_query(
        self,
        question: str,
        llm: BaseLLM,
        filters: dict | None,
        start_time: float,
        user_id: str = "",
        correction_info: dict | None = None,
        temperature: float | None = None,
        force_multi_query: bool = False,
        override_top_k: int | None = None,
        search_query: str | None = None,
        terminology_info: dict | None = None,
    ) -> RAGResponse:
        """RAG mode: retrieve documents then generate."""
        # Use expanded search query if provided, else fall back to question
        search_query = search_query or question

        # Step 0+1: Retrieve (with optional HyDE and Multi-Query)
        hyde_used = False
        multi_query_used = False
        hyde_embedding = None

        # Determine effective top_k (don't mutate shared retriever state)
        effective_top_k = override_top_k or None

        # Generate HyDE embedding if enabled
        if settings.query_expansion_enabled:
            try:
                from .query_expander import QueryExpander
                expander = QueryExpander(llm=llm)
                hyde_embedding = await expander.expand_hyde(question)
                hyde_used = True
            except Exception as e:
                logger.warning("HyDE expansion failed: %s", e)

        # Use Multi-Query retrieval if enabled (settings or agentic routing)
        use_multi_query = settings.multi_query_enabled or force_multi_query
        if use_multi_query:
            try:
                from .multi_query import MultiQueryRetriever
                mq_retriever = MultiQueryRetriever(
                    retriever=self.retriever,
                    llm=llm,
                    query_count=settings.multi_query_count,
                )
                retrieval_results = await mq_retriever.retrieve(
                    query=search_query,
                    top_k=effective_top_k,
                    filters=filters,
                    hyde_embedding=hyde_embedding,
                )
                multi_query_used = True
            except Exception as e:
                logger.warning("Multi-query retrieval failed, falling back: %s", e)

        # Standard retrieval (fallback)
        if not multi_query_used:
            retrieval_results = await self.retriever.retrieve(
                query=search_query,
                top_k=effective_top_k,
                filters=filters,
                hyde_embedding=hyde_embedding if hyde_used else None,
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

        # Limit context chunks for LLM (if configured)
        if settings.context_max_chunks > 0:
            context_chunks = context_chunks[:settings.context_max_chunks]

        # Determine model name for model-aware prompting
        model_name = getattr(llm, 'model', None)

        system_prompt, user_prompt = self.prompt_manager.build_rag_prompt(
            query=question,
            context_chunks=context_chunks,
            model_hint=model_name,
        )

        # Step 5: Generate
        response = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Step 6: Quality assessment
        safety_warning = self.quality.get_safety_message(confidence)
        content = response.content
        postprocessed = self._postprocess_answer(content)
        content = postprocessed if postprocessed else content.strip()

        # Self-RAG: evaluate groundedness and retry if needed
        self_rag_meta = None
        if settings.self_rag_enabled:
            try:
                from .self_rag import SelfRAGEvaluator
                # Build context string for grading
                context_str = "\n\n".join(c["content"] for c in context_chunks)
                self_rag = SelfRAGEvaluator(
                    llm=llm,
                    max_retries=settings.self_rag_max_retries,
                )
                content, self_rag_meta = await self_rag.evaluate_and_retry(
                    query=question,
                    answer=content,
                    context=context_str,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    llm=llm,
                    temperature=temperature,
                    max_tokens=settings.llm_max_tokens,
                )
            except Exception as e:
                logger.warning("Self-RAG evaluation failed: %s", e)

        # Guardrails: output check
        try:
            from .guardrails import get_guardrails_manager
            guard = await get_guardrails_manager()
            out_result = await guard.check_output(content, user_id=user_id)
            if not out_result.passed and out_result.action == "block":
                content = out_result.message or "응답이 안전 필터에 의해 차단되었습니다."
            elif out_result.modified_text:
                content = out_result.modified_text
        except Exception as e:
            logger.debug("Guardrails output check skipped: %s", e)

        metadata = self.quality.build_response_metadata(
            confidence=confidence,
            sources=sources,
            model_used=f"{llm.provider_name}/{llm.model}",
            latency_ms=latency_ms,
            response_mode="rag",
        )

        # Add safety warning to metadata
        if safety_warning:
            metadata["safety_warning"] = safety_warning

        # Add query correction info if present
        if correction_info:
            metadata["query_correction"] = correction_info

        # Add terminology expansion info if present
        if terminology_info:
            metadata["terminology_expansion"] = terminology_info

        # Add HyDE info if used
        if hyde_used:
            metadata["hyde_used"] = True

        if multi_query_used:
            metadata["multi_query_used"] = True

        if self_rag_meta:
            metadata["self_rag"] = self_rag_meta

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
        user_id: str = "",
        correction_info: dict | None = None,
        temperature: float | None = None,
    ) -> RAGResponse:
        """Direct mode: LLM only, no retrieval."""
        system_prompt, user_prompt = self.prompt_manager.build_direct_prompt(question)

        response = await llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
        )

        content = response.content
        postprocessed = self._postprocess_answer(content)
        content = postprocessed if postprocessed else content.strip()
        # Guardrails: output check
        try:
            from .guardrails import get_guardrails_manager
            guard = await get_guardrails_manager()
            out_result = await guard.check_output(content, user_id=user_id)
            if not out_result.passed and out_result.action == "block":
                content = out_result.message or "응답이 안전 필터에 의해 차단되었습니다."
            elif out_result.modified_text:
                content = out_result.modified_text
        except Exception as e:
            logger.debug("Guardrails output check skipped: %s", e)

        latency_ms = int((time.time() - start_time) * 1000)

        metadata = {
            "confidence_score": 0.5,
            "confidence_level": "medium",
            "model_used": f"{llm.provider_name}/{llm.model}",
            "latency_ms": latency_ms,
            "response_mode": "direct",
            "note": "검색 근거 없이 LLM만으로 생성된 답변입니다.",
        }

        # Add query correction info if present
        if correction_info:
            metadata["query_correction"] = correction_info

        return RAGResponse(
            content=content,
            sources=[],
            confidence=0.5,
            confidence_level="medium",
            safety_warning="이 답변은 문서 검색 없이 LLM만으로 생성되었습니다. 정확도를 직접 확인해주세요.",
            metadata=metadata,
        )

    async def stream_query(
        self,
        question: str,
        mode: str = "rag",
        filters: dict | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        user_id: str = "",
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
        temp_llm = None
        if provider or model:
            temp_llm = create_llm(provider=provider, model=model, temperature=temperature or settings.llm_temperature)
            llm = temp_llm

        try:
            yield {"event": "start", "data": {"message_id": message_id}}

            # Guardrails: input check
            try:
                from .guardrails import get_guardrails_manager
                guard = await get_guardrails_manager()
                guard_result = await guard.check_input(question, user_id=user_id)
                if not guard_result.passed and guard_result.action == "block":
                    yield {
                        "event": "chunk",
                        "data": {"content": guard_result.message or "입력이 필터링되었습니다."},
                    }
                    yield {
                        "event": "end",
                        "data": {
                            "blocked_by": "guardrails",
                            "triggered_rules": guard_result.triggered_rules,
                        },
                    }
                    return
            except Exception as e:
                logger.debug("Guardrails input check skipped: %s", e)

            if mode == "direct":
                system, user_prompt = self.prompt_manager.build_direct_prompt(question)
                # Note: Output guardrails not applied in streaming mode (tokens streamed one-by-one)
                async for token in llm.stream(prompt=user_prompt, system=system, temperature=temperature, max_tokens=settings.llm_max_tokens):
                    yield {"event": "chunk", "data": {"content": token}}
                latency_ms = int((time.time() - start_time) * 1000)
                yield {"event": "end", "data": {"confidence_score": 0.5, "confidence_level": "medium", "latency_ms": latency_ms, "note": "검색 근거 없이 LLM만으로 생성된 답변"}}
                return

            # RAG mode with streaming
            # Step 0: HyDE query expansion (optional)
            hyde_embedding = None
            if settings.query_expansion_enabled:
                try:
                    from .query_expander import QueryExpander
                    expander = QueryExpander(llm=llm)
                    hyde_embedding = await expander.expand_hyde(question)
                except Exception as e:
                    logger.warning("HyDE expansion failed in streaming: %s", e)

            # Step 1: Retrieve
            results = await self.retriever.retrieve(query=question, filters=filters, hyde_embedding=hyde_embedding)
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
            # Limit context chunks for LLM (if configured)
            if settings.context_max_chunks > 0:
                context_chunks = context_chunks[:settings.context_max_chunks]
            model_name = getattr(llm, 'model', None)
            system, user_prompt = self.prompt_manager.build_rag_prompt(
                query=question, context_chunks=context_chunks,
                model_hint=model_name,
            )

            # Safety warning first
            if self.quality.should_add_safety_warning(confidence):
                warning = self.quality.get_safety_message(confidence)
                yield {"event": "chunk", "data": {"content": warning + "\n\n"}}

            # Stream LLM response
            # Note: Output guardrails not applied in streaming mode (tokens streamed one-by-one)
            async for token in llm.stream(prompt=user_prompt, system=system, temperature=temperature, max_tokens=settings.llm_max_tokens):
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
        finally:
            if temp_llm is not None:
                await temp_llm.close()
