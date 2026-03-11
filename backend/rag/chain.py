"""RAG chain: orchestrates retrieval → prompt → LLM → response."""

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import AsyncIterator

import yaml

from core.llm import BaseLLM, create_llm
from core.llm.factory import get_default_llm
from rag.prompt import PromptManager
from rag.quality import QualityController
from rag.retriever import HybridRetriever, RetrievalResult
from config.settings import settings

logger = logging.getLogger(__name__)

# Precompiled regex patterns for postprocessing (avoid per-call compilation)
_RE_QA_PATTERN = re.compile(r'^Q[:：].*?(?:A[:：])\s*', re.DOTALL)
_RE_A_PREFIX = re.compile(r'^(A[:：]|답변[:：])\s*')
_RE_SOURCE_TAG = re.compile(r'\[출[처처][^\]]*\]')
_RE_CJK_LEAKAGE = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]+[^\n]*$', re.MULTILINE)
_RE_WHITESPACE = re.compile(r'[\s\-\n•·]')
_RE_CHINESE = re.compile(r'[\u4e00-\u9fff]')
# Strip Chinese/Japanese chars from streaming tokens (preserves Korean Hangul U+AC00-D7AF)
_RE_STRIP_CJK = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]')
# Form template detection: 별지 서식, checkboxes, form fields
_RE_FORM_TEMPLATE = re.compile(
    r'별지\s*(?:제?\s*\d+호?\s*)?서식|'
    r'[□☐■☑✓✔]{2,}|'           # 2+ consecutive checkboxes
    r'접수번호\s*[:：]|'
    r'처리기간\s*[:：]|'
    r'작\s*성\s*일\s*[:：]|'
    r'신\s*고\s*인|신\s*고\s*자|'
    r'기\s*관\s*명\s*[:：]|'
    r'담당부서\s*[:：]'
)


_source_filters_cache: dict | None = None


def _load_source_filters() -> dict:
    """Load source filter keywords from YAML (cached after first load)."""
    global _source_filters_cache
    if _source_filters_cache is None:
        yaml_path = Path(__file__).parent.parent / "prompts" / "source_filters.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                _source_filters_cache = yaml.safe_load(f)
        else:
            logger.warning("source_filters.yaml not found at %s; keyword routing disabled", yaml_path)
            _source_filters_cache = {}
    return _source_filters_cache


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
        self.llm = llm or get_default_llm()
        self.prompt_manager = prompt_manager or PromptManager()
        self.quality = quality or QualityController()

    @staticmethod
    def _deprioritize_form_chunks(results: list) -> list:
        """Deprioritize form template chunks (별지 서식) by moving them to the end.

        Form template chunks contain checkboxes, form fields, and layout markers
        that are not useful for answering factual questions. When mixed with
        actual article content, they dilute the context quality.
        """
        if not results or len(results) <= 1:
            return results

        substantive = []
        form_chunks = []
        for r in results:
            content = r.content if hasattr(r, 'content') else r.get('content', '')
            if _RE_FORM_TEMPLATE.search(content):
                form_chunks.append(r)
            else:
                substantive.append(r)

        if not substantive:
            # All chunks are form templates — keep original order
            return results

        if form_chunks:
            logger.debug(
                "Deprioritized %d form template chunk(s) out of %d total",
                len(form_chunks), len(results),
            )

        return substantive + form_chunks

    async def _expand_appendix_refs(self, results: list, retriever) -> list:
        """Promote appendix (별표) table chunks to top of results.

        When regulation articles reference '별표X에 따른...', the actual
        table chunk with amounts/grades may be ranked low. This method
        finds table chunks in the results and promotes them to the top
        so the LLM can provide concrete answers.
        """
        if not results or len(results) <= 3:
            return results

        import re as _re
        from config.settings import settings as _settings
        check_limit = _settings.context_max_chunks if _settings.context_max_chunks > 0 else 10

        # Find 별표 references in top results
        byulpyo_refs = set()
        for r in results[:check_limit]:
            content = r.content if hasattr(r, "content") else ""
            matches = _re.findall(r"별표\s*(\d+(?:의\d+)?)", content)
            for m in matches:
                byulpyo_refs.add(m)

        if not byulpyo_refs:
            return results

        # Find table chunks in ALL results (including those beyond context_max_chunks)
        table_chunks = []
        non_table_chunks = []
        for r in results:
            content = r.content if hasattr(r, "content") else ""
            # Check if this is an actual appendix table (has header + table format)
            is_table = (
                ("【별표" in content or "별표 " in content.split("\n")[0] if content else False) and
                (content.count("|") >= 3 or "기준표" in content or "단위" in content)
            )
            if is_table:
                table_chunks.append(r)
            else:
                non_table_chunks.append(r)

        if not table_chunks:
            return results

        logger.info(
            "Promoted %d appendix table chunk(s) to top for references: %s",
            len(table_chunks), byulpyo_refs,
        )

        # Put table chunks at position 2 (after most relevant result)
        promoted = non_table_chunks[:1] + table_chunks + non_table_chunks[1:]
        return promoted


    @staticmethod
    def _auto_detect_filters(question: str, filters: dict | None) -> dict | None:
        """Auto-detect source_type filter based on question keywords.

        Keywords are loaded from prompts/source_filters.yaml for easy maintenance.
        Uses score-based matching: the source_type with the most keyword hits wins.
        This prevents conflicts where a query matches keywords from multiple sources.
        """
        if filters and filters.get("source_type"):
            return filters  # User already specified, don't override

        q = question.lower()
        config = _load_source_filters()
        source_types = config.get("source_types", {})

        best_source = None
        best_score = 0
        for source_name, source_config in source_types.items():
            keywords = source_config.get("keywords", [])
            score = sum(1 for kw in keywords if kw.lower() in q)
            if score > best_score:
                best_score = score
                best_source = source_name

        if best_source:
            new_filters = dict(filters) if filters else {}
            new_filters["source_type"] = best_source
            return new_filters

        return filters

    @staticmethod
    def _detect_multi_hop(question: str) -> bool:
        """Detect multi-hop questions that need cross-regulation retrieval.

        Keywords are loaded from prompts/source_filters.yaml.
        Multi-hop questions reference multiple regulations or ask about
        relationships/processes spanning different rule sets.  Activating
        multi-query retrieval significantly improves recall for these.
        """
        q = question.lower()
        config = _load_source_filters()
        multi_hop_keywords = config.get("multi_hop_keywords", [])
        return any(kw in q for kw in multi_hop_keywords)

    @staticmethod
    def _trim_direct_question(question: str) -> str:
        """Trim question to fit model context window in direct mode.

        Direct mode sends the full question (including OCR file content) to the LLM
        without RAG retrieval. This ensures the input doesn't exceed the model's
        context window. Only affects direct mode — RAG mode has its own budget logic.
        """
        max_ctx = getattr(settings, 'vllm_max_model_len', 4096)
        max_out = settings.llm_max_tokens
        # Direct mode system prompt is short (~200 tokens), use 500 token margin
        budget_chars = int((max_ctx - max_out - 500) * 1.3)
        if budget_chars > 0 and len(question) > budget_chars:
            original_len = len(question)
            question = question[:budget_chars]
            logger.warning(
                "Direct query trimmed: %d -> %d chars (budget %d tokens)",
                original_len, budget_chars, max_ctx - max_out - 500,
            )
        return question

    @staticmethod
    def _make_cache_key(question: str, mode: str, filters: dict | None, provider: str | None, model: str | None) -> str:
        """Build a deterministic cache key for a RAG query."""
        raw = json.dumps(
            {"q": question, "m": mode, "f": filters, "p": provider, "mdl": model},
            sort_keys=True,
            ensure_ascii=False,
        )
        return "rag:query:" + hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def _postprocess_answer(text: str) -> str:
        """Strip common LLM output artifacts."""
        text = text.strip()
        text = _RE_QA_PATTERN.sub('', text)
        text = _RE_A_PREFIX.sub('', text.strip())
        text = _RE_SOURCE_TAG.sub('', text)
        text = _RE_CJK_LEAKAGE.sub('', text)
        # Strip any HTML tags from LLM output to prevent XSS
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    @staticmethod
    def _is_valid_response(text: str) -> bool:
        """Check if LLM response is valid (not garbled/truncated/wrong language)."""
        stripped = text.strip()
        if len(stripped) < 15:
            return False
        content_chars = _RE_WHITESPACE.sub('', stripped)
        if len(content_chars) < 10:
            return False
        chinese_chars = len(_RE_CHINESE.findall(stripped))
        if chinese_chars > 0 and chinese_chars / max(len(stripped), 1) > 0.05:
            return False
        return True

    async def query(
        self,
        question: str,
        mode: str = "rag",
        filters: dict | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        user_id: str = "",
        query_class: str | None = None,
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

        # Dual model routing (if enabled and no explicit model override)
        selected_model_tier = None
        routing_kwargs: dict = {}
        if settings.model_routing_enabled and not model:
            try:
                from agent.router import QueryRouter, ModelTier
                router = QueryRouter(llm=self.llm)
                routing = await router.route(question)
                selected_model_tier = routing.model_tier

                # Select model based on tier
                if routing.model_tier == ModelTier.LIGHT:
                    model = settings.light_llm_model
                    if settings.default_llm_provider == "vllm":
                        routing_kwargs["base_url"] = settings.vllm_light_base_url
                else:
                    model = settings.main_llm_model
                    if settings.default_llm_provider == "vllm":
                        routing_kwargs["base_url"] = settings.vllm_main_base_url

                logger.info(
                    "Dual model routing: %s -> %s (%s)",
                    routing.category.value, routing.model_tier.value, model,
                )
            except Exception as e:
                logger.warning("Dual model routing failed, using default: %s", e)

        # Use override LLM if specified
        llm = self.llm
        temp_llm = None
        if provider or model:
            temp_llm = create_llm(provider=provider, model=model, temperature=temperature or settings.llm_temperature, **routing_kwargs)
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
            agentic_use_hyde: bool | None = None
            agentic_use_self_rag: bool | None = None
            agentic_max_tokens: int | None = None
            if agentic_meta:
                agentic_use_multi_query = bool(agentic_meta.get("strategy") == "multi_query_rag")
                params = agentic_meta.get("_params", {})
                if params.get("top_k"):
                    agentic_top_k = params["top_k"]
                if "use_hyde" in params:
                    agentic_use_hyde = params["use_hyde"]
                if "use_self_rag" in params:
                    agentic_use_self_rag = params["use_self_rag"]
                if params.get("max_tokens"):
                    agentic_max_tokens = params["max_tokens"]

            # Auto-detect multi-hop questions for multi-query retrieval
            if not agentic_use_multi_query and self._detect_multi_hop(question):
                agentic_use_multi_query = True
                if agentic_top_k is None:
                    agentic_top_k = 30  # Wider net for cross-regulation coverage
                logger.info("Multi-hop query detected, forcing multi-query retrieval (top_k=%d)", agentic_top_k)

            # Auto-detect source_type filter from question
            if mode != "direct" and settings.source_type_filter_enabled:
                filters = self._auto_detect_filters(question, filters)
                if filters and filters.get("source_type"):
                    logger.info("Auto-detected source_type filter: %s", filters["source_type"])

            if mode == "direct":
                response = await self._direct_query(question, llm, start_time, user_id, correction_info, temperature, query_class=query_class)
            else:
                response = await self._rag_query(
                    question, llm, filters, start_time, user_id, correction_info, temperature,
                    force_multi_query=agentic_use_multi_query, override_top_k=agentic_top_k,
                    search_query=search_query, terminology_info=terminology_info,
                    use_hyde=agentic_use_hyde, use_self_rag=agentic_use_self_rag,
                    override_max_tokens=agentic_max_tokens,
                )

            # Add agentic routing metadata (exclude internal _params)
            if agentic_meta:
                response.metadata["agentic_routing"] = {
                    k: v for k, v in agentic_meta.items() if not k.startswith("_")
                }

            # Add model_tier info from dual model routing
            if selected_model_tier is not None:
                response.metadata["model_tier"] = selected_model_tier.value

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
        use_hyde: bool | None = None,
        use_self_rag: bool | None = None,
        override_max_tokens: int | None = None,
    ) -> RAGResponse:
        """RAG mode: retrieve documents then generate."""
        # Use expanded search query if provided, else fall back to question
        search_query = search_query or question

        # Step 0+1: Retrieve (with optional HyDE and Multi-Query)
        hyde_used = False
        multi_query_used = False
        hyde_embedding = None

        # Determine effective overrides (don't mutate shared retriever state)
        effective_top_k = override_top_k or None
        effective_rerank_top_n = 8 if force_multi_query else None

        # Generate HyDE embedding if enabled
        # Per-query routing override; fall back to global setting
        effective_use_hyde = use_hyde if use_hyde is not None else settings.query_expansion_enabled
        if effective_use_hyde:
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
                    rerank_top_n=effective_rerank_top_n,
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
                rerank_top_n=effective_rerank_top_n,
            )

        # Deprioritize form template chunks (별지 서식)
        retrieval_results = self._deprioritize_form_chunks(retrieval_results)

        # Step 2.5: Expand appendix references (별표 cross-reference)
        retrieval_results = await self._expand_appendix_refs(retrieval_results, self.retriever)

        # Step 2: Calculate confidence
        chunk_scores = [r.score for r in retrieval_results]
        confidence = self.quality.calculate_confidence(chunk_scores)

        # Step 3: Limit retrieval results to context_max_chunks BEFORE building sources
        # This ensures sources shown to user match what the LLM actually uses
        if settings.context_max_chunks > 0:
            retrieval_results = retrieval_results[:settings.context_max_chunks]

        sources = [
            {
                "chunk_id": r.id,
                "content": r.content,
                "score": round(r.score, 3),
                "metadata": r.metadata,
                "source_url": (
                    f"/api/documents/{r.metadata.get('document_id')}/download"
                    if r.metadata.get("document_id")
                    else ""
                ),
            }
            for r in retrieval_results
        ]

        # Step 4: Build prompt
        context_chunks = [
            {"content": r.content, "metadata": r.metadata}
            for r in retrieval_results
        ]

        # Determine model name for model-aware prompting
        model_name = getattr(llm, 'model', None)

        # Token budget: trim context to fit model context window
        max_ctx = getattr(settings, 'vllm_max_model_len', 4096)
        max_out = settings.llm_max_tokens
        # Reserve tokens for system prompt + few-shot + query + safety margin
        budget = max_ctx - max_out - 2500
        if budget > 0:
            total_chars = 0
            trimmed = []
            for chunk in context_chunks:
                est_tokens = len(chunk["content"]) / 1.3  # Korean+Qwen: ~1.3 chars per token
                if total_chars + est_tokens > budget:
                    logger.info("Trimming context: %d chunks → %d (token budget %d)", len(context_chunks), len(trimmed), budget)
                    break
                total_chars += est_tokens
                trimmed.append(chunk)
            context_chunks = trimmed if trimmed else context_chunks[:1]

        system_prompt, user_prompt = self.prompt_manager.build_rag_prompt(
            query=question,
            context_chunks=context_chunks,
            model_hint=model_name,
        )

        # Step 5: Generate (with validation retry for garbled/wrong-language output)
        effective_max_tokens = override_max_tokens or settings.llm_max_tokens
        max_retries = 2
        response = None
        for attempt in range(max_retries + 1):
            response = await llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )
            if self._is_valid_response(response.content):
                break
            if attempt < max_retries:
                logger.warning(
                    "Invalid LLM response (attempt %d/%d), retrying: %s",
                    attempt + 1, max_retries + 1, response.content[:50],
                )

        latency_ms = int((time.time() - start_time) * 1000)

        # Step 6: Quality assessment
        safety_warning = self.quality.get_safety_message(confidence)
        content = response.content
        postprocessed = self._postprocess_answer(content)
        content = postprocessed if postprocessed else content.strip()

        # Self-RAG: evaluate groundedness and retry if needed
        self_rag_meta = None
        effective_use_self_rag = use_self_rag if use_self_rag is not None else settings.self_rag_enabled
        if effective_use_self_rag:
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
        query_class: str | None = None,
    ) -> RAGResponse:
        """Direct mode: LLM only, no retrieval."""
        # Token budget: trim question to fit model context window
        question = self._trim_direct_question(question)

        # Select system prompt based on query_class
        prompt_name = "direct_system"
        if query_class == "chitchat":
            prompt_name = "chitchat_system"
        elif query_class == "general":
            prompt_name = "general_system"

        try:
            system_prompt = self.prompt_manager.get_system_prompt(prompt_name)
        except KeyError:
            system_prompt, _ = self.prompt_manager.build_direct_prompt(question)
        user_prompt = question

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
        query_class: str | None = None,
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

        # Check stream cache for identical queries (TTL: 120s)
        stream_cache_key = self._make_cache_key(question, mode, filters, provider, model)
        stream_cache = None
        try:
            from core.cache import get_cache
            stream_cache = await get_cache()
            cached = await stream_cache.get(stream_cache_key)
            if cached is not None:
                logger.debug("Stream cache hit for query: %s", question[:50])
                yield {"event": "start", "data": {"message_id": message_id}}
                # Replay cached sources
                for src in cached.get("sources", []):
                    yield {"event": "source", "data": src}
                # Replay cached content as single chunk
                if cached.get("content"):
                    yield {"event": "chunk", "data": {"content": cached["content"]}}
                # Replay end event with updated latency
                latency_ms = int((time.time() - start_time) * 1000)
                end_data = cached.get("end_data", {})
                end_data["latency_ms"] = latency_ms
                end_data["cached"] = True
                yield {"event": "end", "data": end_data}
                return
        except Exception:
            stream_cache = None

        # Use override LLM if specified
        llm = self.llm
        temp_llm = None
        if provider or model:
            temp_llm = create_llm(provider=provider, model=model, temperature=temperature or settings.llm_temperature)
            llm = temp_llm

        selected_model_tier = None
        routing_kwargs: dict = {}
        try:
            yield {"event": "start", "data": {"message_id": message_id}}

            # Dual model routing (if enabled and no explicit model override)
            if settings.model_routing_enabled and not model:
                try:
                    from agent.router import QueryRouter, ModelTier
                    router = QueryRouter(llm=self.llm)
                    routing = await router.route(question)
                    selected_model_tier = routing.model_tier

                    # Select model based on tier
                    if routing.model_tier == ModelTier.LIGHT:
                        model = settings.light_llm_model
                        if settings.default_llm_provider == "vllm":
                            routing_kwargs["base_url"] = settings.vllm_light_base_url
                    else:
                        model = settings.main_llm_model
                        if settings.default_llm_provider == "vllm":
                            routing_kwargs["base_url"] = settings.vllm_main_base_url

                    yield {"event": "routing", "data": {
                        "category": routing.category.value,
                        "model_tier": routing.model_tier.value,
                        "model": model,
                        "confidence": routing.confidence,
                    }}
                    logger.info(
                        "Dual model routing: %s -> %s (%s)",
                        routing.category.value, routing.model_tier.value, model,
                    )
                except Exception as e:
                    logger.warning("Dual model routing failed, using default: %s", e)

            # Apply model override after routing (rebuild temp_llm if model was resolved above)
            if model and not temp_llm:
                temp_llm = create_llm(provider=provider, model=model, temperature=temperature or settings.llm_temperature, **routing_kwargs)
                llm = temp_llm

            # --- Pre-retrieval: guardrails + query preprocessing (parallel) ---
            # Guardrails runs concurrently with query correction + terminology
            import asyncio as _asyncio

            async def _check_guardrails():
                try:
                    from .guardrails import get_guardrails_manager
                    guard = await get_guardrails_manager()
                    return await guard.check_input(question, user_id=user_id)
                except Exception as e:
                    logger.debug("Guardrails input check skipped: %s", e)
                    return None

            guard_task = _asyncio.create_task(_check_guardrails())

            # Query correction (sync, ~0.1ms)
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
                    logger.info("Stream: Query corrected: '%s' -> '%s'", correction.original[:50], correction.corrected[:50])
            except Exception as e:
                logger.debug("Stream: Query correction skipped: %s", e)

            # Terminology expansion (sync, ~0.1ms, depends on corrected query)
            terminology_info = None
            search_query = question
            try:
                from .terminology import get_terminology_service
                ts = get_terminology_service()
                expansion = ts.expand_query(question)
                if expansion.was_expanded:
                    terminology_info = {
                        "matched_terms": expansion.matched_terms,
                        "expansions": expansion.expansions,
                    }
                    search_query = expansion.expanded_query
                    logger.info("Stream: Query expanded with terminology: %d terms matched", len(expansion.matched_terms))
            except Exception as e:
                logger.debug("Stream: Terminology expansion skipped: %s", e)

            # Await guardrails result
            guard_result = await guard_task
            if guard_result and not guard_result.passed and guard_result.action == "block":
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

            # Agentic RAG routing
            agentic_meta = None
            agentic_use_multi_query = False
            agentic_top_k = None
            agentic_use_hyde: bool | None = None
            agentic_use_self_rag: bool | None = None
            agentic_max_tokens: int | None = None
            if settings.agentic_rag_enabled and mode == "rag":
                try:
                    from .agentic import AgenticRAGRouter
                    agentic_router = AgenticRAGRouter(llm=llm)
                    decision = await agentic_router.route(question)
                    agentic_meta = {
                        "strategy": decision.strategy,
                        "routing_confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                    }
                    logger.info("Stream: Agentic RAG: strategy=%s (confidence=%.2f)", decision.strategy, decision.confidence)
                    if decision.params.get("mode") == "direct":
                        mode = "direct"
                    if decision.params.get("temperature") is not None:
                        temperature = temperature if temperature is not None else decision.params["temperature"]
                    agentic_use_multi_query = bool(decision.strategy == "multi_query_rag")
                    if decision.params.get("top_k"):
                        agentic_top_k = decision.params["top_k"]
                    if "use_hyde" in decision.params:
                        agentic_use_hyde = decision.params["use_hyde"]
                    if "use_self_rag" in decision.params:
                        agentic_use_self_rag = decision.params["use_self_rag"]
                    if decision.params.get("max_tokens"):
                        agentic_max_tokens = decision.params["max_tokens"]
                except Exception as e:
                    logger.warning("Stream: Agentic RAG routing failed: %s", e)

            # Auto-detect multi-hop for multi-query retrieval
            if not agentic_use_multi_query and self._detect_multi_hop(question):
                agentic_use_multi_query = True
                if agentic_top_k is None:
                    agentic_top_k = 30
                logger.info("Stream: Multi-hop query detected, forcing multi-query retrieval")

            # Auto-detect source_type filter from question
            if mode != "direct" and settings.source_type_filter_enabled:
                filters = self._auto_detect_filters(question, filters)
                if filters and filters.get("source_type"):
                    logger.info("Stream: Auto-detected source_type filter: %s", filters["source_type"])

            if mode == "direct":
                # Token budget: trim question to fit model context window
                question = self._trim_direct_question(question)

                # Select system prompt based on query_class
                _prompt_name = "direct_system"
                if query_class == "chitchat":
                    _prompt_name = "chitchat_system"
                elif query_class == "general":
                    _prompt_name = "general_system"
                try:
                    system = self.prompt_manager.get_system_prompt(_prompt_name)
                except KeyError:
                    system, _ = self.prompt_manager.build_direct_prompt(question)
                user_prompt = question

                accumulated = []
                async for token in llm.stream(prompt=user_prompt, system=system, temperature=temperature, max_tokens=settings.llm_max_tokens):
                    accumulated.append(token)
                    yield {"event": "chunk", "data": {"content": token}}

                # Post-stream output guardrails
                full_response = "".join(accumulated)
                guardrail_triggered = False
                try:
                    from .guardrails import get_guardrails_manager
                    guard = await get_guardrails_manager()
                    out_result = await guard.check_output(full_response, user_id=user_id)
                    if not out_result.passed and out_result.action == "block":
                        guardrail_triggered = True
                        yield {"event": "guardrail_warning", "data": {"message": out_result.message or "응답이 안전 필터에 의해 차단되었습니다.", "triggered_rules": out_result.triggered_rules}}
                except Exception as e:
                    logger.debug("Streaming guardrails output check skipped: %s", e)

                latency_ms = int((time.time() - start_time) * 1000)
                end_data = {"confidence_score": 0.5, "confidence_level": "medium", "latency_ms": latency_ms, "note": "검색 근거 없이 LLM만으로 생성된 답변"}
                if guardrail_triggered:
                    end_data["guardrail_blocked"] = True
                if correction_info:
                    end_data["query_correction"] = correction_info
                if terminology_info:
                    end_data["terminology_expansion"] = terminology_info
                if selected_model_tier is not None:
                    end_data["model_tier"] = selected_model_tier.value
                yield {"event": "end", "data": end_data}
                return

            # RAG mode with streaming
            # Step 0: HyDE query expansion (optional)
            hyde_embedding = None
            effective_use_hyde = agentic_use_hyde if agentic_use_hyde is not None else settings.query_expansion_enabled
            if effective_use_hyde:
                try:
                    from .query_expander import QueryExpander
                    expander = QueryExpander(llm=llm)
                    hyde_embedding = await expander.expand_hyde(question)
                except Exception as e:
                    logger.warning("HyDE expansion failed in streaming: %s", e)

            # Step 1: Retrieve (with optional multi-query for multi-hop questions)
            use_multi_query_stream = settings.multi_query_enabled or agentic_use_multi_query
            multi_query_stream_used = False
            if use_multi_query_stream:
                try:
                    from .multi_query import MultiQueryRetriever
                    mq_retriever = MultiQueryRetriever(
                        retriever=self.retriever,
                        llm=llm,
                        query_count=settings.multi_query_count,
                    )
                    results = await mq_retriever.retrieve(
                        query=search_query,
                        top_k=agentic_top_k,
                        filters=filters,
                        hyde_embedding=hyde_embedding,
                        rerank_top_n=8 if agentic_use_multi_query else None,
                    )
                    multi_query_stream_used = True
                    logger.info("Streaming: multi-query retrieval used")
                except Exception as e:
                    logger.warning("Streaming: multi-query failed, falling back: %s", e)

            if not multi_query_stream_used:
                results = await self.retriever.retrieve(query=search_query, filters=filters, hyde_embedding=hyde_embedding)

            # Deprioritize form template chunks (별지 서식)
            results = self._deprioritize_form_chunks(results)

            # Expand appendix references (별표 cross-reference)
            results = await self._expand_appendix_refs(results, self.retriever)

            # Limit results to context_max_chunks (match sources to what LLM uses)
            if settings.context_max_chunks > 0:
                results = results[:settings.context_max_chunks]

            chunk_scores = [r.score for r in results]
            confidence = self.quality.calculate_confidence(chunk_scores)

            # Emit sources (include content for frontend expand)
            for r in results:
                doc_id = r.metadata.get("document_id", "")
                yield {
                    "event": "source",
                    "data": {
                        "chunk_id": r.id,
                        "filename": r.metadata.get("filename", ""),
                        "page": r.metadata.get("page_number"),
                        "score": round(r.score, 3),
                        "content": r.content[:500] if r.content else "",
                        "source_url": f"/api/documents/{doc_id}/download" if doc_id else "",
                    },
                }

            # Step 2: Build prompt and stream
            context_chunks = [{"content": r.content, "metadata": r.metadata} for r in results]
            # Limit context chunks for LLM (if configured)
            if settings.context_max_chunks > 0:
                context_chunks = context_chunks[:settings.context_max_chunks]

            # Token budget: trim context to fit model context window
            max_ctx = getattr(settings, 'vllm_max_model_len', 4096)
            max_out = settings.llm_max_tokens
            # Reserve tokens for system prompt + few-shot + query + safety margin
            budget = max_ctx - max_out - 2500
            if budget > 0:
                total_chars = 0
                trimmed_s = []
                for chunk in context_chunks:
                    est_tokens = len(chunk["content"]) / 1.3  # Korean+Qwen: ~1.3 chars/token
                    if total_chars + est_tokens > budget:
                        logger.info("Stream: trimming context %d→%d chunks (budget %d)", len(context_chunks), len(trimmed_s), budget)
                        break
                    total_chars += est_tokens
                    trimmed_s.append(chunk)
                context_chunks = trimmed_s if trimmed_s else context_chunks[:1]

            model_name = getattr(llm, 'model', None)
            system, user_prompt = self.prompt_manager.build_rag_prompt(
                query=question, context_chunks=context_chunks,
                model_hint=model_name,
            )

            # Safety warning as separate event (don't pollute the answer content)
            if self.quality.should_add_safety_warning(confidence):
                yield {"event": "confidence_warning", "data": {"message": "검색 신뢰도가 낮습니다. 답변의 정확성을 확인해주세요.", "confidence": round(confidence, 3)}}

            # Stream LLM response with post-stream guardrails
            # Buffer initial tokens to strip "답변:" / "A:" prefix
            effective_max_tokens = agentic_max_tokens or settings.llm_max_tokens
            accumulated = []
            prefix_stripped = False
            prefix_buffer = ""
            ttft_ms: int | None = None
            generation_start_time = time.time()
            llm_error_occurred = False
            cjk_consecutive = 0  # Track consecutive CJK-only tokens
            try:
                async for token in llm.stream(prompt=user_prompt, system=system, temperature=temperature, max_tokens=effective_max_tokens):
                    if not prefix_stripped:
                        prefix_buffer += token
                        # Wait until we have enough chars to check for prefix
                        if len(prefix_buffer) < 5:
                            continue
                        # Strip common prefixes
                        stripped = _RE_A_PREFIX.sub('', prefix_buffer.lstrip())
                        prefix_stripped = True
                        stripped = _RE_STRIP_CJK.sub('', stripped)
                        if stripped:
                            if ttft_ms is None:
                                ttft_ms = int((time.time() - start_time) * 1000)
                            accumulated.append(stripped)
                            yield {"event": "chunk", "data": {"content": stripped}}
                        continue
                    # Strip Chinese chars from token (Qwen leakage fix)
                    clean_token = _RE_STRIP_CJK.sub('', token)
                    if not clean_token:
                        # Token was entirely Chinese — skip and count
                        cjk_consecutive += 1
                        if cjk_consecutive >= 3:
                            logger.warning("Chinese leakage: 3+ consecutive CJK-only tokens, stopping stream")
                            break
                        continue
                    cjk_consecutive = 0
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - start_time) * 1000)
                    accumulated.append(clean_token)
                    yield {"event": "chunk", "data": {"content": clean_token}}
            except Exception as llm_err:
                logger.error("LLM stream error: %s", llm_err)
                llm_error_occurred = True
                yield {"event": "error", "data": {"message": "LLM 응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."}}

            # Post-stream: detect Chinese leakage (Qwen model issue)
            full_response = "".join(accumulated)
            if full_response and not llm_error_occurred:
                chinese_chars = len(_RE_CHINESE.findall(full_response))
                chinese_ratio = chinese_chars / max(len(full_response.strip()), 1)
                if chinese_ratio > 0.05:
                    logger.warning("Chinese leakage detected (%.1f%%), sending warning", chinese_ratio * 100)
                    yield {"event": "chinese_warning", "data": {"message": "중국어가 포함된 응답이 감지되었습니다. 재시도해주세요.", "ratio": round(chinese_ratio, 3)}}

            # Post-stream output guardrails (skip if LLM errored)
            guardrail_triggered = False
            if not llm_error_occurred:
                try:
                    from .guardrails import get_guardrails_manager
                    guard = await get_guardrails_manager()
                    out_result = await guard.check_output(full_response, user_id=user_id)
                    if not out_result.passed and out_result.action == "block":
                        guardrail_triggered = True
                        yield {"event": "guardrail_warning", "data": {"message": out_result.message or "응답이 안전 필터에 의해 차단되었습니다.", "triggered_rules": out_result.triggered_rules}}
                except Exception as e:
                    logger.debug("Streaming guardrails output check skipped: %s", e)

            # Self-RAG: post-stream grounding check
            self_rag_meta = None
            effective_use_self_rag = agentic_use_self_rag if agentic_use_self_rag is not None else settings.self_rag_enabled
            if not llm_error_occurred and effective_use_self_rag and full_response and not guardrail_triggered:
                try:
                    from .self_rag import SelfRAGEvaluator
                    context_str = "\n\n".join(c["content"] for c in context_chunks)
                    self_rag = SelfRAGEvaluator(
                        llm=llm,
                        max_retries=0,  # No retry in streaming — grade only
                    )
                    grade = await self_rag.grade_answer(question, full_response, context_str)
                    self_rag_meta = {
                        "self_rag_enabled": True,
                        "grounded": grade.grounded,
                        "hallucination": grade.hallucination,
                        "relevance": grade.relevance,
                        "confidence": grade.confidence,
                        "passed": grade.passed,
                    }
                    if not grade.passed:
                        yield {
                            "event": "self_rag_warning",
                            "data": {
                                "message": "근거 검증에서 주의가 필요한 답변입니다. 원문을 직접 확인해주세요.",
                                "grounded": grade.grounded,
                                "hallucination": grade.hallucination,
                            },
                        }
                    logger.info(
                        "Stream Self-RAG: grounded=%s, hallucination=%s, passed=%s",
                        grade.grounded, grade.hallucination, grade.passed,
                    )
                except Exception as e:
                    logger.warning("Stream Self-RAG grading failed: %s", e)

            latency_ms = int((time.time() - start_time) * 1000)
            # TPS: total characters / generation duration (seconds)
            full_response_len = len("".join(accumulated))
            generation_elapsed = time.time() - generation_start_time
            tps = round(full_response_len / generation_elapsed, 1) if generation_elapsed > 0 and full_response_len > 0 else None
            end_data = {
                "confidence_score": round(confidence, 3),
                "confidence_level": self.quality.get_confidence_level(confidence),
                "latency_ms": latency_ms,
                "source_count": len(results),
            }
            if ttft_ms is not None:
                end_data["ttft_ms"] = ttft_ms
            if tps is not None:
                end_data["tps"] = tps
            if guardrail_triggered:
                end_data["guardrail_blocked"] = True
            if correction_info:
                end_data["query_correction"] = correction_info
            if terminology_info:
                end_data["terminology_expansion"] = terminology_info
            if agentic_meta:
                end_data["agentic_routing"] = agentic_meta
            if selected_model_tier is not None:
                end_data["model_tier"] = selected_model_tier.value
            if self_rag_meta:
                end_data["self_rag"] = self_rag_meta
            if llm_error_occurred:
                end_data["llm_error"] = True

            # Store in cache for future identical queries (TTL: 120s)
            if stream_cache is not None and not guardrail_triggered and not llm_error_occurred:
                try:
                    source_data = [
                        {
                            "chunk_id": r.id,
                            "filename": r.metadata.get("filename", ""),
                            "page": r.metadata.get("page_number"),
                            "score": round(r.score, 3),
                            "content": r.content[:500] if r.content else "",
                            "source_url": (
                                f"/api/documents/{r.metadata.get('document_id')}/download"
                                if r.metadata.get("document_id")
                                else ""
                            ),
                        }
                        for r in results
                    ]
                    await stream_cache.set(stream_cache_key, {
                        "content": full_response,
                        "sources": source_data,
                        "end_data": end_data,
                    }, ttl=120)
                except Exception:
                    pass

            yield {"event": "end", "data": end_data}
        finally:
            if temp_llm is not None:
                await temp_llm.close()
