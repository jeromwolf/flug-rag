"""Agentic RAG: Dynamic strategy routing based on question analysis.

Analyzes user intent and selects the optimal RAG strategy:
- standard_rag: Simple fact-finding questions
- multi_query_rag: Complex/comparison questions needing multiple perspectives
- direct_llm: General knowledge or chitchat (no retrieval needed)
- deep_retrieval: Detailed/expert questions needing thorough search
"""

import json
import logging
from dataclasses import dataclass

from config.settings import settings
from core.llm import BaseLLM, create_llm
from rag.prompt import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of agentic routing analysis."""
    strategy: str  # standard_rag, multi_query_rag, direct_llm, deep_retrieval
    confidence: float
    reasoning: str
    params: dict  # Dynamic parameter overrides (top_k, temperature, etc.)


class AgenticRAGRouter:
    """Analyzes questions and routes to optimal RAG strategy.

    Strategies:
    - standard_rag: Default vector+BM25 hybrid search
    - multi_query_rag: Generate alternative queries for better recall
    - direct_llm: Skip retrieval, use LLM knowledge directly
    - deep_retrieval: Increase top_k, use more context for detailed answers
    """

    # Default params for each strategy
    STRATEGY_DEFAULTS = {
        "standard_rag": {
            "mode": "rag",
            "top_k": None,  # Use default
            "rerank_top_n": None,
            "temperature": None,
            "use_multi_query": False,
        },
        "multi_query_rag": {
            "mode": "rag",
            "top_k": None,
            "rerank_top_n": None,
            "temperature": 0.3,
            "use_multi_query": True,
        },
        "direct_llm": {
            "mode": "direct",
            "top_k": 0,
            "rerank_top_n": 0,
            "temperature": 0.7,
            "use_multi_query": False,
        },
        "deep_retrieval": {
            "mode": "rag",
            "top_k": 30,
            "rerank_top_n": 7,
            "temperature": 0.3,
            "use_multi_query": False,
        },
    }

    def __init__(
        self,
        llm: BaseLLM | None = None,
        prompt_manager: PromptManager | None = None,
    ):
        self._llm = llm
        self._prompt_manager = prompt_manager

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

    async def route(self, query: str) -> RoutingDecision:
        """Analyze question and determine optimal strategy.

        Args:
            query: User's question.

        Returns:
            RoutingDecision with strategy and parameter overrides.
        """
        # Try LLM-based routing first
        try:
            return await self._llm_route(query)
        except Exception as e:
            logger.warning("Agentic LLM routing failed, using rule-based fallback: %s", e)
            return self._rule_based_route(query)

    async def _llm_route(self, query: str) -> RoutingDecision:
        """Use LLM to analyze query and select strategy."""
        prompt = self.prompt_manager.get_system_prompt("agentic_routing").format(query=query)

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=256,
        )

        content = response.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1

        if start >= 0 and end > start:
            data = json.loads(content[start:end])
            strategy = data.get("strategy", "standard_rag")

            # Validate strategy
            if strategy not in self.STRATEGY_DEFAULTS:
                strategy = "standard_rag"

            # Merge LLM-suggested params with defaults
            default_params = dict(self.STRATEGY_DEFAULTS[strategy])
            llm_params = data.get("params", {})
            if isinstance(llm_params, dict):
                for key in ("top_k", "temperature"):
                    if key in llm_params and llm_params[key] is not None:
                        default_params[key] = llm_params[key]

            return RoutingDecision(
                strategy=strategy,
                confidence=float(data.get("confidence", 0.8)),
                reasoning=data.get("reasoning", ""),
                params=default_params,
            )

        raise ValueError("Failed to parse routing response")

    @staticmethod
    def _rule_based_route(query: str) -> RoutingDecision:
        """Rule-based fallback routing using keyword patterns.

        Used when LLM routing fails.
        """
        import re

        query_lower = query.strip().lower()
        query_len = len(query_lower)

        # Chitchat / greeting detection
        chitchat_patterns = r'^(안녕|하이|ㅎㅇ|감사|고마워|수고|반갑|잘\s*가)'
        if re.search(chitchat_patterns, query_lower) and query_len < 20:
            return RoutingDecision(
                strategy="direct_llm",
                confidence=0.9,
                reasoning="Short greeting/chitchat detected",
                params=dict(AgenticRAGRouter.STRATEGY_DEFAULTS["direct_llm"]),
            )

        # Complex/comparison questions → multi-query
        complex_patterns = r'(비교|차이|vs|대비|종합|정리|분석|요약|전체|모든|각각)'
        if re.search(complex_patterns, query_lower):
            return RoutingDecision(
                strategy="multi_query_rag",
                confidence=0.8,
                reasoning="Complex/comparison question detected",
                params=dict(AgenticRAGRouter.STRATEGY_DEFAULTS["multi_query_rag"]),
            )

        # Detailed/expert questions → deep retrieval
        expert_patterns = r'(상세|자세|구체적|절차|과정|단계|방법|어떻게|세부)'
        if re.search(expert_patterns, query_lower) and query_len > 20:
            return RoutingDecision(
                strategy="deep_retrieval",
                confidence=0.75,
                reasoning="Detailed/expert question detected",
                params=dict(AgenticRAGRouter.STRATEGY_DEFAULTS["deep_retrieval"]),
            )

        # General knowledge (no document keywords)
        document_patterns = r'(법|규정|조|항|호|시행|매뉴얼|지침|기준|표준|점검|설비|배관|가스|안전)'
        if not re.search(document_patterns, query_lower):
            return RoutingDecision(
                strategy="direct_llm",
                confidence=0.6,
                reasoning="No document-related keywords found",
                params=dict(AgenticRAGRouter.STRATEGY_DEFAULTS["direct_llm"]),
            )

        # Default: standard RAG
        return RoutingDecision(
            strategy="standard_rag",
            confidence=0.7,
            reasoning="Default routing to standard RAG",
            params=dict(AgenticRAGRouter.STRATEGY_DEFAULTS["standard_rag"]),
        )

    def get_strategy_params(self, decision: RoutingDecision) -> dict:
        """Get final parameters for the chosen strategy.

        Merges strategy defaults with any custom params from routing.
        """
        return decision.params
