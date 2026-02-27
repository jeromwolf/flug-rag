"""Question type router for intelligent routing."""

import json
from dataclasses import dataclass
from enum import Enum

from core.llm import BaseLLM, create_llm
from rag.prompt import PromptManager


class QueryCategory(str, Enum):
    DOCUMENT_SEARCH = "document_search"    # Needs RAG retrieval
    GENERAL_QUERY = "general_query"        # Direct LLM answer
    COMPLEX_TASK = "complex_task"          # Multi-step planning needed
    TOOL_REQUIRED = "tool_required"        # External tool execution
    CHITCHAT = "chitchat"                  # Casual conversation


class ModelTier(str, Enum):
    MAIN = "main"    # Complex queries: 72B (prod) / 14b (dev)
    LIGHT = "light"  # Simple queries: 32B (prod) / 7b (dev)


def _tier_for_category(category: QueryCategory) -> ModelTier:
    """Determine model tier based on query category."""
    if category in (QueryCategory.COMPLEX_TASK, QueryCategory.TOOL_REQUIRED, QueryCategory.DOCUMENT_SEARCH):
        return ModelTier.MAIN
    # GENERAL_QUERY, CHITCHAT -> light
    return ModelTier.LIGHT


@dataclass
class RoutingResult:
    category: QueryCategory
    confidence: float
    reasoning: str
    model_tier: ModelTier = ModelTier.MAIN  # Model tier determined by query complexity
    rewritten_query: str | None = None  # If query was rewritten using history


class QueryRouter:
    """Routes user queries to appropriate handlers."""

    def __init__(self, llm: BaseLLM | None = None, prompt_manager: PromptManager | None = None):
        self.llm = llm or create_llm()
        self.prompt_manager = prompt_manager or PromptManager()

    async def route(self, query: str, history: list[dict] | None = None) -> RoutingResult:
        """Classify query into routing category.

        Args:
            query: User's question text.
            history: Previous conversation history for context.

        Returns:
            RoutingResult with category, confidence, and reasoning.
        """
        # Step 1: Rewrite query if history exists (resolve pronouns, references)
        rewritten = None
        if history and len(history) >= 2:
            rewritten = await self._rewrite_query(query, history)

        effective_query = rewritten or query

        # Step 2: Route
        system, user_prompt = self.prompt_manager.build_router_prompt(effective_query)
        response = await self.llm.generate(prompt=user_prompt, system=system, temperature=0.1)

        # Parse JSON response
        result = self._parse_routing(response.content)
        result.rewritten_query = rewritten
        return result

    async def _rewrite_query(self, query: str, history: list[dict]) -> str | None:
        """Rewrite query using conversation history to resolve references."""
        system, user_prompt = self.prompt_manager.build_rewrite_prompt(query, history)
        response = await self.llm.generate(prompt=user_prompt, system=system, temperature=0.1)
        rewritten = response.content.strip()
        # Only use rewrite if it's meaningfully different
        if rewritten and rewritten != query and len(rewritten) > 3:
            return rewritten
        return None

    def _parse_routing(self, response_text: str) -> RoutingResult:
        """Parse LLM routing response (JSON expected)."""
        try:
            # Try to extract JSON from response
            text = response_text.strip()
            # Handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            category = QueryCategory(data.get("category", "general_query"))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            # Parse model_tier from JSON if present, otherwise derive from category
            if "model_tier" in data:
                try:
                    model_tier = ModelTier(data["model_tier"])
                except ValueError:
                    model_tier = _tier_for_category(category)
            else:
                model_tier = _tier_for_category(category)
            return RoutingResult(category=category, confidence=confidence, reasoning=reasoning, model_tier=model_tier)
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback: use keyword heuristic
            return self._fallback_route(response_text)

    def _fallback_route(self, query: str) -> RoutingResult:
        """Fallback keyword-based routing when LLM parsing fails."""
        q = query.lower()

        # Complex task first (multiple verbs or "그리고", "및", "또한")
        # Check before doc_keywords since complex queries may also contain doc terms
        complex_keywords = ["그리고", "및", "또한", "비교", "분석", "정리"]
        if sum(1 for kw in complex_keywords if kw in q) >= 2:
            category = QueryCategory.COMPLEX_TASK
            return RoutingResult(
                category=category,
                confidence=0.5,
                reasoning="Keyword match: complex multi-step indicators",
                model_tier=_tier_for_category(category),
            )

        # Document search keywords
        doc_keywords = ["규정", "매뉴얼", "보고서", "절차", "기준", "문서", "안전", "점검", "가스"]
        if any(kw in q for kw in doc_keywords):
            category = QueryCategory.DOCUMENT_SEARCH
            return RoutingResult(
                category=category,
                confidence=0.6,
                reasoning="Keyword match: document-related terms detected",
                model_tier=_tier_for_category(category),
            )

        # Chitchat
        chat_keywords = ["안녕", "감사", "고마워", "ㅎㅎ", "ㅋㅋ", "반가", "잘가"]
        if any(kw in q for kw in chat_keywords):
            category = QueryCategory.CHITCHAT
            return RoutingResult(
                category=category,
                confidence=0.8,
                reasoning="Keyword match: greeting/chitchat detected",
                model_tier=_tier_for_category(category),
            )

        # Default to general
        category = QueryCategory.GENERAL_QUERY
        return RoutingResult(
            category=category,
            confidence=0.4,
            reasoning="No strong signal; defaulting to general query",
            model_tier=_tier_for_category(category),
        )
