"""Agentic RAG: Dynamic strategy routing based on question analysis.

Analyzes user intent and selects the optimal RAG strategy:
- standard_rag: Simple fact-finding questions
- deep_retrieval: Complex/multi-document questions needing thorough search
- direct_llm: General knowledge or chitchat (no retrieval needed)

Routing is driven by `prompts/routing_rules.yaml` — no code changes needed
to adjust rules.  The rules file is hot-reloaded on modification.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from config.settings import settings
from core.llm import BaseLLM, create_llm
from rag.prompt import PromptManager

logger = logging.getLogger(__name__)

_RULES_PATH = Path(__file__).parent.parent / "prompts" / "routing_rules.yaml"

# Module-level cache so rules survive individual route() calls
_rules_cache: dict | None = None
_rules_mtime: float = 0.0

# Pattern for detecting Korean regulation names (e.g. "여비업무처리지침", "안전관리규정")
_REGULATION_PATTERN = re.compile(
    r"[가-힣]{2,15}(?:지침|규정|규칙|세칙|요령|매뉴얼|운영규정|관리규정|처리지침)"
)


def _load_rules() -> dict:
    """Load routing rules from YAML, hot-reloading when the file changes."""
    global _rules_cache, _rules_mtime
    try:
        mtime = _RULES_PATH.stat().st_mtime
        if _rules_cache is None or mtime > _rules_mtime:
            with open(_RULES_PATH, encoding="utf-8") as f:
                _rules_cache = yaml.safe_load(f)
            _rules_mtime = mtime
            logger.info("Routing rules loaded/reloaded from %s", _RULES_PATH)
    except Exception as e:
        logger.warning("Failed to load routing rules: %s", e)
        if _rules_cache is None:
            _rules_cache = {"strategies": {}, "rules": []}
    return _rules_cache  # type: ignore[return-value]


def _count_regulations(query: str) -> int:
    """Count the number of distinct regulation names mentioned in the query."""
    return len(set(_REGULATION_PATTERN.findall(query)))


def _match_rule(rule: dict, query: str, query_len: int, reg_count: int) -> bool:
    """Return True if *query* satisfies all conditions in *rule*."""
    conditions = rule.get("conditions", {})
    if not conditions:
        return True  # catch-all

    if "max_length" in conditions and query_len > conditions["max_length"]:
        return False
    if "min_length" in conditions and query_len < conditions["min_length"]:
        return False
    if "min_regulation_count" in conditions and reg_count < conditions["min_regulation_count"]:
        return False
    if "patterns" in conditions:
        if not any(re.search(p, query, re.IGNORECASE) for p in conditions["patterns"]):
            return False
    return True


@dataclass
class RoutingDecision:
    """Result of agentic routing analysis."""

    strategy: str  # standard_rag, deep_retrieval, direct_llm, …
    confidence: float
    reasoning: str
    params: dict  # Strategy parameters (mode, use_hyde, use_self_rag, top_k, temperature, …)


class AgenticRAGRouter:
    """Analyzes questions and routes to optimal RAG strategy.

    Routing logic is entirely driven by `prompts/routing_rules.yaml`.
    No LLM call is made — routing latency is ~0 ms.

    Strategies defined in the YAML:
    - standard_rag: Default hybrid search, no HyDE/Self-RAG
    - deep_retrieval: HyDE + Self-RAG enabled, higher top_k
    - direct_llm:   Skip retrieval entirely
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        prompt_manager: PromptManager | None = None,
    ):
        # llm / prompt_manager kept for backward-compatibility but not used for routing
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
        """Rule-based routing only (no LLM call, ~0 ms latency).

        Args:
            query: User's question.

        Returns:
            RoutingDecision with strategy and parameter overrides.
        """
        return self._rule_based_route(query)

    @staticmethod
    def _rule_based_route(query: str) -> RoutingDecision:
        """Evaluate YAML rules in order and return first matching decision."""
        rules_config = _load_rules()
        strategies: dict = rules_config.get("strategies", {})
        rules: list = rules_config.get("rules", [])

        query_stripped = query.strip()
        query_len = len(query_stripped)
        reg_count = _count_regulations(query_stripped)

        for rule in rules:
            if _match_rule(rule, query_stripped, query_len, reg_count):
                strategy_name: str = rule["strategy"]
                strategy_params = dict(strategies.get(strategy_name, {}))
                strategy_params.pop("description", None)

                logger.info(
                    "Route: '%s' → %s (%s, regs=%d)",
                    query_stripped[:40],
                    strategy_name,
                    rule["name"],
                    reg_count,
                )
                return RoutingDecision(
                    strategy=strategy_name,
                    confidence=rule.get("confidence", 0.7),
                    reasoning=rule["name"],
                    params=strategy_params,
                )

        # Fallback — should never be reached because the YAML has a catch-all rule
        logger.warning("No routing rule matched for query '%s', using standard_rag", query_stripped[:40])
        fallback_params = dict(
            strategies.get(
                "standard_rag",
                {"mode": "rag", "use_hyde": False, "use_self_rag": False},
            )
        )
        fallback_params.pop("description", None)
        return RoutingDecision(
            strategy="standard_rag",
            confidence=0.5,
            reasoning="No rule matched, default standard_rag",
            params=fallback_params,
        )

    def get_strategy_params(self, decision: RoutingDecision) -> dict:
        """Get final parameters for the chosen strategy.

        Merges strategy defaults with any custom params from routing.
        """
        return decision.params
