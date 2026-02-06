"""Prompt template management with YAML-based templates and few-shot support."""

from pathlib import Path
from typing import Any

import yaml

from config.settings import settings


class PromptManager:
    """Manages prompt templates loaded from YAML files."""

    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or settings.prompts_dir
        self._system_prompts: dict[str, str] = {}
        self._few_shot_examples: list[dict] = []
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from YAML files."""
        system_path = self.prompts_dir / "system.yaml"
        if system_path.exists():
            with open(system_path, "r", encoding="utf-8") as f:
                self._system_prompts = yaml.safe_load(f) or {}

        few_shot_path = self.prompts_dir / "few_shot.yaml"
        if few_shot_path.exists():
            with open(few_shot_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self._few_shot_examples = data.get("examples", [])

    def get_system_prompt(self, name: str) -> str:
        """Get a system prompt by name."""
        if name not in self._system_prompts:
            raise KeyError(f"System prompt not found: {name}. Available: {list(self._system_prompts.keys())}")
        return self._system_prompts[name]

    def build_rag_prompt(
        self,
        query: str,
        context_chunks: list[dict],
        few_shot: list[dict] | None = None,
    ) -> tuple[str, str]:
        """Build a complete RAG prompt with context and few-shot examples.

        Args:
            query: User's question.
            context_chunks: List of {content, metadata} dicts from retrieval.
            few_shot: Optional few-shot examples. If None, uses loaded examples.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        # Build context string
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("metadata", {}).get("filename", "Unknown")
            page = chunk.get("metadata", {}).get("page_number", "")
            page_str = f" (p.{page})" if page else ""
            context_parts.append(f"[문서 {i}] {source}{page_str}\n{chunk['content']}")

        context = "\n\n".join(context_parts)

        # System prompt with context
        system = self.get_system_prompt("rag_system").format(context=context)

        # Build user prompt with few-shot
        examples = few_shot or self._few_shot_examples
        user_parts = []

        if examples:
            user_parts.append("참고 예시:")
            for ex in examples[:2]:  # Max 2 examples
                user_parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")
            user_parts.append("")

        user_parts.append(f"질문: {query}")
        user_prompt = "\n\n".join(user_parts)

        return system, user_prompt

    def build_direct_prompt(self, query: str) -> tuple[str, str]:
        """Build a prompt for direct LLM response (no RAG)."""
        system = self.get_system_prompt("direct_system")
        return system, query

    def build_router_prompt(self, query: str) -> tuple[str, str]:
        """Build a prompt for question routing."""
        system = self.get_system_prompt("router_system")
        return system, f"질문: {query}"

    def build_rewrite_prompt(self, query: str, history: list[dict]) -> tuple[str, str]:
        """Build a prompt for query rewriting."""
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in history[-6:]  # Last 3 turns
        )
        system = self.get_system_prompt("query_rewrite").format(
            history=history_text,
            query=query,
        )
        return system, query

    def reload(self):
        """Reload prompts from disk."""
        self._load_prompts()
