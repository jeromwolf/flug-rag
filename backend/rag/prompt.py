"""Prompt template management with YAML-based templates and few-shot support."""

from __future__ import annotations

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
        self._legal_examples: list[dict] = []
        self._technical_examples: list[dict] = []
        self._general_examples: list[dict] = []
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
                self._legal_examples = data.get("legal_examples", [])
                self._technical_examples = data.get("technical_examples", [])
                self._general_examples = data.get("general_examples", [])

    def get_system_prompt(self, name: str) -> str:
        """Get a system prompt by name."""
        if name not in self._system_prompts:
            raise KeyError(f"System prompt not found: {name}. Available: {list(self._system_prompts.keys())}")
        return self._system_prompts[name]

    @staticmethod
    def detect_document_type(context_chunks: list[dict]) -> str:
        """Detect document type from retrieved context metadata.

        Returns:
            "legal" for 법률/규정 documents,
            "technical" for 기술/매뉴얼/사내문서,
            "general" for unclassified (홍보물, 출장보고서, ALIO공시 등).
        """
        import re

        legal_score = 0
        technical_score = 0
        general_score = 0

        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            filename = (metadata.get("filename", "") or "").lower()
            category = metadata.get("category", "") or ""
            source_type = metadata.get("source_type", "") or ""
            content = chunk.get("content", "") or ""

            # General signals (공시, 홍보물, 출장보고서)
            if source_type in ("홍보물", "출장보고서", "ALIO공시"):
                general_score += 3
            if category in ("brochure", "travel_report", "public_disclosure"):
                general_score += 2

            # Legal signals
            if category == "규정":
                legal_score += 2
            if source_type in ("법률", "내부규정", "정관"):
                legal_score += 3
            if re.search(r'(법|시행령|시행규칙|조례|규정)', filename):
                legal_score += 2
            if re.search(r'제\d+조', content):
                legal_score += 1

            # Technical signals
            if category in ("매뉴얼", "점검", "계획서", "교육"):
                technical_score += 2
            if re.search(r'(매뉴얼|지침서|절차서|안내서|점검표)', filename):
                technical_score += 2
            if re.search(r'(점검|측정|설비|배관|밸브)', content) and not re.search(r'제\d+조', content):
                technical_score += 1

        if legal_score > technical_score and legal_score > general_score:
            return "legal"
        elif technical_score > legal_score and technical_score > general_score:
            return "technical"
        return "general"

    def _get_domain_examples(self, doc_type: str) -> list[dict]:
        """Get few-shot examples for the detected document type."""
        if doc_type == "legal" and self._legal_examples:
            return self._legal_examples
        elif doc_type == "technical" and self._technical_examples:
            return self._technical_examples
        elif doc_type == "general" and self._general_examples:
            return self._general_examples
        return self._general_examples or self._few_shot_examples

    @staticmethod
    def _get_conciseness_suffix(model_hint: str | None = None) -> str:
        """Get extra conciseness instruction based on model size."""
        if not model_hint:
            return ""
        model_lower = model_hint.lower()
        small_patterns = ["7b", "8b", "3b", "1b", "1.5b", "mini", "tiny", "small"]
        medium_patterns = ["14b", "13b", "12b", "15b", "20b"]
        if any(p in model_lower for p in small_patterns):
            return (
                "\n\n[추가 지시 - 반드시 따르세요]\n"
                "- 반드시 1~3문장으로만 답변하세요.\n"
                "- 법 조문의 원문을 그대로 인용하세요.\n"
                "- 추가 해석이나 부연 설명을 절대 덧붙이지 마세요."
            )
        if any(p in model_lower for p in medium_patterns):
            return (
                "\n\n[추가 지시 - 반드시 따르세요]\n"
                "- 사실(factual) 질문: 50~150자로 간결하게 답변하세요.\n"
                "- 부정(negative) 질문: 100~250자로, '확인되지 않습니다'와 함께 이유를 설명하세요.\n"
                "- 핵심 사실만 답변하고, 컨텍스트의 원문 표현을 그대로 사용하세요.\n"
                "- 불필요한 서론, 반복, 부연 설명을 삼가세요."
            )
        return ""

    def build_rag_prompt(
        self,
        query: str,
        context_chunks: list[dict],
        few_shot: list[dict] | None = None,
        model_hint: str | None = None,
    ) -> tuple[str, str]:
        """Build a complete RAG prompt with context and few-shot examples.

        Auto-detects document type from context metadata and selects
        the appropriate system prompt and few-shot examples.

        Args:
            query: User's question.
            context_chunks: List of {content, metadata} dicts from retrieval.
            few_shot: Optional few-shot examples override.
            model_hint: Optional model name for model-size-aware prompting.

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

        # Detect document type and select prompt
        doc_type = self.detect_document_type(context_chunks)
        prompt_map = {
            "legal": "rag_legal_system",
            "technical": "rag_technical_system",
        }
        prompt_name = prompt_map.get(doc_type, "rag_system")
        if prompt_name not in self._system_prompts:
            prompt_name = "rag_system"

        system = self.get_system_prompt(prompt_name).format(context=context)
        system += self._get_conciseness_suffix(model_hint)

        # Select domain-appropriate few-shot examples
        examples = few_shot or self._get_domain_examples(doc_type)
        user_parts = []

        if examples:
            user_parts.append("참고 예시:")
            for ex in examples[:settings.few_shot_max_examples]:
                user_parts.append(f"질문: {ex['question']}\n답변: {ex['answer']}")
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
