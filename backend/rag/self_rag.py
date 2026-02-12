"""Self-RAG: Self-Reflective Retrieval-Augmented Generation.

After generating an answer, evaluates groundedness and relevance.
If the answer contains hallucinations, retries with a stricter prompt.
"""

import json
import logging
from dataclasses import dataclass

from core.llm import BaseLLM, create_llm
from rag.prompt import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """Result of self-RAG grading."""
    grounded: bool  # Is the answer grounded in context?
    relevance: str  # "high", "medium", "low"
    hallucination: bool  # Does the answer contain hallucinated content?
    reasoning: str  # Explanation of the grading
    confidence: float  # 0.0 ~ 1.0
    passed: bool  # Overall pass/fail


class SelfRAGEvaluator:
    """Evaluates and potentially retries RAG-generated answers.

    Self-RAG Process:
    1. Grade the generated answer for groundedness and hallucination
    2. If grading fails, regenerate with a stricter prompt
    3. Return the best answer with grading metadata
    """

    # Stricter prompt suffix to reduce hallucination
    STRICT_SUFFIX = (
        "\n\n[중요 제약]\n"
        "- 위 컨텍스트에 명시적으로 있는 내용만 답변하세요.\n"
        "- 추론이나 추정을 하지 마세요.\n"
        "- 컨텍스트에서 직접 인용하여 답변하세요.\n"
        "- 확실하지 않은 내용은 '컨텍스트에서 확인할 수 없습니다'라고 답변하세요."
    )

    def __init__(
        self,
        llm: BaseLLM | None = None,
        prompt_manager: PromptManager | None = None,
        max_retries: int = 1,
    ):
        self._llm = llm
        self._prompt_manager = prompt_manager
        self.max_retries = max_retries

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

    async def grade_answer(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> GradingResult:
        """Grade an answer for groundedness and hallucination.

        Args:
            query: Original question.
            answer: Generated answer to evaluate.
            context: The context chunks that were provided.

        Returns:
            GradingResult with grounding assessment.
        """
        prompt = self.prompt_manager.get_system_prompt("self_rag_grading").format(
            context=context,
            query=query,
            answer=answer,
        )

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistent grading
                max_tokens=512,
            )

            content = response.content.strip()
            # Extract JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                grounded = data.get("grounded", False)
                hallucination = data.get("hallucination", False)
                relevance = data.get("relevance", "low")
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning", "")

                passed = grounded and not hallucination and relevance in ("high", "medium")

                return GradingResult(
                    grounded=grounded,
                    relevance=relevance,
                    hallucination=hallucination,
                    reasoning=reasoning,
                    confidence=confidence,
                    passed=passed,
                )

        except Exception as e:
            logger.warning("Self-RAG grading failed: %s", e)

        # Default: pass (don't block on grading failure)
        return GradingResult(
            grounded=True,
            relevance="medium",
            hallucination=False,
            reasoning="Grading failed, defaulting to pass",
            confidence=0.5,
            passed=True,
        )

    async def evaluate_and_retry(
        self,
        query: str,
        answer: str,
        context: str,
        system_prompt: str,
        user_prompt: str,
        llm: BaseLLM | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict]:
        """Evaluate answer and retry if it fails grading.

        Args:
            query: Original question.
            answer: Initial generated answer.
            context: Context string from retrieved chunks.
            system_prompt: System prompt used for generation.
            user_prompt: User prompt used for generation.
            llm: LLM to use for regeneration.
            temperature: Temperature for regeneration.
            max_tokens: Max tokens for regeneration.

        Returns:
            Tuple of (final_answer, grading_metadata).
        """
        llm = llm or self.llm
        grading_metadata = {"self_rag_enabled": True, "attempts": 1}

        # Grade the initial answer
        grade = await self.grade_answer(query, answer, context)
        grading_metadata["initial_grade"] = {
            "grounded": grade.grounded,
            "relevance": grade.relevance,
            "hallucination": grade.hallucination,
            "confidence": grade.confidence,
            "passed": grade.passed,
        }

        if grade.passed:
            logger.info("Self-RAG: initial answer passed grading (confidence=%.2f)", grade.confidence)
            grading_metadata["final_passed"] = True
            return answer, grading_metadata

        # Retry with stricter prompt
        logger.info(
            "Self-RAG: initial answer failed grading (grounded=%s, hallucination=%s). Retrying...",
            grade.grounded, grade.hallucination,
        )

        retry_answer = answer  # fallback if max_retries == 0
        for retry in range(self.max_retries):
            strict_system = system_prompt + self.STRICT_SUFFIX
            response = await llm.generate(
                prompt=user_prompt,
                system=strict_system,
                temperature=max(0.1, (temperature or 0.7) * 0.5),  # Reduce temperature
                max_tokens=max_tokens,
            )

            retry_answer = response.content.strip()
            retry_grade = await self.grade_answer(query, retry_answer, context)

            grading_metadata["attempts"] = retry + 2
            grading_metadata[f"retry_{retry + 1}_grade"] = {
                "grounded": retry_grade.grounded,
                "relevance": retry_grade.relevance,
                "hallucination": retry_grade.hallucination,
                "confidence": retry_grade.confidence,
                "passed": retry_grade.passed,
            }

            if retry_grade.passed:
                logger.info("Self-RAG: retry %d passed grading", retry + 1)
                grading_metadata["final_passed"] = True
                return retry_answer, grading_metadata

        # All retries failed - return the retry answer with warning
        logger.warning("Self-RAG: all retries failed grading, using last attempt")
        grading_metadata["final_passed"] = False
        return retry_answer, grading_metadata
