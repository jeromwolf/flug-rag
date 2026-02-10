"""RAG 답변 품질 평가 모듈.

의미 유사도(Semantic Similarity) + ROUGE 점수로 답변 정확도를 측정한다.
기존 bge-m3 임베딩 인프라를 재활용한다.
"""

import asyncio
import logging
import re
from dataclasses import dataclass

import numpy as np
from rouge_score import rouge_scorer

from core.embeddings.local import LocalEmbedding

logger = logging.getLogger(__name__)


@dataclass
class AnswerEvaluation:
    """단일 답변 평가 결과."""

    semantic_similarity: float  # 코사인 유사도 (0~1)
    rouge_1: float  # ROUGE-1 F1
    rouge_l: float  # ROUGE-L F1
    answer_length: int  # 생성 답변 길이
    expected_length: int  # 골든 답변 길이
    length_ratio: float  # 길이 비율
    composite_score: float  # 종합 점수 (가중 평균)
    grade: str  # A/B/C/D/F 등급


class _KoreanCharTokenizer:
    """Character-level Korean tokenizer for ROUGE.

    Korean is agglutinative (particles attach to nouns), so character-level
    matching works much better than word-level for ROUGE scoring.
    Also strips common LLM artifacts (출처 tags, A: prefix) before scoring.
    """

    def tokenize(self, text: str) -> list[str]:
        text = text.strip()
        # Remove citation suffixes and Q/A prefixes
        text = re.sub(r"\[출처[^\]]*\]", "", text)
        text = re.sub(r"^[AQ][:：]\s*", "", text)
        # Keep Korean chars, digits, ASCII letters
        return [c for c in text if re.match(r"[\uAC00-\uD7A3a-zA-Z0-9]", c)]


class AnswerEvaluator:
    """RAG 답변 품질 평가기.

    - 의미 유사도: bge-m3 임베딩 → cosine similarity
    - ROUGE: rouge-score 라이브러리로 ROUGE-1, ROUGE-L F1 (한국어 문자 수준)
    - 종합 점수: 0.5 * semantic + 0.3 * rouge_l + 0.2 * rouge_1
    - 등급: A(>=0.8), B(>=0.65), C(>=0.5), D(>=0.35), F(<0.35)
    """

    # 종합 점수 가중치
    W_SEMANTIC = 0.5
    W_ROUGE_L = 0.3
    W_ROUGE_1 = 0.2

    # 등급 기준
    GRADE_THRESHOLDS = [
        (0.8, "A"),
        (0.65, "B"),
        (0.5, "C"),
        (0.35, "D"),
    ]

    def __init__(self):
        self._embedder = LocalEmbedding()
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], tokenizer=_KoreanCharTokenizer(),
        )

    @staticmethod
    def _compute_grade(score: float) -> str:
        for threshold, grade in AnswerEvaluator.GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "F"

    def _score_pair(
        self,
        vec_expected: np.ndarray,
        vec_actual: np.ndarray,
        expected: str,
        actual: str,
    ) -> AnswerEvaluation:
        """Compute all metrics for a single (expected, actual) pair."""
        semantic_sim = float(np.dot(vec_expected, vec_actual))
        semantic_sim = max(0.0, min(1.0, semantic_sim))

        scores = self._rouge.score(expected, actual)
        rouge_1 = scores["rouge1"].fmeasure
        rouge_l = scores["rougeL"].fmeasure

        expected_len = len(expected)
        actual_len = len(actual)
        length_ratio = actual_len / expected_len if expected_len > 0 else 0.0

        composite = (
            self.W_SEMANTIC * semantic_sim
            + self.W_ROUGE_L * rouge_l
            + self.W_ROUGE_1 * rouge_1
        )

        return AnswerEvaluation(
            semantic_similarity=round(semantic_sim, 4),
            rouge_1=round(rouge_1, 4),
            rouge_l=round(rouge_l, 4),
            answer_length=actual_len,
            expected_length=expected_len,
            length_ratio=round(length_ratio, 2),
            composite_score=round(composite, 4),
            grade=self._compute_grade(composite),
        )

    async def evaluate(self, expected: str, actual: str) -> AnswerEvaluation:
        """골든 답변 vs 실제 답변 평가.

        Args:
            expected: 골든 답변 (정답)
            actual: RAG가 생성한 답변

        Returns:
            AnswerEvaluation with all metrics
        """
        # empty/whitespace guard
        if not expected.strip() or not actual.strip():
            return AnswerEvaluation(
                semantic_similarity=0.0,
                rouge_1=0.0,
                rouge_l=0.0,
                answer_length=len(actual),
                expected_length=len(expected),
                length_ratio=0.0,
                composite_score=0.0,
                grade="F",
            )

        embs = await self._embedder.embed_texts([expected, actual])
        vec_expected = np.array(embs[0])
        vec_actual = np.array(embs[1])
        return self._score_pair(vec_expected, vec_actual, expected, actual)

    async def evaluate_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[AnswerEvaluation]:
        """배치 평가 (임베딩을 한꺼번에 처리하여 속도 최적화).

        Args:
            pairs: [(expected, actual), ...] 튜플 리스트

        Returns:
            AnswerEvaluation 리스트
        """
        if not pairs:
            return []

        # Separate valid and empty pairs
        valid_indices = []
        results: list[AnswerEvaluation | None] = [None] * len(pairs)
        texts_to_embed = []

        for i, (expected, actual) in enumerate(pairs):
            if not expected.strip() or not actual.strip():
                results[i] = AnswerEvaluation(
                    semantic_similarity=0.0,
                    rouge_1=0.0,
                    rouge_l=0.0,
                    answer_length=len(actual),
                    expected_length=len(expected),
                    length_ratio=0.0,
                    composite_score=0.0,
                    grade="F",
                )
            else:
                valid_indices.append(i)
                texts_to_embed.append(expected)
                texts_to_embed.append(actual)

        if texts_to_embed:
            all_embs = await self._embedder.embed_texts(texts_to_embed)
            for j, idx in enumerate(valid_indices):
                expected, actual = pairs[idx]
                vec_expected = np.array(all_embs[j * 2])
                vec_actual = np.array(all_embs[j * 2 + 1])
                results[idx] = self._score_pair(vec_expected, vec_actual, expected, actual)

        return results  # type: ignore[return-value]

    @staticmethod
    def summarize(evaluations: list[AnswerEvaluation]) -> dict:
        """평가 결과 요약 통계.

        Returns:
            {
                "avg_composite": float,
                "avg_semantic": float,
                "avg_rouge_1": float,
                "avg_rouge_l": float,
                "grade_distribution": {"A": int, "B": int, ...},
            }
        """
        if not evaluations:
            return {}

        n = len(evaluations)
        grade_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

        total_composite = 0.0
        total_semantic = 0.0
        total_rouge_1 = 0.0
        total_rouge_l = 0.0

        for e in evaluations:
            total_composite += e.composite_score
            total_semantic += e.semantic_similarity
            total_rouge_1 += e.rouge_1
            total_rouge_l += e.rouge_l
            grade_dist[e.grade] = grade_dist.get(e.grade, 0) + 1

        return {
            "avg_composite": round(total_composite / n, 4),
            "avg_semantic": round(total_semantic / n, 4),
            "avg_rouge_1": round(total_rouge_1 / n, 4),
            "avg_rouge_l": round(total_rouge_l / n, 4),
            "grade_distribution": grade_dist,
        }
