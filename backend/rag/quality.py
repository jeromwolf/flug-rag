"""Quality control: confidence scoring, safety guards, golden data integration."""

import numpy as np

from config.settings import settings


class QualityController:
    """Manages answer quality assessment and safety guards."""

    def __init__(
        self,
        confidence_high: float | None = None,
        confidence_low: float | None = None,
    ):
        self.confidence_high = confidence_high if confidence_high is not None else settings.confidence_high
        self.confidence_low = confidence_low if confidence_low is not None else settings.confidence_low

    def calculate_confidence(self, chunk_scores: list[float]) -> float:
        """Calculate confidence score from retrieval scores.

        Uses top-3 chunk scores average with consistency penalty.

        Args:
            chunk_scores: List of relevance scores from retrieval.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not chunk_scores:
            return 0.0

        # Top 3 scores
        top_scores = sorted(chunk_scores, reverse=True)[:3]
        avg_score = sum(top_scores) / len(top_scores)

        # Consistency penalty: high std dev = less reliable
        if len(top_scores) > 1:
            std_dev = float(np.std(top_scores))
            consistency = 1.0 - min(std_dev, 0.2)
        else:
            consistency = 1.0

        return min(avg_score * consistency, 1.0)

    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level label.

        Returns: 'high', 'medium', or 'low'
        """
        if confidence >= self.confidence_high:
            return "high"
        elif confidence >= self.confidence_low:
            return "medium"
        return "low"

    def should_add_safety_warning(self, confidence: float) -> bool:
        """Check if answer needs a safety warning."""
        return confidence < self.confidence_low

    def get_safety_message(self, confidence: float) -> str | None:
        """Get safety warning message for low-confidence answers."""
        if confidence >= self.confidence_low:
            return None

        return (
            "⚠️ 확실하지 않은 답변입니다.\n\n"
            "검색된 문서에서 명확한 정보를 찾지 못했습니다.\n"
            "다음 방법을 시도해보세요:\n"
            "1. 질문을 더 구체적으로 작성\n"
            "2. 관련 문서가 업로드되었는지 확인\n"
            "3. 관리자에게 문의\n"
        )

    def build_response_metadata(
        self,
        confidence: float,
        sources: list[dict],
        model_used: str,
        latency_ms: int,
        response_mode: str = "rag",
    ) -> dict:
        """Build metadata for the response."""
        return {
            "confidence_score": round(confidence, 3),
            "confidence_level": self.get_confidence_level(confidence),
            "source_count": len(sources),
            "model_used": model_used,
            "latency_ms": latency_ms,
            "response_mode": response_mode,
            "has_safety_warning": self.should_add_safety_warning(confidence),
        }
