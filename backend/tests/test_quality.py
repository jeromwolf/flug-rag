"""Tests for quality control module."""

import pytest

from rag.quality import QualityController


class TestQualityController:

    def setup_method(self):
        self.qc = QualityController(confidence_high=0.8, confidence_low=0.5)

    def test_high_confidence(self):
        scores = [0.95, 0.90, 0.88]
        confidence = self.qc.calculate_confidence(scores)
        assert confidence > 0.8

    def test_medium_confidence(self):
        scores = [0.7, 0.6, 0.5]
        confidence = self.qc.calculate_confidence(scores)
        assert 0.4 < confidence < 0.8

    def test_low_confidence(self):
        scores = [0.3, 0.2, 0.1]
        confidence = self.qc.calculate_confidence(scores)
        assert confidence < 0.5

    def test_empty_scores(self):
        assert self.qc.calculate_confidence([]) == 0.0

    def test_single_score(self):
        confidence = self.qc.calculate_confidence([0.9])
        assert confidence == 0.9

    def test_confidence_level_high(self):
        assert self.qc.get_confidence_level(0.9) == "high"

    def test_confidence_level_medium(self):
        assert self.qc.get_confidence_level(0.6) == "medium"

    def test_confidence_level_low(self):
        assert self.qc.get_confidence_level(0.3) == "low"

    def test_safety_warning_needed(self):
        assert self.qc.should_add_safety_warning(0.3) is True
        assert self.qc.should_add_safety_warning(0.6) is False
        assert self.qc.should_add_safety_warning(0.9) is False

    def test_safety_message(self):
        msg = self.qc.get_safety_message(0.3)
        assert msg is not None
        assert "확실하지 않은" in msg

    def test_no_safety_message(self):
        assert self.qc.get_safety_message(0.8) is None

    def test_response_metadata(self):
        meta = self.qc.build_response_metadata(
            confidence=0.85,
            sources=[{"id": "1"}],
            model_used="vllm/qwen2.5",
            latency_ms=2341,
        )
        assert meta["confidence_score"] == 0.85
        assert meta["confidence_level"] == "high"
        assert meta["source_count"] == 1
        assert meta["latency_ms"] == 2341
        assert meta["has_safety_warning"] is False
