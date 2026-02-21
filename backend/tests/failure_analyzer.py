"""벤치마크 실패 패턴 분석기.

benchmark_all.py 결과 JSON을 읽어 실패 문항의 패턴을 분류하고,
각 패턴별 수정 제안을 생성한다. 독립 실행 가능한 스탠드얼론 스크립트.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# sys.path 설정 (프로젝트 내 다른 테스트 파일과 동일한 패턴)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 카테고리/신뢰도 임계값 (benchmark_all.py 와 동일)
# ---------------------------------------------------------------------------
CATEGORY_THRESHOLDS = {
    "factual": 0.45,
    "inference": 0.45,
    "multi_hop": 0.43,
    "negative": 0.40,
}
DEFAULT_THRESHOLD = 0.50

CONFIDENCE_THRESHOLDS = {
    "factual": 0.15,
    "inference": 0.15,
    "multi_hop": 0.05,
    "negative": 0.0,
}
DEFAULT_CONFIDENCE = 0.15


# ---------------------------------------------------------------------------
# 실패 패턴 열거형 (10가지)
# ---------------------------------------------------------------------------
class FailurePattern(Enum):
    """벤치마크 실패 패턴 분류."""

    VERBOSE_ANSWER = "verbose_answer"          # 답변 과잉 - 불필요하게 긴 답변
    HALLUCINATION = "hallucination"            # 환각 - 근거 없는 내용 생성
    WRONG_RETRIEVAL = "wrong_retrieval"        # 검색 오류 - 관련 없는 문서 검색
    INCOMPLETE_LIST = "incomplete_list"        # 목록 불완전 - 열거형 답변 누락
    WRONG_SCOPE = "wrong_scope"               # 범위 오류 - 규정 외 내용 참조
    WEAK_INFERENCE = "weak_inference"          # 추론 부족 - 단순 재진술만 수행
    LANGUAGE_LEAK = "language_leak"            # 언어 혼재 - 비한국어 문장 포함
    FORMAT_ERROR = "format_error"             # 형식 오류 - 불필요한 접두사 등
    LOW_CONFIDENCE = "low_confidence"          # 저신뢰도 - 명확한 패턴 없음
    MULTI_HOP_FAILURE = "multi_hop_failure"    # 다중 규정 연결 실패


# ---------------------------------------------------------------------------
# 패턴별 한국어 설명
# ---------------------------------------------------------------------------
PATTERN_DESCRIPTIONS: dict[FailurePattern, str] = {
    FailurePattern.VERBOSE_ANSWER: "답변이 지나치게 길어 핵심 정보가 희석됨",
    FailurePattern.HALLUCINATION: "근거 문서에 없는 내용을 생성 (환각)",
    FailurePattern.WRONG_RETRIEVAL: "질문과 무관한 문서가 검색됨",
    FailurePattern.INCOMPLETE_LIST: "열거형 답변에서 항목 누락",
    FailurePattern.WRONG_SCOPE: "질문 범위 밖의 규정/문서를 참조",
    FailurePattern.WEAK_INFERENCE: "규정 내용을 단순 재진술, 추론 부족",
    FailurePattern.LANGUAGE_LEAK: "한국어 외 언어(영어/중국어 등) 문장 혼재",
    FailurePattern.FORMAT_ERROR: "불필요한 접두사·형식 오류",
    FailurePattern.LOW_CONFIDENCE: "명확한 패턴 없이 전반적으로 저품질",
    FailurePattern.MULTI_HOP_FAILURE: "여러 문서/규정을 연결하지 못함",
}


# ---------------------------------------------------------------------------
# 패턴별 기본 수정 제안
# ---------------------------------------------------------------------------
PATTERN_FIX_MAP: dict[FailurePattern, list[str]] = {
    FailurePattern.VERBOSE_ANSWER: [
        "시스템 프롬프트에 '핵심만 간결하게 답변' 지시 강화",
        "few-shot 예시에서 짧은 답변 패턴 추가",
        "length_penalty 가중치를 높여 장문 답변 불이익 확대",
        "max_tokens 파라미터를 줄여 물리적 길이 제한",
    ],
    FailurePattern.HALLUCINATION: [
        "Self-RAG 활성화하여 자기반성 단계 추가",
        "시스템 프롬프트에 '검색된 문서에만 근거하여 답변' 강조",
        "confidence threshold 상향 조정",
        "가드레일에 환각 탐지 규칙 추가",
    ],
    FailurePattern.WRONG_RETRIEVAL: [
        "Retriever top_k 값을 늘려 후보 문서 확대",
        "Reranker 가중치 조정 (FlashRank 파라미터 튜닝)",
        "BM25 가중치 대비 벡터 검색 가중치 재조정",
        "Multi-Query 활성화하여 다중 관점 검색",
    ],
    FailurePattern.INCOMPLETE_LIST: [
        "프롬프트에 '모든 항목을 빠짐없이 나열' 지시 추가",
        "검색 top_k 확대로 더 많은 관련 청크 확보",
        "청크 크기를 키워 목록 항목이 분리되지 않도록 조정",
    ],
    FailurePattern.WRONG_SCOPE: [
        "negative 카테고리용 프롬프트 강화 ('해당 문서에 없으면 없다고 답변')",
        "few-shot에 '범위 밖' 답변 예시 추가",
        "가드레일에 범위 외 응답 필터 추가",
    ],
    FailurePattern.WEAK_INFERENCE: [
        "inference 카테고리용 few-shot 예시 강화 (추론 과정 포함)",
        "시스템 프롬프트에 '규정의 의미와 배경을 설명' 지시 추가",
        "CoT (Chain-of-Thought) 프롬프팅 도입",
    ],
    FailurePattern.LANGUAGE_LEAK: [
        "시스템 프롬프트에 '반드시 한국어로만 답변' 지시 강화",
        "출력 가드레일에 비한국어 필터 추가",
        "few-shot 예시를 순수 한국어로 통일",
    ],
    FailurePattern.FORMAT_ERROR: [
        "시스템 프롬프트에서 불필요한 접두사 사용 금지 명시",
        "후처리에서 'A:', '답변:' 등 접두사 자동 제거",
    ],
    FailurePattern.LOW_CONFIDENCE: [
        "전반적인 RAG 파이프라인 품질 점검",
        "임베딩 모델 성능 확인 (bge-m3 vs 대안)",
        "청크 품질 분석 실행 후 저품질 청크 정리",
    ],
    FailurePattern.MULTI_HOP_FAILURE: [
        "Agentic RAG 활성화하여 다단계 검색 수행",
        "Multi-Query 활성화하여 하위 질문 분리",
        "검색 top_k 대폭 확대 (10 → 15~20)",
        "프롬프트에 '여러 규정을 종합하여 답변' 지시 추가",
    ],
}


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------
@dataclass
class FailureAnalysis:
    """개별 실패 문항 분석 결과."""

    question_id: int
    dataset: str
    category: str
    difficulty: str
    question: str
    expected_answer: str
    actual_answer: str
    confidence: float
    sources_count: int
    semantic_similarity: float
    rouge_1: float
    rouge_l: float
    composite_score: float
    grade: str
    length_penalty: float
    category_score: float
    primary_pattern: str                     # FailurePattern.value
    secondary_patterns: list[str]            # [FailurePattern.value, ...]
    pattern_confidence: float                # 주 패턴 판별 확신도 (0~1)
    diagnosis: str                           # 한국어 진단 문장
    fix_suggestions: list[str] = field(default_factory=list)


@dataclass
class PatternSummary:
    """패턴별 집계 통계."""

    pattern: str                             # FailurePattern.value
    pattern_label: str                       # 한국어 설명
    count: int
    affected_categories: list[str]
    affected_difficulties: list[str]
    avg_semantic_similarity: float
    avg_rouge_l: float
    avg_composite_score: float
    avg_confidence: float
    fix_suggestions: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """전체 분석 리포트."""

    total_questions: int
    total_failures: int
    failure_rate: float
    pattern_summaries: list[PatternSummary]
    failures: list[FailureAnalysis]
    recommendations: list[dict]              # 우선순위별 권장사항
    parameter_suggestions: dict              # 파라미터 설정 제안


# ---------------------------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------------------------
def _has_list_markers(text: str) -> bool:
    """텍스트에 목록 마커(1. / - / * / ㄱ. / ① 등)가 있는지 확인."""
    return bool(re.search(r"(?:^|\n)\s*(?:\d+[\.\)]\s|[-\*\u2022\u25CB\u25CF]\s|[①②③④⑤⑥⑦⑧⑨⑩]|[ㄱ-ㅎ][\.\)])", text))


def _contains_non_korean_sentences(text: str) -> bool:
    """한국어가 아닌 완전한 문장이 포함되어 있는지 확인.

    - 영어 문장 (3단어 이상)
    - CJK 범위 중 한국어가 아닌 문자 블록
    """
    # 영어 문장 탐지 (3단어 이상 연속 영어)
    if re.search(r"[A-Z][a-z]+(?:\s+[a-z]+){2,}", text):
        return True

    # 중국어 간체/번체, 일본어 히라가나/카타카나 블록 탐지
    for char in text:
        cp = ord(char)
        cat = unicodedata.category(char)
        if cat.startswith("Lo"):  # Letter, other
            # 한글 음절 범위: U+AC00 ~ U+D7AF
            # 한글 자모 범위: U+1100 ~ U+11FF, U+3130 ~ U+318F
            if not (0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F):
                # CJK 통합 한자 (한국어에서도 사용하므로 허용)
                if 0x4E00 <= cp <= 0x9FFF:
                    continue
                # 일본어 히라가나/카타카나
                if 0x3040 <= cp <= 0x30FF:
                    return True
    return False


def _has_format_prefix(text: str) -> bool:
    """불필요한 답변 접두사가 있는지 확인."""
    stripped = text.strip()
    prefixes = ["A:", "A :", "답변:", "답변 :", "Answer:", "Answer :", "답:", "답 :"]
    for prefix in prefixes:
        if stripped.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# 메인 분석기 클래스
# ---------------------------------------------------------------------------
class FailurePatternAnalyzer:
    """벤치마크 실패 패턴 분석기.

    benchmark_all.py가 생성한 결과 JSON을 읽어,
    실패한 문항들의 패턴을 분류하고 수정 제안을 생성한다.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ----- 메인 분석 진입점 -----

    def analyze(self, results_path: str | Path) -> AnalysisReport:
        """벤치마크 결과 파일을 분석하여 AnalysisReport를 반환한다.

        Args:
            results_path: benchmark_all.py 결과 JSON 경로

        Returns:
            AnalysisReport: 전체 분석 리포트
        """
        results_path = Path(results_path)
        if not results_path.exists():
            raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {results_path}")

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_results = data.get("results", [])
        total_questions = len(all_results)

        # 실패 문항 필터링
        failures_raw = [r for r in all_results if not r.get("success", True)]
        total_failures = len(failures_raw)

        if total_failures == 0:
            # 실패 없음 - 빈 리포트 반환
            return AnalysisReport(
                total_questions=total_questions,
                total_failures=0,
                failure_rate=0.0,
                pattern_summaries=[],
                failures=[],
                recommendations=[{
                    "priority": 1,
                    "title": "모든 문항 통과",
                    "description": "현재 벤치마크에서 실패한 문항이 없습니다. 임계값을 높이거나 골든 데이터셋을 확장하세요.",
                    "action": "임계값 상향 또는 데이터셋 확장",
                }],
                parameter_suggestions={},
            )

        failure_rate = total_failures / total_questions if total_questions > 0 else 0.0

        # 각 실패 문항 분류
        failure_analyses: list[FailureAnalysis] = []
        for result in failures_raw:
            analysis = self._classify_failure(result)
            failure_analyses.append(analysis)

        # 패턴별 집계
        pattern_summaries = self._aggregate_patterns(failure_analyses)

        # 상위 권장사항 생성
        recommendations = self._generate_recommendations(pattern_summaries)

        # 파라미터 설정 제안
        parameter_suggestions = self._suggest_parameters(pattern_summaries)

        return AnalysisReport(
            total_questions=total_questions,
            total_failures=total_failures,
            failure_rate=round(failure_rate, 4),
            pattern_summaries=pattern_summaries,
            failures=failure_analyses,
            recommendations=recommendations,
            parameter_suggestions=parameter_suggestions,
        )

    # ----- 개별 문항 패턴 분류 -----

    def _classify_failure(self, result: dict) -> FailureAnalysis:
        """단일 실패 문항의 패턴을 점수 기반으로 분류한다.

        각 패턴에 대해 점수를 매기고, 최고 점수 패턴을 primary_pattern으로,
        점수 > 0인 나머지를 secondary_patterns으로 설정한다.
        """
        scores: dict[FailurePattern, int] = {p: 0 for p in FailurePattern}

        category = result.get("category", "")
        confidence = result.get("confidence", 0.0)
        semantic = result.get("semantic_similarity", 0.0)
        rouge_l = result.get("rouge_l", 0.0)
        length_penalty = result.get("length_penalty", 1.0)
        composite_score = result.get("composite_score", 0.0)
        category_score = result.get("category_score", 0.0)
        sources_count = result.get("sources_count", 0)
        expected = result.get("expected_answer", "")
        actual = result.get("actual_answer", "")

        # --- VERBOSE_ANSWER: 답변 과잉 ---
        if length_penalty < 0.9:
            scores[FailurePattern.VERBOSE_ANSWER] += 3
        if length_penalty < 0.7:
            scores[FailurePattern.VERBOSE_ANSWER] += 1
        if rouge_l < 0.3 and semantic > 0.5:
            scores[FailurePattern.VERBOSE_ANSWER] += 2
        if len(actual) > len(expected) * 2.5:
            scores[FailurePattern.VERBOSE_ANSWER] += 1

        # --- HALLUCINATION: 환각 ---
        if semantic < 0.4:
            scores[FailurePattern.HALLUCINATION] += 3
        if confidence < 0.4 and semantic < 0.5:
            scores[FailurePattern.HALLUCINATION] += 2
        if actual.startswith("ERROR:"):
            scores[FailurePattern.HALLUCINATION] += 1

        # --- WRONG_RETRIEVAL: 검색 오류 ---
        if confidence < 0.3:
            scores[FailurePattern.WRONG_RETRIEVAL] += 3
        if sources_count < 3:
            scores[FailurePattern.WRONG_RETRIEVAL] += 1
        if confidence < 0.2 and semantic < 0.5:
            scores[FailurePattern.WRONG_RETRIEVAL] += 1

        # --- WRONG_SCOPE: 범위 오류 (주로 negative 카테고리) ---
        if category == "negative":
            threshold = CATEGORY_THRESHOLDS.get("negative", 0.40)
            if semantic > 0.4 and category_score < threshold:
                scores[FailurePattern.WRONG_SCOPE] += 3
            if confidence > 0.5 and category_score < threshold:
                scores[FailurePattern.WRONG_SCOPE] += 1

        # --- WEAK_INFERENCE: 추론 부족 (주로 inference 카테고리) ---
        if category == "inference":
            if rouge_l > semantic:
                scores[FailurePattern.WEAK_INFERENCE] += 2
            # 답변이 기대와 표면적으로 비슷하지만 의미적으로 부족
            if semantic < 0.6 and rouge_l > 0.2:
                scores[FailurePattern.WEAK_INFERENCE] += 1
        if category == "inference" and composite_score < 0.5:
            scores[FailurePattern.WEAK_INFERENCE] += 1

        # --- MULTI_HOP_FAILURE: 다중 규정 연결 실패 ---
        if category == "multi_hop":
            if confidence < 0.5:
                scores[FailurePattern.MULTI_HOP_FAILURE] += 3
            if sources_count < 5:
                scores[FailurePattern.MULTI_HOP_FAILURE] += 1
            if semantic < 0.6:
                scores[FailurePattern.MULTI_HOP_FAILURE] += 1

        # --- INCOMPLETE_LIST: 목록 불완전 ---
        if _has_list_markers(expected):
            if len(actual) < len(expected) * 0.7:
                scores[FailurePattern.INCOMPLETE_LIST] += 3
            elif len(actual) < len(expected) * 0.85:
                scores[FailurePattern.INCOMPLETE_LIST] += 1

        # --- LANGUAGE_LEAK: 언어 혼재 ---
        if _contains_non_korean_sentences(actual):
            scores[FailurePattern.LANGUAGE_LEAK] += 2

        # --- FORMAT_ERROR: 형식 오류 ---
        if _has_format_prefix(actual):
            scores[FailurePattern.FORMAT_ERROR] += 2

        # --- LOW_CONFIDENCE: 폴백 (명확한 패턴 없을 때) ---
        max_score = max(scores.values())
        if max_score == 0:
            scores[FailurePattern.LOW_CONFIDENCE] += 1

        # 주 패턴 결정 (최고 점수)
        primary_pattern = max(scores, key=lambda p: scores[p])
        primary_score = scores[primary_pattern]

        # 보조 패턴 (점수 > 0, 주 패턴 제외)
        secondary_patterns = [
            p.value for p in FailurePattern
            if scores[p] > 0 and p != primary_pattern
        ]

        # 패턴 확신도 (0~1): 주 패턴 점수 / 전체 점수 합
        total_score = sum(scores.values())
        pattern_confidence = primary_score / total_score if total_score > 0 else 0.0

        # 진단 문장 생성
        diagnosis = self._generate_diagnosis(primary_pattern, result)

        # 수정 제안 생성
        analysis = FailureAnalysis(
            question_id=result.get("id", 0),
            dataset=result.get("dataset", ""),
            category=category,
            difficulty=result.get("difficulty", ""),
            question=result.get("question", ""),
            expected_answer=expected,
            actual_answer=actual,
            confidence=confidence,
            sources_count=sources_count,
            semantic_similarity=semantic,
            rouge_1=result.get("rouge_1", 0.0),
            rouge_l=rouge_l,
            composite_score=composite_score,
            grade=result.get("grade", ""),
            length_penalty=length_penalty,
            category_score=category_score,
            primary_pattern=primary_pattern.value,
            secondary_patterns=secondary_patterns,
            pattern_confidence=round(pattern_confidence, 3),
            diagnosis=diagnosis,
        )

        analysis.fix_suggestions = self._generate_fix_suggestions(analysis)

        return analysis

    # ----- 진단 문장 생성 -----

    def _generate_diagnosis(self, pattern: FailurePattern, result: dict) -> str:
        """주 패턴과 결과 데이터를 기반으로 한국어 진단 문장을 생성한다."""
        cat = result.get("category", "")
        confidence = result.get("confidence", 0.0)
        semantic = result.get("semantic_similarity", 0.0)
        rouge_l = result.get("rouge_l", 0.0)
        length_penalty = result.get("length_penalty", 1.0)

        base = PATTERN_DESCRIPTIONS.get(pattern, "분류 불가")

        if pattern == FailurePattern.VERBOSE_ANSWER:
            return f"{base}. 길이 패널티={length_penalty:.2f}, ROUGE-L={rouge_l:.2f}"
        elif pattern == FailurePattern.HALLUCINATION:
            return f"{base}. 의미 유사도={semantic:.2f}, 신뢰도={confidence:.2f}"
        elif pattern == FailurePattern.WRONG_RETRIEVAL:
            sources = result.get("sources_count", 0)
            return f"{base}. 신뢰도={confidence:.2f}, 검색 소스={sources}개"
        elif pattern == FailurePattern.WRONG_SCOPE:
            return f"[{cat}] {base}. 의미 유사도={semantic:.2f}이나 범위 외 내용 포함"
        elif pattern == FailurePattern.WEAK_INFERENCE:
            return f"[{cat}] {base}. ROUGE-L={rouge_l:.2f} > 의미 유사도={semantic:.2f} (표면 유사, 의미 부족)"
        elif pattern == FailurePattern.MULTI_HOP_FAILURE:
            return f"[{cat}] {base}. 신뢰도={confidence:.2f}, 검색 소스={result.get('sources_count', 0)}개"
        elif pattern == FailurePattern.INCOMPLETE_LIST:
            return f"{base}. 기대 답변 대비 실제 답변 길이 부족"
        elif pattern == FailurePattern.LANGUAGE_LEAK:
            return f"{base}. 한국어 외 문장이 답변에 포함됨"
        elif pattern == FailurePattern.FORMAT_ERROR:
            return f"{base}. 답변 시작부에 불필요한 접두사 존재"
        else:
            return f"{base}. 신뢰도={confidence:.2f}, 종합={result.get('composite_score', 0.0):.2f}"

    # ----- 수정 제안 생성 -----

    def _generate_fix_suggestions(self, analysis: FailureAnalysis) -> list[str]:
        """FailureAnalysis를 기반으로 구체적인 수정 제안을 생성한다."""
        suggestions: list[str] = []

        primary = FailurePattern(analysis.primary_pattern)
        # 주 패턴에 대한 기본 제안
        suggestions.extend(PATTERN_FIX_MAP.get(primary, []))

        # 카테고리 특화 추가 제안
        if analysis.category == "negative" and primary != FailurePattern.WRONG_SCOPE:
            suggestions.append("negative 카테고리: 범위 외 질문 처리 프롬프트 보강")

        if analysis.category == "multi_hop" and primary != FailurePattern.MULTI_HOP_FAILURE:
            suggestions.append("multi_hop 카테고리: 다중 문서 연결 검색 전략 개선")

        # 난이도 기반 추가 제안
        if analysis.difficulty == "hard":
            suggestions.append("고난이도 문항: 모델 크기 업그레이드 고려 (7B → 14B+)")

        # 중복 제거
        seen: set[str] = set()
        deduped: list[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                deduped.append(s)

        return deduped

    # ----- 패턴별 집계 -----

    def _aggregate_patterns(self, failures: list[FailureAnalysis]) -> list[PatternSummary]:
        """실패 분석 결과를 패턴별로 집계한다."""
        pattern_groups: dict[str, list[FailureAnalysis]] = {}
        for fa in failures:
            pattern_groups.setdefault(fa.primary_pattern, []).append(fa)

        summaries: list[PatternSummary] = []

        for pattern_val, group in sorted(pattern_groups.items(), key=lambda x: -len(x[1])):
            count = len(group)
            categories = sorted(set(fa.category for fa in group))
            difficulties = sorted(set(fa.difficulty for fa in group))

            avg_semantic = sum(fa.semantic_similarity for fa in group) / count
            avg_rouge = sum(fa.rouge_l for fa in group) / count
            avg_composite = sum(fa.composite_score for fa in group) / count
            avg_conf = sum(fa.confidence for fa in group) / count

            pattern_enum = FailurePattern(pattern_val)
            fix_suggestions = PATTERN_FIX_MAP.get(pattern_enum, [])

            summaries.append(PatternSummary(
                pattern=pattern_val,
                pattern_label=PATTERN_DESCRIPTIONS.get(pattern_enum, pattern_val),
                count=count,
                affected_categories=categories,
                affected_difficulties=difficulties,
                avg_semantic_similarity=round(avg_semantic, 4),
                avg_rouge_l=round(avg_rouge, 4),
                avg_composite_score=round(avg_composite, 4),
                avg_confidence=round(avg_conf, 4),
                fix_suggestions=fix_suggestions,
            ))

        return summaries

    # ----- 상위 권장사항 생성 -----

    def _generate_recommendations(self, summaries: list[PatternSummary]) -> list[dict]:
        """패턴 요약을 기반으로 우선순위별 권장사항을 생성한다."""
        if not summaries:
            return []

        recommendations: list[dict] = []
        total_failures = sum(s.count for s in summaries)

        for priority, summary in enumerate(summaries, 1):
            pct = summary.count / total_failures * 100 if total_failures > 0 else 0

            rec = {
                "priority": priority,
                "pattern": summary.pattern,
                "title": f"{summary.pattern_label} ({summary.count}건, {pct:.0f}%)",
                "description": (
                    f"카테고리: {', '.join(summary.affected_categories)} | "
                    f"난이도: {', '.join(summary.affected_difficulties)} | "
                    f"평균 종합점수: {summary.avg_composite_score:.3f}"
                ),
                "actions": summary.fix_suggestions[:3],  # 상위 3개 액션
                "impact": "높음" if pct >= 30 else ("중간" if pct >= 15 else "낮음"),
            }
            recommendations.append(rec)

        return recommendations

    # ----- 파라미터 설정 제안 -----

    def _suggest_parameters(self, summaries: list[PatternSummary]) -> dict:
        """지배적 패턴을 기반으로 RAG 파라미터 설정 변경을 제안한다."""
        if not summaries:
            return {}

        suggestions: dict[str, object] = {}
        pattern_counts: dict[str, int] = {s.pattern: s.count for s in summaries}

        # VERBOSE_ANSWER가 지배적 → 길이 제한 강화
        if pattern_counts.get(FailurePattern.VERBOSE_ANSWER.value, 0) >= 2:
            suggestions["max_tokens"] = "현재 값의 70%로 축소 권장"
            suggestions["length_penalty_threshold"] = "2.5 → 2.0 으로 하향"
            suggestions["system_prompt"] = "간결성 지시 강화 필요"

        # HALLUCINATION이 지배적 → Self-RAG 활성화
        if pattern_counts.get(FailurePattern.HALLUCINATION.value, 0) >= 2:
            suggestions["SELF_RAG_ENABLED"] = True
            suggestions["confidence_threshold"] = "현재 값 + 0.1 상향"

        # WRONG_RETRIEVAL이 지배적 → 검색 파라미터 조정
        if pattern_counts.get(FailurePattern.WRONG_RETRIEVAL.value, 0) >= 2:
            suggestions["MULTI_QUERY_ENABLED"] = True
            suggestions["retriever_top_k"] = "현재 값 + 5 확대"
            suggestions["reranker_weight"] = "상향 조정 권장"

        # MULTI_HOP_FAILURE가 지배적 → 다단계 검색 활성화
        if pattern_counts.get(FailurePattern.MULTI_HOP_FAILURE.value, 0) >= 2:
            suggestions["AGENTIC_RAG_ENABLED"] = True
            suggestions["MULTI_QUERY_ENABLED"] = True
            suggestions["retriever_top_k"] = "15~20으로 대폭 확대"

        # WEAK_INFERENCE가 지배적 → 추론 강화
        if pattern_counts.get(FailurePattern.WEAK_INFERENCE.value, 0) >= 2:
            suggestions["few_shot_inference_examples"] = "추론 과정 포함 예시 추가"
            suggestions["system_prompt"] = suggestions.get("system_prompt", "") + " + 추론 지시 강화"

        # WRONG_SCOPE가 지배적 → negative 처리 강화
        if pattern_counts.get(FailurePattern.WRONG_SCOPE.value, 0) >= 2:
            suggestions["few_shot_negative_examples"] = "범위 외 답변 예시 추가"
            suggestions["guardrails_scope_filter"] = True

        # LANGUAGE_LEAK → 언어 가드레일
        if pattern_counts.get(FailurePattern.LANGUAGE_LEAK.value, 0) >= 1:
            suggestions["output_language_filter"] = "한국어 전용 필터 활성화"

        return suggestions

    # ----- 리포트 콘솔 출력 -----

    def print_report(self, report: AnalysisReport) -> None:
        """분석 리포트를 한국어로 콘솔에 출력한다."""
        print(f"\n{'='*70}")
        print(f"  벤치마크 실패 패턴 분석 리포트")
        print(f"{'='*70}")

        print(f"\n[전체 요약]")
        print(f"  총 문항 수: {report.total_questions}")
        print(f"  실패 문항 수: {report.total_failures}")
        print(f"  실패율: {report.failure_rate*100:.1f}%")

        if report.total_failures == 0:
            print(f"\n  모든 문항이 통과했습니다!")
            print(f"{'='*70}\n")
            return

        # 패턴별 요약
        print(f"\n[패턴별 분포]")
        total_f = report.total_failures
        for ps in report.pattern_summaries:
            pct = ps.count / total_f * 100 if total_f > 0 else 0
            print(f"\n  {ps.pattern_label}")
            print(f"    건수: {ps.count}건 ({pct:.0f}%)")
            print(f"    카테고리: {', '.join(ps.affected_categories)}")
            print(f"    난이도: {', '.join(ps.affected_difficulties)}")
            print(f"    평균 의미유사도: {ps.avg_semantic_similarity:.3f}")
            print(f"    평균 ROUGE-L: {ps.avg_rouge_l:.3f}")
            print(f"    평균 종합점수: {ps.avg_composite_score:.3f}")
            print(f"    평균 신뢰도: {ps.avg_confidence:.3f}")

        # 개별 실패 문항 상세
        print(f"\n[실패 문항 상세 ({report.total_failures}건)]")
        for fa in report.failures:
            print(f"\n  --- Q{fa.question_id} [{fa.dataset}] ({fa.category}/{fa.difficulty}) ---")
            print(f"  질문: {fa.question[:70]}{'...' if len(fa.question) > 70 else ''}")
            print(f"  패턴: {fa.primary_pattern} (확신도: {fa.pattern_confidence:.1%})")
            if fa.secondary_patterns:
                print(f"  보조 패턴: {', '.join(fa.secondary_patterns)}")
            print(f"  진단: {fa.diagnosis}")
            print(f"  점수: composite={fa.composite_score:.3f} semantic={fa.semantic_similarity:.3f} "
                  f"rouge_l={fa.rouge_l:.3f} conf={fa.confidence:.3f}")

            if self.verbose:
                print(f"  기대 답변: {fa.expected_answer[:100]}{'...' if len(fa.expected_answer) > 100 else ''}")
                print(f"  실제 답변: {fa.actual_answer[:100]}{'...' if len(fa.actual_answer) > 100 else ''}")

            if fa.fix_suggestions:
                print(f"  수정 제안:")
                for j, sug in enumerate(fa.fix_suggestions[:3], 1):
                    print(f"    {j}. {sug}")

        # 권장사항
        print(f"\n[우선순위별 권장사항]")
        for rec in report.recommendations:
            impact_marker = {"높음": "!!!", "중간": "!!", "낮음": "!"}.get(rec["impact"], "")
            print(f"\n  [{rec['priority']}순위] {impact_marker} {rec['title']}")
            print(f"    {rec['description']}")
            if rec.get("actions"):
                for action in rec["actions"]:
                    print(f"    -> {action}")
            print(f"    영향도: {rec['impact']}")

        # 파라미터 제안
        if report.parameter_suggestions:
            print(f"\n[파라미터 설정 제안]")
            for key, value in report.parameter_suggestions.items():
                print(f"  {key}: {value}")

        print(f"\n{'='*70}\n")

    # ----- 리포트 JSON 저장 -----

    def save_report(self, report: AnalysisReport, output_path: str | Path) -> None:
        """분석 리포트를 JSON 파일로 저장한다."""
        output_path = Path(output_path)

        report_dict = {
            "total_questions": report.total_questions,
            "total_failures": report.total_failures,
            "failure_rate": report.failure_rate,
            "pattern_summaries": [asdict(ps) for ps in report.pattern_summaries],
            "failures": [asdict(fa) for fa in report.failures],
            "recommendations": report.recommendations,
            "parameter_suggestions": report.parameter_suggestions,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        print(f"분석 리포트 저장됨: {output_path}")


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="벤치마크 실패 패턴 분석기 - benchmark_all.py 결과를 분석하여 패턴별 수정 제안을 생성합니다."
    )
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="벤치마크 결과 JSON 파일 경로 (benchmark_all.py 출력)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="분석 결과 JSON 저장 경로 (미지정시 콘솔만 출력)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력 (기대/실제 답변 포함)",
    )
    args = parser.parse_args()

    analyzer = FailurePatternAnalyzer(verbose=args.verbose)

    try:
        report = analyzer.analyze(args.results)
        analyzer.print_report(report)

        if args.output:
            analyzer.save_report(report, args.output)
    except FileNotFoundError as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}", file=sys.stderr)
        sys.exit(1)
