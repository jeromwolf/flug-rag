"""RAG 통합 튜닝 파이프라인.

문서 인제스트 -> 골든 데이터셋 생성 -> 검증 -> 베이스라인 벤치마크 ->
실패 분석 -> 자동 튜닝 -> 튜닝 벤치마크 -> 최종 리포트

기존 인프라를 결합하여 end-to-end 자동화를 제공한다.

Usage:
    # 전체 파이프라인
    python tests/tuning_pipeline.py --dataset-name "출장보고서" --data-dir data/uploads/국외출장_결과보고서

    # 특정 단계만
    python tests/tuning_pipeline.py --dataset-name "출장보고서" --step baseline_benchmark

    # 이전 상태에서 재개
    python tests/tuning_pipeline.py --dataset-name "출장보고서" --resume

    # 골든 데이터셋 검증만
    python tests/tuning_pipeline.py --validate-only tests/golden_dataset_travel.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# sys.path 설정
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / "data"

INGEST_EXTENSIONS = {".pdf", ".hwp", ".docx", ".xlsx", ".pptx", ".txt"}


# ===========================================================================
# PipelineStep
# ===========================================================================

class PipelineStep(Enum):
    """파이프라인 단계 열거형."""

    INGEST = "ingest"
    GENERATE_GOLDEN = "generate_golden"
    VALIDATE_DATASET = "validate_dataset"
    BASELINE_BENCHMARK = "baseline_benchmark"
    ANALYZE_FAILURES = "analyze_failures"
    AUTO_TUNE = "auto_tune"
    TUNED_BENCHMARK = "tuned_benchmark"
    FINAL_REPORT = "final_report"


# ===========================================================================
# PipelineConfig
# ===========================================================================

@dataclass
class PipelineConfig:
    """파이프라인 실행 설정."""

    data_dir: str | None = None
    dataset_name: str = ""
    dp_mode: str = "auto"
    extra_metadata: dict | None = None
    golden_output_path: str | None = None
    category_distribution: dict[str, int] = field(default_factory=lambda: {
        "factual": 10, "inference": 8, "multi_hop": 5, "negative": 3,
    })
    source_filter: dict | None = None
    model: str | None = None
    target_success_rate: float = 0.95
    max_tuning_iterations: int = 3
    optimizer_grid: str = "combined"
    optimizer_sample_size: int = 15
    state_file: str | None = None


# ===========================================================================
# PipelineState
# ===========================================================================

@dataclass
class PipelineState:
    """파이프라인 실행 상태. 재개를 위해 디스크에 저장된다."""

    dataset_name: str = ""
    started_at: str = ""
    updated_at: str = ""
    completed_steps: list[str] = field(default_factory=list)
    current_step: str | None = None

    # 단계별 결과
    ingest_stats: dict = field(default_factory=dict)
    golden_path: str = ""
    validation_result: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)
    failure_analysis: dict = field(default_factory=dict)
    tuned_params: dict = field(default_factory=dict)
    tuned_metrics: dict = field(default_factory=dict)
    tuning_history: list[dict] = field(default_factory=list)
    final_report: dict = field(default_factory=dict)

    def is_step_completed(self, step: PipelineStep) -> bool:
        return step.value in self.completed_steps

    def mark_completed(self, step: PipelineStep) -> None:
        if step.value not in self.completed_steps:
            self.completed_steps.append(step.value)
        self.current_step = None
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_in_progress(self, step: PipelineStep) -> None:
        self.current_step = step.value
        self.updated_at = datetime.now(timezone.utc).isoformat()


# ===========================================================================
# DatasetValidator
# ===========================================================================

class DatasetValidator:
    """골든 데이터셋 스키마 및 품질 검증기."""

    REQUIRED_FIELDS = {"id", "category", "question", "answer"}
    VALID_CATEGORIES = {"factual", "inference", "multi_hop", "negative"}
    VALID_DIFFICULTIES = {"easy", "medium", "hard"}

    @staticmethod
    def validate(dataset_path: str | Path) -> dict:
        """골든 데이터셋 JSON을 검증하여 결과를 반환한다.

        Returns:
            {
                "valid": bool,
                "errors": [...],
                "warnings": [...],
                "stats": {...},
            }
        """
        dataset_path = Path(dataset_path)
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        # 파일 존재 여부
        if not dataset_path.exists():
            return {
                "valid": False,
                "errors": [f"파일이 존재하지 않습니다: {dataset_path}"],
                "warnings": [],
                "stats": {},
            }

        # JSON 파싱
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"JSON 파싱 오류: {e}"],
                "warnings": [],
                "stats": {},
            }

        # 최상위 구조
        questions = data.get("questions", [])
        if not questions:
            errors.append("질문 목록(questions)이 비어 있습니다.")
            return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}

        stats["total_questions"] = len(questions)

        # 카테고리 분포
        category_counts: dict[str, int] = {}
        difficulty_counts: dict[str, int] = {}
        answer_lengths: list[int] = []
        question_texts: list[str] = []
        ids_seen: set[int] = set()

        for idx, q in enumerate(questions):
            # 필수 필드 확인
            missing = DatasetValidator.REQUIRED_FIELDS - set(q.keys())
            if missing:
                errors.append(f"Q#{idx+1}: 필수 필드 누락 - {missing}")

            # id 중복
            qid = q.get("id")
            if qid is not None:
                if qid in ids_seen:
                    errors.append(f"Q#{idx+1}: 중복 ID - {qid}")
                ids_seen.add(qid)

            # 카테고리 유효성
            cat = q.get("category", "")
            if cat not in DatasetValidator.VALID_CATEGORIES:
                warnings.append(f"Q#{idx+1}: 알 수 없는 카테고리 '{cat}'")
            category_counts[cat] = category_counts.get(cat, 0) + 1

            # 난이도 유효성
            diff = q.get("difficulty", "")
            if diff and diff not in DatasetValidator.VALID_DIFFICULTIES:
                warnings.append(f"Q#{idx+1}: 알 수 없는 난이도 '{diff}'")
            if diff:
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            # 답변 길이
            answer = q.get("answer", "")
            if isinstance(answer, str):
                answer_lengths.append(len(answer))
                if len(answer) < 5:
                    warnings.append(f"Q#{idx+1}: 답변이 매우 짧습니다 ({len(answer)}자)")

            # 질문 텍스트 (중복 탐지용)
            question_text = q.get("question", "")
            if isinstance(question_text, str):
                question_texts.append(question_text)

        stats["category_distribution"] = category_counts
        stats["difficulty_distribution"] = difficulty_counts

        # 카테고리 균형 경고 (전체 대비 20% 미만인 카테고리)
        total = len(questions)
        for cat, cnt in category_counts.items():
            pct = cnt / total if total > 0 else 0
            if pct < 0.05 and total >= 10:
                warnings.append(
                    f"카테고리 '{cat}'이 전체의 {pct*100:.0f}%로 매우 적습니다 ({cnt}/{total})"
                )

        # 답변 길이 통계
        if answer_lengths:
            stats["answer_length_stats"] = {
                "min": min(answer_lengths),
                "max": max(answer_lengths),
                "avg": round(sum(answer_lengths) / len(answer_lengths), 1),
                "median": sorted(answer_lengths)[len(answer_lengths) // 2],
            }

        # 중복 질문 탐지 (Jaccard > 0.8)
        duplicates_found = 0
        for i in range(len(question_texts)):
            tokens_i = set(question_texts[i].replace(" ", ""))
            if not tokens_i:
                continue
            for j in range(i + 1, len(question_texts)):
                tokens_j = set(question_texts[j].replace(" ", ""))
                if not tokens_j:
                    continue
                intersection = tokens_i & tokens_j
                union = tokens_i | tokens_j
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard > 0.8:
                    duplicates_found += 1
                    warnings.append(
                        f"유사 질문 쌍 (Jaccard={jaccard:.2f}): Q#{i+1} <-> Q#{j+1}"
                    )

        stats["duplicate_pairs"] = duplicates_found

        valid = len(errors) == 0
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
        }


# ===========================================================================
# TuningPipeline
# ===========================================================================

class TuningPipeline:
    """RAG 통합 튜닝 파이프라인.

    문서 인제스트부터 최종 리포트까지 전체 RAG 튜닝 사이클을 자동화한다.
    각 단계별 상태를 저장하여 중단 후 재개가 가능하다.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = PipelineState(dataset_name=config.dataset_name)

        # 상태 파일 경로 결정
        if config.state_file:
            self._state_path = Path(config.state_file)
        else:
            safe_name = config.dataset_name.replace(" ", "_").replace("/", "_")
            self._state_path = DATA_DIR / f"tuning_state_{safe_name}.json"

        # 골든 데이터셋 경로 결정
        if config.golden_output_path:
            self._golden_path = Path(config.golden_output_path)
        else:
            safe_name = config.dataset_name.replace(" ", "_").replace("/", "_")
            self._golden_path = TESTS_DIR / f"golden_dataset_{safe_name}.json"

        # 벤치마크 결과 경로
        safe_name = config.dataset_name.replace(" ", "_").replace("/", "_")
        self._baseline_results_path = DATA_DIR / f"benchmark_results_all_{safe_name}.json"
        self._tuned_results_path = DATA_DIR / f"benchmark_results_tuned_{safe_name}.json"
        self._failure_analysis_path = DATA_DIR / f"failure_analysis_{safe_name}.json"

    # ------------------------------------------------------------------
    # 메인 실행
    # ------------------------------------------------------------------

    async def run(
        self,
        start_from: PipelineStep | None = None,
        stop_after: PipelineStep | None = None,
    ) -> PipelineState:
        """전체 또는 부분 파이프라인 실행.

        Args:
            start_from: 시작 단계 (None이면 처음부터)
            stop_after: 종료 단계 (None이면 끝까지)

        Returns:
            PipelineState: 최종 상태
        """
        all_steps = list(PipelineStep)

        # 시작/종료 인덱스 결정
        start_idx = 0
        if start_from:
            for i, step in enumerate(all_steps):
                if step == start_from:
                    start_idx = i
                    break

        stop_idx = len(all_steps) - 1
        if stop_after:
            for i, step in enumerate(all_steps):
                if step == stop_after:
                    stop_idx = i
                    break

        # 초기 상태
        if not self.state.started_at:
            self.state.started_at = datetime.now(timezone.utc).isoformat()

        print(f"\n{'='*70}")
        print(f"  RAG 통합 튜닝 파이프라인")
        print(f"{'='*70}")
        print(f"  데이터셋: {self.config.dataset_name}")
        if self.config.data_dir:
            print(f"  문서 디렉토리: {self.config.data_dir}")
        print(f"  목표 성공률: {self.config.target_success_rate*100:.0f}%")
        print(f"  모델: {self.config.model or '기본값'}")
        print(f"  실행 범위: {all_steps[start_idx].value} -> {all_steps[stop_idx].value}")
        print(f"{'='*70}\n")

        # 각 단계 순차 실행
        for i in range(start_idx, stop_idx + 1):
            step = all_steps[i]

            # 이미 완료된 단계 건너뛰기
            if self.state.is_step_completed(step):
                print(f"[건너뜀] {step.value} (이미 완료)")
                continue

            print(f"\n{'─'*60}")
            print(f"  단계: {step.value}")
            print(f"{'─'*60}")

            self.state.mark_in_progress(step)
            self._save_state()

            try:
                await self.run_step(step)
                self.state.mark_completed(step)
                self._save_state()
                print(f"  [완료] {step.value}")
            except Exception as e:
                logger.error("단계 실패: %s - %s", step.value, e)
                print(f"\n  [실패] {step.value}: {e}")
                self._save_state()
                raise

        print(f"\n{'='*70}")
        print(f"  파이프라인 완료")
        print(f"  상태 파일: {self._state_path}")
        print(f"{'='*70}\n")

        return self.state

    async def run_step(self, step: PipelineStep) -> None:
        """단일 단계 실행."""
        dispatch = {
            PipelineStep.INGEST: self._step_ingest,
            PipelineStep.GENERATE_GOLDEN: self._step_generate_golden,
            PipelineStep.VALIDATE_DATASET: self._step_validate_dataset,
            PipelineStep.BASELINE_BENCHMARK: self._step_baseline_benchmark,
            PipelineStep.ANALYZE_FAILURES: self._step_analyze_failures,
            PipelineStep.AUTO_TUNE: self._step_auto_tune,
            PipelineStep.TUNED_BENCHMARK: self._step_tuned_benchmark,
            PipelineStep.FINAL_REPORT: self._step_final_report,
        }
        handler = dispatch.get(step)
        if handler is None:
            raise ValueError(f"알 수 없는 단계: {step}")
        await handler()

    # ------------------------------------------------------------------
    # Step 1: 문서 인제스트
    # ------------------------------------------------------------------

    async def _step_ingest(self) -> None:
        """data_dir 내 문서들을 IngestPipeline으로 인제스트."""
        if not self.config.data_dir:
            print("  [건너뜀] data_dir이 지정되지 않음 (이미 인제스트된 데이터 사용)")
            self.state.ingest_stats = {"skipped": True, "reason": "no_data_dir"}
            return

        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"문서 디렉토리를 찾을 수 없습니다: {data_dir}")

        # 파일 수집
        files: list[Path] = []
        for ext in INGEST_EXTENSIONS:
            files.extend(data_dir.rglob(f"*{ext}"))
        files.sort()

        if not files:
            raise ValueError(f"인제스트할 문서가 없습니다: {data_dir} (지원 확장자: {INGEST_EXTENSIONS})")

        print(f"  발견 문서: {len(files)}개")

        from pipeline.ingest import IngestPipeline

        pipeline = IngestPipeline()
        success_count = 0
        fail_count = 0
        total_chunks = 0

        for i, file_path in enumerate(files, 1):
            print(f"  [{i:03d}/{len(files)}] {file_path.name}...", end=" ", flush=True)
            try:
                result = await pipeline.ingest(
                    file_path,
                    extra_metadata=self.config.extra_metadata,
                    dp_mode=self.config.dp_mode,
                )
                if result.status == "completed":
                    success_count += 1
                    total_chunks += result.chunk_count
                    print(f"OK ({result.chunk_count} 청크)")
                else:
                    fail_count += 1
                    print(f"FAIL: {result.error}")
            except Exception as e:
                fail_count += 1
                print(f"ERROR: {e}")

        self.state.ingest_stats = {
            "total_files": len(files),
            "success": success_count,
            "failed": fail_count,
            "total_chunks": total_chunks,
        }

        print(f"\n  인제스트 완료: 성공={success_count}, 실패={fail_count}, 총 청크={total_chunks}")

    # ------------------------------------------------------------------
    # Step 2: 골든 데이터셋 생성
    # ------------------------------------------------------------------

    async def _step_generate_golden(self) -> None:
        """GoldenDatasetGenerator로 골든 데이터셋 생성."""
        # 이미 존재하는 골든 데이터셋 확인
        if self._golden_path.exists():
            print(f"  [참고] 기존 골든 데이터셋 발견: {self._golden_path}")
            print(f"  기존 파일을 사용합니다. 재생성하려면 파일을 삭제하세요.")
            self.state.golden_path = str(self._golden_path)
            return

        from tests.golden_generator import GoldenDatasetGenerator, GenerationConfig

        gen_config = GenerationConfig(
            dataset_name=self.config.dataset_name,
            category_distribution=self.config.category_distribution,
            source_filter=self.config.source_filter,
        )

        llm = None
        if self.config.model:
            from core.llm import create_llm
            llm = create_llm(model=self.config.model)

        generator = GoldenDatasetGenerator(config=gen_config, llm=llm)
        dataset = await generator.generate()

        # 저장
        self._golden_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._golden_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        total_qs = dataset.get("dataset_info", {}).get("total_questions", 0)
        print(f"  골든 데이터셋 저장: {self._golden_path} ({total_qs}개 Q&A)")

        self.state.golden_path = str(self._golden_path)

    # ------------------------------------------------------------------
    # Step 3: 데이터셋 검증
    # ------------------------------------------------------------------

    async def _step_validate_dataset(self) -> None:
        """DatasetValidator로 골든 데이터셋 검증."""
        golden_path = self.state.golden_path or str(self._golden_path)

        result = DatasetValidator.validate(golden_path)
        self.state.validation_result = result

        # 결과 출력
        print(f"  검증 결과: {'유효' if result['valid'] else '오류 있음'}")
        if result.get("stats"):
            stats = result["stats"]
            print(f"  총 문항 수: {stats.get('total_questions', 0)}")
            if stats.get("category_distribution"):
                cat_str = ", ".join(
                    f"{k}({v})" for k, v in stats["category_distribution"].items()
                )
                print(f"  카테고리: {cat_str}")
            if stats.get("answer_length_stats"):
                als = stats["answer_length_stats"]
                print(f"  답변 길이: 최소={als['min']}, 최대={als['max']}, "
                      f"평균={als['avg']}, 중앙값={als['median']}")
            if stats.get("duplicate_pairs", 0) > 0:
                print(f"  유사 질문 쌍: {stats['duplicate_pairs']}개")

        if result.get("errors"):
            print(f"\n  [오류]")
            for err in result["errors"]:
                print(f"    - {err}")

        if result.get("warnings"):
            print(f"\n  [경고]")
            for warn in result["warnings"][:10]:
                print(f"    - {warn}")
            if len(result["warnings"]) > 10:
                print(f"    ... 외 {len(result['warnings'])-10}건")

        if not result["valid"]:
            raise ValueError("골든 데이터셋 검증 실패. 오류를 수정한 후 재시도하세요.")

    # ------------------------------------------------------------------
    # Step 4: 베이스라인 벤치마크
    # ------------------------------------------------------------------

    async def _step_baseline_benchmark(self) -> None:
        """현재 설정으로 벤치마크 실행."""
        golden_path = self.state.golden_path or str(self._golden_path)

        from tests.benchmark_all import run_benchmark, DATASET_CONFIGS, load_dataset

        # 동적으로 DATASET_CONFIGS에 등록
        ds_key = self.config.dataset_name.replace(" ", "_").lower()
        DATASET_CONFIGS[ds_key] = {
            "name": self.config.dataset_name,
            "path": Path(golden_path),
        }

        print(f"  데이터셋 '{ds_key}' 등록 완료")
        print(f"  벤치마크 실행 중...")

        results = await run_benchmark(
            model_name=self.config.model,
            dataset_filter=ds_key,
        )

        # 메트릭 수집
        if results:
            total = len(results)
            success_count = sum(1 for r in results if r.success)
            success_rate = success_count / total if total > 0 else 0.0
            avg_composite = sum(r.composite_score for r in results) / total if total > 0 else 0.0
            avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0.0
            avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0.0

            self.state.baseline_metrics = {
                "total_questions": total,
                "success_count": success_count,
                "success_rate": round(success_rate, 4),
                "avg_composite_score": round(avg_composite, 4),
                "avg_confidence": round(avg_confidence, 4),
                "avg_latency_ms": round(avg_latency, 1),
            }

            print(f"\n  [베이스라인 결과]")
            print(f"    성공률: {success_count}/{total} ({success_rate*100:.1f}%)")
            print(f"    평균 종합점수: {avg_composite:.4f}")
            print(f"    평균 신뢰도: {avg_confidence:.4f}")

            # 결과 파일 경로 확인 (benchmark_all.py가 저장한 파일)
            expected_result_path = DATA_DIR / f"benchmark_results_all_{ds_key}.json"
            if expected_result_path.exists():
                self._baseline_results_path = expected_result_path
            else:
                # 모델 이름이 포함된 경로 탐색
                for p in DATA_DIR.glob(f"benchmark_results_all_{ds_key}*.json"):
                    self._baseline_results_path = p
                    break
        else:
            self.state.baseline_metrics = {
                "total_questions": 0,
                "success_count": 0,
                "success_rate": 0.0,
            }
            print("  벤치마크 결과가 비어 있습니다.")

    # ------------------------------------------------------------------
    # Step 5: 실패 분석
    # ------------------------------------------------------------------

    async def _step_analyze_failures(self) -> None:
        """FailurePatternAnalyzer로 실패 패턴 분석."""
        # 결과 파일 찾기
        results_path = self._baseline_results_path
        if not results_path.exists():
            # 대안 경로 탐색
            ds_key = self.config.dataset_name.replace(" ", "_").lower()
            for p in DATA_DIR.glob(f"benchmark_results_all*{ds_key}*.json"):
                results_path = p
                break

        if not results_path.exists():
            print("  [건너뜀] 벤치마크 결과 파일을 찾을 수 없습니다.")
            self.state.failure_analysis = {"skipped": True, "reason": "no_results_file"}
            return

        from tests.failure_analyzer import FailurePatternAnalyzer

        analyzer = FailurePatternAnalyzer(verbose=True)
        report = analyzer.analyze(str(results_path))
        analyzer.print_report(report)

        # 리포트 저장
        analyzer.save_report(report, str(self._failure_analysis_path))

        # 상태에 요약 저장
        self.state.failure_analysis = {
            "total_questions": report.total_questions,
            "total_failures": report.total_failures,
            "failure_rate": report.failure_rate,
            "top_patterns": [
                {
                    "pattern": ps.pattern,
                    "label": ps.pattern_label,
                    "count": ps.count,
                }
                for ps in report.pattern_summaries[:5]
            ],
            "parameter_suggestions": report.parameter_suggestions,
            "recommendations_count": len(report.recommendations),
            "results_path": str(results_path),
        }

        if report.total_failures == 0:
            print("  실패 문항 없음! 추가 튜닝 불필요.")

    # ------------------------------------------------------------------
    # Step 6: 자동 튜닝
    # ------------------------------------------------------------------

    async def _step_auto_tune(self) -> None:
        """실패 분석 결과의 파라미터 제안을 적용하고 옵티마이저 실행."""
        # 실패 분석이 없거나 실패 없으면 건너뛰기
        fa = self.state.failure_analysis
        if fa.get("skipped") or fa.get("total_failures", 0) == 0:
            print("  [건너뜀] 실패 분석 결과가 없거나 실패 문항이 없습니다.")
            self.state.tuned_params = {}
            return

        # 파라미터 제안 적용
        param_suggestions = fa.get("parameter_suggestions", {})
        print(f"  실패 분석 파라미터 제안: {len(param_suggestions)}개")
        for key, val in param_suggestions.items():
            print(f"    {key}: {val}")

        # 옵티마이저 실행
        golden_path = self.state.golden_path or str(self._golden_path)

        from tests.benchmark_optimizer import (
            ParamConfig, PARAM_GRIDS, test_config, select_sample_indices,
        )
        from config.settings import settings
        from core.llm import create_llm
        from rag.evaluator import AnswerEvaluator

        # 데이터셋 로드
        with open(golden_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        questions = dataset.get("questions", [])

        if not questions:
            print("  [건너뜀] 데이터셋에 질문이 없습니다.")
            return

        # 그리드 선택
        grid_name = self.config.optimizer_grid
        if grid_name in PARAM_GRIDS:
            configs = PARAM_GRIDS[grid_name]
        else:
            configs = PARAM_GRIDS.get("combined", [])
            print(f"  그리드 '{grid_name}'를 찾을 수 없어 'combined' 사용")

        if not configs:
            print("  [건너뜀] 파라미터 그리드가 비어 있습니다.")
            return

        print(f"\n  옵티마이저 실행: 그리드={grid_name}, 조합={len(configs)}개")

        # 공유 컴포넌트 초기화
        settings.default_llm_provider = "ollama"
        llm = create_llm(model=self.config.model) if self.config.model else create_llm()
        evaluator = AnswerEvaluator()

        # 샘플 인덱스 선택
        sample_size = self.config.optimizer_sample_size
        sample_indices = None
        if 0 < sample_size < len(questions):
            sample_indices = select_sample_indices(questions, sample_size)
            print(f"  샘플 크기: {sample_size}개 / 전체 {len(questions)}개")

        # 각 구성 테스트
        results = []
        for i, config in enumerate(configs, 1):
            print(f"  [{i}/{len(configs)}] 테스트: {config.name}...", end=" ", flush=True)
            try:
                result = await test_config(config, questions, evaluator, llm, sample_indices)
                results.append(result)
                print(
                    f"score={result.avg_composite:.3f} "
                    f"success={result.success_rate:.1%}"
                )
            except Exception as e:
                print(f"ERROR: {e}")

        if not results:
            print("  옵티마이저 결과 없음.")
            return

        # 최적 구성 선택
        ranked = sorted(results, key=lambda r: (r.avg_composite, r.success_rate), reverse=True)
        best = ranked[0]

        best_params = {
            "name": best.config.name,
            "vector_weight": best.config.vector_weight,
            "bm25_weight": best.config.bm25_weight,
            "retrieval_top_k": best.config.retrieval_top_k,
            "rerank_top_n": best.config.rerank_top_n,
            "temperature": best.config.temperature,
            "score_threshold": best.config.score_threshold,
            "context_max_chunks": best.config.context_max_chunks,
            "max_tokens": best.config.max_tokens,
            "use_rerank": best.config.use_rerank,
            "query_expansion": best.config.query_expansion,
            "bm25_k1": best.config.bm25_k1,
            "bm25_b": best.config.bm25_b,
            "multi_query": best.config.multi_query,
            "self_rag": best.config.self_rag,
            "agentic_rag": best.config.agentic_rag,
            "optimizer_score": best.avg_composite,
            "optimizer_success_rate": best.success_rate,
        }

        self.state.tuned_params = best_params

        print(f"\n  [최적 구성] {best.config.name}")
        print(f"    종합 점수: {best.avg_composite:.4f}")
        print(f"    성공률: {best.success_rate:.1%}")

    # ------------------------------------------------------------------
    # Step 7: 튜닝 벤치마크 (반복 가능)
    # ------------------------------------------------------------------

    async def _step_tuned_benchmark(self) -> None:
        """튜닝된 파라미터로 벤치마크 재실행. 목표 미달 시 반복."""
        if not self.state.tuned_params:
            print("  [건너뜀] 튜닝 파라미터가 없습니다.")
            # 베이스라인을 그대로 사용
            self.state.tuned_metrics = self.state.baseline_metrics.copy()
            return

        from config.settings import settings
        from tests.benchmark_all import run_benchmark, DATASET_CONFIGS

        golden_path = self.state.golden_path or str(self._golden_path)
        ds_key = self.config.dataset_name.replace(" ", "_").lower()

        # DATASET_CONFIGS에 등록 (이전 단계에서 이미 등록되었을 수 있음)
        DATASET_CONFIGS[ds_key] = {
            "name": self.config.dataset_name,
            "path": Path(golden_path),
        }

        iteration = 0
        best_rate = 0.0

        while iteration < self.config.max_tuning_iterations:
            iteration += 1
            print(f"\n  === 튜닝 반복 {iteration}/{self.config.max_tuning_iterations} ===")

            # 파라미터 적용
            tuned = self.state.tuned_params
            original_values = {}
            settings_map = {
                "vector_weight": "vector_weight",
                "bm25_weight": "bm25_weight",
                "retrieval_top_k": "retrieval_top_k",
                "rerank_top_n": "rerank_top_n",
                "temperature": "llm_temperature",
                "score_threshold": "retrieval_score_threshold",
                "context_max_chunks": "context_max_chunks",
                "max_tokens": "llm_max_tokens",
                "use_rerank": "use_rerank",
                "query_expansion": "query_expansion_enabled",
                "bm25_k1": "bm25_k1",
                "bm25_b": "bm25_b",
                "multi_query": "multi_query_enabled",
                "self_rag": "self_rag_enabled",
                "agentic_rag": "agentic_rag_enabled",
            }

            for param_key, settings_key in settings_map.items():
                if param_key in tuned and hasattr(settings, settings_key):
                    original_values[settings_key] = getattr(settings, settings_key)
                    setattr(settings, settings_key, tuned[param_key])

            try:
                results = await run_benchmark(
                    model_name=self.config.model,
                    dataset_filter=ds_key,
                )

                if results:
                    total = len(results)
                    success_count = sum(1 for r in results if r.success)
                    success_rate = success_count / total if total > 0 else 0.0
                    avg_composite = sum(r.composite_score for r in results) / total if total > 0 else 0.0
                    avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0.0

                    metrics = {
                        "iteration": iteration,
                        "total_questions": total,
                        "success_count": success_count,
                        "success_rate": round(success_rate, 4),
                        "avg_composite_score": round(avg_composite, 4),
                        "avg_confidence": round(avg_confidence, 4),
                        "params_used": {k: v for k, v in tuned.items()
                                       if k not in ("optimizer_score", "optimizer_success_rate", "name")},
                    }

                    self.state.tuning_history.append(metrics)
                    self.state.tuned_metrics = metrics

                    print(f"\n  [튜닝 결과 #{iteration}]")
                    print(f"    성공률: {success_count}/{total} ({success_rate*100:.1f}%)")
                    print(f"    평균 종합점수: {avg_composite:.4f}")

                    best_rate = max(best_rate, success_rate)

                    # 목표 달성 확인
                    if success_rate >= self.config.target_success_rate:
                        print(f"\n  목표 성공률 {self.config.target_success_rate*100:.0f}% 달성!")
                        break

                    # 목표 미달 - 다음 반복에서 실패 분석 재실행
                    if iteration < self.config.max_tuning_iterations:
                        print(f"  목표 미달. 실패 재분석 후 재시도...")

                        # 실패 재분석
                        tuned_results_path = None
                        for p in DATA_DIR.glob(f"benchmark_results_all*{ds_key}*.json"):
                            tuned_results_path = p
                            break

                        if tuned_results_path and tuned_results_path.exists():
                            from tests.failure_analyzer import FailurePatternAnalyzer
                            analyzer = FailurePatternAnalyzer()
                            report = analyzer.analyze(str(tuned_results_path))

                            if report.total_failures > 0:
                                print(f"  잔여 실패: {report.total_failures}건")
                                for ps in report.pattern_summaries[:3]:
                                    print(f"    - {ps.pattern_label}: {ps.count}건")
                else:
                    print("  벤치마크 결과 없음.")
                    break

            finally:
                # 원래 설정 복원
                for settings_key, original_val in original_values.items():
                    setattr(settings, settings_key, original_val)

        if best_rate < self.config.target_success_rate:
            print(f"\n  [주의] {self.config.max_tuning_iterations}회 반복 후에도 "
                  f"목표({self.config.target_success_rate*100:.0f}%) 미달 "
                  f"(최고: {best_rate*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 8: 최종 리포트
    # ------------------------------------------------------------------

    async def _step_final_report(self) -> None:
        """전후 비교 리포트 생성."""
        baseline = self.state.baseline_metrics
        tuned = self.state.tuned_metrics

        print(f"\n{'='*70}")
        print(f"  최종 튜닝 리포트: {self.config.dataset_name}")
        print(f"{'='*70}")

        # 인제스트 정보
        if self.state.ingest_stats and not self.state.ingest_stats.get("skipped"):
            ist = self.state.ingest_stats
            print(f"\n[인제스트]")
            print(f"  총 파일: {ist.get('total_files', 0)}")
            print(f"  성공: {ist.get('success', 0)}, 실패: {ist.get('failed', 0)}")
            print(f"  총 청크: {ist.get('total_chunks', 0)}")

        # 검증 정보
        if self.state.validation_result:
            vr = self.state.validation_result
            stats = vr.get("stats", {})
            print(f"\n[골든 데이터셋]")
            print(f"  경로: {self.state.golden_path}")
            print(f"  총 문항: {stats.get('total_questions', 'N/A')}")
            print(f"  유효: {'예' if vr.get('valid') else '아니오'}")

        # 전후 비교 테이블
        print(f"\n[전후 비교]")
        print(f"  {'항목':20s}  {'베이스라인':>12s}  {'튜닝 후':>12s}  {'변화':>10s}")
        print(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*10}")

        comparisons = [
            ("성공률", "success_rate", True, "%"),
            ("종합 점수", "avg_composite_score", True, ""),
            ("신뢰도", "avg_confidence", True, ""),
        ]

        for label, key, higher_better, fmt in comparisons:
            b_val = baseline.get(key, 0.0)
            t_val = tuned.get(key, 0.0)
            diff = t_val - b_val

            if fmt == "%":
                b_str = f"{b_val*100:.1f}%"
                t_str = f"{t_val*100:.1f}%"
                d_str = f"{diff*100:+.1f}%p"
            else:
                b_str = f"{b_val:.4f}"
                t_str = f"{t_val:.4f}"
                d_str = f"{diff:+.4f}"

            # 개선 여부 표시
            if diff > 0.001 and higher_better:
                indicator = " (+)"
            elif diff < -0.001 and higher_better:
                indicator = " (-)"
            else:
                indicator = ""

            print(f"  {label:20s}  {b_str:>12s}  {t_str:>12s}  {d_str:>10s}{indicator}")

        # 성공 건수
        b_success = baseline.get("success_count", 0)
        b_total = baseline.get("total_questions", 0)
        t_success = tuned.get("success_count", 0)
        t_total = tuned.get("total_questions", 0)
        print(f"  {'성공 건수':20s}  {b_success:>8d}/{b_total:<3d}  {t_success:>8d}/{t_total:<3d}")

        # 실패 분석 요약
        if self.state.failure_analysis and not self.state.failure_analysis.get("skipped"):
            fa = self.state.failure_analysis
            print(f"\n[실패 패턴 (베이스라인)]")
            print(f"  총 실패: {fa.get('total_failures', 0)}건 "
                  f"({fa.get('failure_rate', 0)*100:.1f}%)")
            for p in fa.get("top_patterns", []):
                print(f"    - {p['label']}: {p['count']}건")

        # 튜닝 파라미터
        if self.state.tuned_params:
            tp = self.state.tuned_params
            print(f"\n[적용된 튜닝 파라미터]")
            print(f"  최적 구성: {tp.get('name', 'N/A')}")
            skip_keys = {"name", "optimizer_score", "optimizer_success_rate"}
            for key, val in tp.items():
                if key not in skip_keys:
                    print(f"  {key}: {val}")

        # 튜닝 히스토리
        if self.state.tuning_history:
            print(f"\n[튜닝 이력]")
            for h in self.state.tuning_history:
                print(f"  반복 #{h.get('iteration', '?')}: "
                      f"성공률={h.get('success_rate', 0)*100:.1f}%, "
                      f"점수={h.get('avg_composite_score', 0):.4f}")

        # 목표 달성 여부
        final_rate = tuned.get("success_rate", 0.0)
        target = self.config.target_success_rate
        print(f"\n[결론]")
        if final_rate >= target:
            print(f"  목표 성공률 {target*100:.0f}% 달성! ({final_rate*100:.1f}%)")
        else:
            print(f"  목표 성공률 {target*100:.0f}% 미달 ({final_rate*100:.1f}%)")
            print(f"  추가 튜닝이 필요합니다.")

        print(f"\n{'='*70}\n")

        # 상태에 최종 리포트 저장
        self.state.final_report = {
            "dataset_name": self.config.dataset_name,
            "target_success_rate": target,
            "baseline_success_rate": baseline.get("success_rate", 0.0),
            "tuned_success_rate": final_rate,
            "target_achieved": final_rate >= target,
            "improvement": round(final_rate - baseline.get("success_rate", 0.0), 4),
            "tuning_iterations": len(self.state.tuning_history),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # 상태 관리
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """상태를 JSON 파일에 원자적으로 저장한다."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".json.tmp")

        state_dict = asdict(self.state)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

        # 원자적 교체 (os.replace는 원자적)
        os.replace(str(tmp_path), str(self._state_path))

    def _load_state(self) -> PipelineState | None:
        """저장된 상태를 로드한다."""
        if not self._state_path.exists():
            return None

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = PipelineState(
                dataset_name=data.get("dataset_name", ""),
                started_at=data.get("started_at", ""),
                updated_at=data.get("updated_at", ""),
                completed_steps=data.get("completed_steps", []),
                current_step=data.get("current_step"),
                ingest_stats=data.get("ingest_stats", {}),
                golden_path=data.get("golden_path", ""),
                validation_result=data.get("validation_result", {}),
                baseline_metrics=data.get("baseline_metrics", {}),
                failure_analysis=data.get("failure_analysis", {}),
                tuned_params=data.get("tuned_params", {}),
                tuned_metrics=data.get("tuned_metrics", {}),
                tuning_history=data.get("tuning_history", []),
                final_report=data.get("final_report", {}),
            )
            return state
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("상태 파일 로드 실패: %s", e)
            return None

    def resume(self) -> None:
        """이전 상태에서 재개를 위해 상태를 로드한다."""
        loaded = self._load_state()
        if loaded:
            self.state = loaded
            print(f"  이전 상태 로드 완료: {len(self.state.completed_steps)}개 단계 완료")
            if self.state.completed_steps:
                print(f"  완료된 단계: {', '.join(self.state.completed_steps)}")
        else:
            print("  이전 상태 없음. 처음부터 시작합니다.")


# ===========================================================================
# CLI 진입점
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG 통합 튜닝 파이프라인 - 인제스트부터 최종 리포트까지",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""예시:
  # 전체 파이프라인
  python tests/tuning_pipeline.py --dataset-name "출장보고서" --data-dir data/uploads/국외출장_결과보고서

  # 특정 단계만
  python tests/tuning_pipeline.py --dataset-name "출장보고서" --step baseline_benchmark

  # 범위 지정
  python tests/tuning_pipeline.py --dataset-name "출장보고서" --start-from analyze_failures --stop-after auto_tune

  # 이전 상태에서 재개
  python tests/tuning_pipeline.py --dataset-name "출장보고서" --resume

  # 골든 데이터셋 검증만
  python tests/tuning_pipeline.py --validate-only tests/golden_dataset_travel.json
""",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="문서 디렉토리 경로 (인제스트 단계에 필요)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="데이터셋 이름 (필수, --validate-only 제외)",
    )
    parser.add_argument(
        "--step",
        choices=[s.value for s in PipelineStep],
        default=None,
        help="특정 단계만 실행",
    )
    parser.add_argument(
        "--start-from",
        choices=[s.value for s in PipelineStep],
        default=None,
        help="시작 단계 지정",
    )
    parser.add_argument(
        "--stop-after",
        choices=[s.value for s in PipelineStep],
        default=None,
        help="종료 단계 지정",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="LLM 모델 이름 (예: qwen2.5:14b)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.95,
        help="목표 성공률 (기본값: 0.95)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이전 상태에서 재개",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="골든 데이터셋 검증만 실행 (JSON 경로)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="combined",
        help="옵티마이저 파라미터 그리드 (기본값: combined)",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help='카테고리별 수량 JSON (예: \'{"factual":10,"inference":8,"multi_hop":5,"negative":3}\')',
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help='소스 필터 JSON (예: \'{"source_type": "출장보고서"}\')',
    )
    parser.add_argument(
        "--golden-path",
        type=str,
        default=None,
        help="골든 데이터셋 출력 경로",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="최대 튜닝 반복 횟수 (기본값: 3)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=15,
        help="옵티마이저 샘플 크기 (기본값: 15)",
    )
    parser.add_argument(
        "--dp-mode",
        type=str,
        default="auto",
        choices=["auto", "force_dp", "local_only"],
        help="문서 파싱 모드 (기본값: auto)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력",
    )

    args = parser.parse_args()

    # 로깅 설정
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --validate-only 모드
    if args.validate_only:
        print(f"\n{'='*60}")
        print(f"  골든 데이터셋 검증")
        print(f"{'='*60}")
        print(f"  파일: {args.validate_only}\n")

        result = DatasetValidator.validate(args.validate_only)

        print(f"  유효: {'예' if result['valid'] else '아니오'}")

        if result.get("stats"):
            stats = result["stats"]
            print(f"  총 문항: {stats.get('total_questions', 0)}")
            if stats.get("category_distribution"):
                cat_str = ", ".join(
                    f"{k}({v})" for k, v in stats["category_distribution"].items()
                )
                print(f"  카테고리: {cat_str}")
            if stats.get("difficulty_distribution"):
                diff_str = ", ".join(
                    f"{k}({v})" for k, v in stats["difficulty_distribution"].items()
                )
                print(f"  난이도: {diff_str}")
            if stats.get("answer_length_stats"):
                als = stats["answer_length_stats"]
                print(f"  답변 길이: 최소={als['min']}, 최대={als['max']}, "
                      f"평균={als['avg']}, 중앙값={als['median']}")
            if stats.get("duplicate_pairs", 0) > 0:
                print(f"  유사 질문 쌍: {stats['duplicate_pairs']}개")

        if result.get("errors"):
            print(f"\n  [오류]")
            for err in result["errors"]:
                print(f"    - {err}")

        if result.get("warnings"):
            print(f"\n  [경고]")
            for warn in result["warnings"]:
                print(f"    - {warn}")

        print(f"\n{'='*60}\n")
        sys.exit(0 if result["valid"] else 1)

    # dataset-name 필수 확인
    if not args.dataset_name:
        parser.error("--dataset-name은 필수입니다 (--validate-only 제외)")

    # 카테고리 수량 파싱
    category_distribution = {"factual": 10, "inference": 8, "multi_hop": 5, "negative": 3}
    if args.counts:
        try:
            category_distribution = json.loads(args.counts)
        except json.JSONDecodeError:
            parser.error(f"--counts JSON 파싱 실패: {args.counts}")

    # 소스 필터 파싱
    source_filter = None
    if args.source_filter:
        try:
            source_filter = json.loads(args.source_filter)
        except json.JSONDecodeError:
            parser.error(f"--source-filter JSON 파싱 실패: {args.source_filter}")

    # PipelineConfig 구성
    config = PipelineConfig(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        dp_mode=args.dp_mode,
        golden_output_path=args.golden_path,
        category_distribution=category_distribution,
        source_filter=source_filter,
        model=args.model,
        target_success_rate=args.target,
        max_tuning_iterations=args.max_iterations,
        optimizer_grid=args.grid,
        optimizer_sample_size=args.sample_size,
    )

    # 파이프라인 생성
    pipeline = TuningPipeline(config)

    # 재개 모드
    if args.resume:
        pipeline.resume()

    # 실행 범위 결정
    start_from = None
    stop_after = None

    if args.step:
        # 특정 단계만 실행
        step_enum = PipelineStep(args.step)
        start_from = step_enum
        stop_after = step_enum
    else:
        if args.start_from:
            start_from = PipelineStep(args.start_from)
        if args.stop_after:
            stop_after = PipelineStep(args.stop_after)

    # 비동기 실행
    async def run():
        try:
            await pipeline.run(start_from=start_from, stop_after=stop_after)
        except KeyboardInterrupt:
            print("\n\n  [중단] 사용자에 의해 중단됨. --resume으로 재개 가능.")
            sys.exit(130)
        except Exception as e:
            print(f"\n  [오류] 파이프라인 실패: {e}")
            print(f"  상태 파일: {pipeline._state_path}")
            print(f"  --resume 옵션으로 재개 가능.")
            raise

    asyncio.run(run())


if __name__ == "__main__":
    main()
