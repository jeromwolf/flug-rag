"""골든 데이터셋 배치 평가 엔진.

관리자 UI에서 골든 데이터셋 전체를 RAG 파이프라인에 일괄 실행하고
실시간 SSE 스트리밍으로 진행 상황과 결과를 전달한다.

핵심 로직은 tests/benchmark_evaluation.py와 동일하되,
API 엔드포인트에서 호출 가능하도록 모듈화.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from rag.evaluator import AnswerEvaluator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLDEN_DATASETS_DIR = Path(__file__).parent.parent / "tests" / "golden_datasets"
BENCHMARK_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "benchmarks"

# ---------------------------------------------------------------------------
# Thresholds (canonical — same as benchmark_evaluation.py)
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single question benchmark result."""
    id: int
    category: str
    difficulty: str
    question: str
    expected_answer: str
    actual_answer: str
    confidence: float
    sources_count: int
    latency_ms: float
    success: bool
    semantic_similarity: float = 0.0
    rouge_1: float = 0.0
    rouge_l: float = 0.0
    composite_score: float = 0.0
    grade: str = ""
    length_penalty: float = 1.0
    category_score: float = 0.0


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _new_cat_stats() -> dict:
    return {
        "total": 0, "success": 0,
        "confidence_sum": 0.0, "score_sum": 0.0,
        "sim_sum": 0.0, "rouge_sum": 0.0,
    }


def _new_diff_stats() -> dict:
    return {"total": 0, "success": 0}


def _build_summary(
    results: list[BenchmarkResult],
    category_stats: dict[str, dict],
    difficulty_stats: dict[str, dict],
    total_time_sec: float,
) -> dict[str, Any]:
    """Build summary, category_stats, difficulty_stats dicts from results."""
    n = len(results)
    if n == 0:
        return {"summary": {}, "category_stats": {}, "difficulty_stats": {}}

    success_count = sum(1 for r in results if r.success)
    grade_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in results:
        grade_dist[r.grade] = grade_dist.get(r.grade, 0) + 1

    summary = {
        "total_questions": n,
        "success_count": success_count,
        "success_rate": round(success_count / n, 4),
        "avg_confidence": round(sum(r.confidence for r in results) / n, 4),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
        "total_time_sec": round(total_time_sec, 1),
        "avg_composite_score": round(sum(r.composite_score for r in results) / n, 4),
        "avg_semantic_similarity": round(sum(r.semantic_similarity for r in results) / n, 4),
        "avg_rouge_1": round(sum(r.rouge_1 for r in results) / n, 4),
        "avg_rouge_l": round(sum(r.rouge_l for r in results) / n, 4),
        "grade_distribution": grade_dist,
    }

    cat_out = {}
    for cat, s in category_stats.items():
        t = s["total"]
        if t == 0:
            continue
        cat_out[cat] = {
            "total": t,
            "success": s["success"],
            "success_rate": round(s["success"] / t, 4),
            "avg_confidence": round(s["confidence_sum"] / t, 4),
            "avg_composite_score": round(s["score_sum"] / t, 4),
            "avg_semantic_similarity": round(s["sim_sum"] / t, 4),
            "avg_rouge_l": round(s["rouge_sum"] / t, 4),
        }

    diff_out = {}
    for diff, s in difficulty_stats.items():
        t = s["total"]
        if t == 0:
            continue
        diff_out[diff] = {
            "total": t,
            "success": s["success"],
            "success_rate": round(s["success"] / t, 4),
        }

    return {
        "summary": summary,
        "category_stats": cat_out,
        "difficulty_stats": diff_out,
    }


# ---------------------------------------------------------------------------
# BatchEvaluator
# ---------------------------------------------------------------------------

class BatchEvaluator:
    """골든 데이터셋 배치 평가 엔진."""

    def __init__(self):
        self._evaluator: AnswerEvaluator | None = None

    def _get_evaluator(self) -> AnswerEvaluator:
        if self._evaluator is None:
            logger.info("AnswerEvaluator 초기화 (bge-m3 로드)...")
            self._evaluator = AnswerEvaluator()
        return self._evaluator

    # ---- Dataset listing / loading ----

    @staticmethod
    def list_datasets() -> list[dict]:
        """Scan tests/golden_datasets/*.json and return metadata."""
        datasets = []
        if not GOLDEN_DATASETS_DIR.exists():
            return datasets

        for fp in sorted(GOLDEN_DATASETS_DIR.glob("*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                questions = data.get("questions", [])
                categories = list({q.get("category", "unknown") for q in questions})
                datasets.append({
                    "name": fp.stem,
                    "filename": fp.name,
                    "question_count": len(questions),
                    "categories": sorted(categories),
                    "description": data.get("dataset_info", {}).get("description", ""),
                })
            except Exception as e:
                logger.warning("Failed to read dataset %s: %s", fp, e)

        return datasets

    @staticmethod
    def load_dataset(name: str, limit: int | None = None) -> list[dict]:
        """Load a named dataset. Normalizes answer/source fields."""
        fp = GOLDEN_DATASETS_DIR / f"{name}.json"
        if not fp.exists():
            # Try with .json extension already in name
            fp = GOLDEN_DATASETS_DIR / name
        if not fp.exists():
            raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {name}")

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])

        # Normalize fields
        for q in questions:
            if "answer" not in q and "expected_answer" in q:
                q["answer"] = q["expected_answer"]
            if "source_regulation" not in q and "source_document" in q:
                q["source_regulation"] = q["source_document"]
            if "source_regulation" not in q and "source_article" in q:
                q["source_regulation"] = q["source_article"]
            q.setdefault("difficulty", "medium")
            q.setdefault("category", "factual")

        if limit and limit > 0:
            questions = questions[:limit]

        return questions

    # ---- History ----

    @staticmethod
    def list_history() -> list[dict]:
        """List previous benchmark result files."""
        history = []
        if not BENCHMARK_OUTPUT_DIR.exists():
            return history

        for fp in sorted(BENCHMARK_OUTPUT_DIR.glob("batch_eval_*.json"), reverse=True):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                summary = data.get("summary", {})
                history.append({
                    "filename": fp.name,
                    "dataset": data.get("dataset", ""),
                    "total_questions": summary.get("total_questions", 0),
                    "success_rate": summary.get("success_rate", 0),
                    "created_at": data.get("created_at", ""),
                })
            except Exception:
                pass

        return history

    @staticmethod
    def get_history_result(filename: str) -> dict | None:
        """Load a specific history result."""
        fp = BENCHMARK_OUTPUT_DIR / filename
        if not fp.exists():
            return None
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- Core evaluation stream ----

    async def run_stream(
        self,
        dataset_name: str,
        limit: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Run batch evaluation as an async generator yielding SSE events.

        Events:
            init: {"total": N, "dataset": "...", "started_at": "..."}
            progress: {"done": i, "total": N, "result": {...BenchmarkResult...}}
            complete: {"summary": {...}, "category_stats": {...}, "difficulty_stats": {...}}
            error: {"message": "..."}
        """
        from rag import RAGChain

        # Load dataset
        try:
            questions = self.load_dataset(dataset_name, limit)
        except FileNotFoundError as e:
            yield {"event": "error", "data": {"message": str(e)}}
            return

        total = len(questions)
        if total == 0:
            yield {"event": "error", "data": {"message": "데이터셋에 문항이 없습니다."}}
            return

        # Init event
        started_at = datetime.now(timezone.utc).isoformat()
        yield {
            "event": "init",
            "data": {"total": total, "dataset": dataset_name, "started_at": started_at},
        }

        # Initialize evaluator (slow — loads bge-m3)
        evaluator = self._get_evaluator()

        # Initialize RAG chain
        rag_chain = RAGChain()

        # Prepare stats accumulators
        cat_counts: dict[str, int] = {}
        for q in questions:
            cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
        category_stats: dict[str, dict] = {cat: _new_cat_stats() for cat in cat_counts}
        difficulty_stats: dict[str, dict] = {}

        results: list[BenchmarkResult] = []
        total_start = time.time()

        for i, q in enumerate(questions, 1):
            start_time = time.time()
            try:
                # Query RAG pipeline
                response = await rag_chain.query(q["question"])
                content = response.content
                confidence = response.confidence
                sources = response.sources
                latency_ms = (time.time() - start_time) * 1000

                # Evaluate
                eval_result = await evaluator.evaluate(q["answer"], content)
                cat_score = AnswerEvaluator.compute_category_score(
                    eval_result, q["category"],
                    expected=q["answer"], actual=content,
                )

                # Pass/fail decision
                threshold = CATEGORY_THRESHOLDS.get(q["category"], DEFAULT_THRESHOLD)
                conf_threshold = CONFIDENCE_THRESHOLDS.get(q["category"], DEFAULT_CONFIDENCE)
                success = cat_score >= threshold and confidence > conf_threshold

                result = BenchmarkResult(
                    id=q.get("id", i),
                    category=q["category"],
                    difficulty=q.get("difficulty", "medium"),
                    question=q["question"],
                    expected_answer=q["answer"],
                    actual_answer=content,
                    confidence=round(confidence, 4),
                    sources_count=len(sources),
                    latency_ms=round(latency_ms, 1),
                    success=success,
                    semantic_similarity=eval_result.semantic_similarity,
                    rouge_1=eval_result.rouge_1,
                    rouge_l=eval_result.rouge_l,
                    composite_score=eval_result.composite_score,
                    grade=eval_result.grade,
                    length_penalty=eval_result.length_penalty,
                    category_score=cat_score,
                )

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error("Question %d failed: %s", q.get("id", i), e)
                result = BenchmarkResult(
                    id=q.get("id", i),
                    category=q["category"],
                    difficulty=q.get("difficulty", "medium"),
                    question=q["question"],
                    expected_answer=q["answer"],
                    actual_answer=f"[ERROR] {str(e)[:200]}",
                    confidence=0.0,
                    sources_count=0,
                    latency_ms=round(latency_ms, 1),
                    success=False,
                    grade="F",
                )

            results.append(result)

            # Update stats
            cat = q["category"]
            diff = q.get("difficulty", "medium")

            category_stats[cat]["total"] += 1
            category_stats[cat]["confidence_sum"] += result.confidence
            category_stats[cat]["score_sum"] += result.composite_score
            category_stats[cat]["sim_sum"] += result.semantic_similarity
            category_stats[cat]["rouge_sum"] += result.rouge_l

            if diff not in difficulty_stats:
                difficulty_stats[diff] = _new_diff_stats()
            difficulty_stats[diff]["total"] += 1

            if result.success:
                category_stats[cat]["success"] += 1
                difficulty_stats[diff]["success"] += 1

            # Yield progress event
            yield {
                "event": "progress",
                "data": {
                    "done": i,
                    "total": total,
                    "result": asdict(result),
                },
            }

        # Final summary
        total_time = time.time() - total_start
        stats = _build_summary(results, category_stats, difficulty_stats, total_time)

        # Save results to file
        try:
            BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = BENCHMARK_OUTPUT_DIR / f"batch_eval_{dataset_name}_{ts}.json"
            output_data = {
                "dataset": dataset_name,
                "created_at": started_at,
                **stats,
                "results": [asdict(r) for r in results],
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved batch eval results to %s", output_path)
        except Exception as e:
            logger.warning("Failed to save results: %s", e)

        # Yield complete event
        yield {"event": "complete", "data": stats}
