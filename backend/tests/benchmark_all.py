"""통합 RAG 벤치마크 테스트 스크립트.

모든 골든 데이터셋(내부규정, 홍보물, 출장보고서, ALIO공시)을 한번에 평가한다.
답변 정확도를 의미 유사도(Semantic Similarity) + ROUGE 점수로 평가한다.
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from core.llm import create_llm
from rag.chain import RAGChain
from rag.evaluator import AnswerEvaluation, AnswerEvaluator
from rag.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"

DATASET_CONFIGS = {
    "internal_rules": {
        "name": "내부규정",
        "path": TESTS_DIR / "golden_dataset_internal_rules.json",
    },
    "brochure": {
        "name": "홍보물",
        "path": TESTS_DIR / "golden_dataset_brochure.json",
    },
    "travel": {
        "name": "출장보고서",
        "path": TESTS_DIR / "golden_dataset_travel.json",
    },
    "alio": {
        "name": "ALIO공시",
        "path": TESTS_DIR / "golden_dataset_alio.json",
    },
}

# ---------------------------------------------------------------------------
# Thresholds (reused from benchmark_internal_rules.py)
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

MAX_RETRIES = 1


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single question benchmark result."""

    id: int
    dataset: str
    category: str
    difficulty: str
    question: str
    expected_answer: str
    actual_answer: str
    confidence: float
    sources_count: int
    latency_ms: float
    success: bool
    # 답변 품질 평가
    semantic_similarity: float = 0.0
    rouge_1: float = 0.0
    rouge_l: float = 0.0
    composite_score: float = 0.0
    grade: str = ""
    length_penalty: float = 1.0
    category_score: float = 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(key: str) -> tuple[str, list[dict]]:
    """Load a golden dataset by key. Returns (display_name, questions)."""
    config = DATASET_CONFIGS[key]
    path = config["path"]
    if not path.exists():
        print(f"  경고: {path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return config["name"], []

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset.get("questions", [])
    # Tag each question with dataset key
    for q in questions:
        q["_dataset"] = key
        # Normalize source field: handle both source_regulation and source_document
        if "source_regulation" not in q and "source_document" in q:
            q["source_regulation"] = q["source_document"]
        elif "source_document" not in q and "source_regulation" in q:
            q["source_document"] = q["source_regulation"]

    return config["name"], questions


def load_all_datasets(
    dataset_filter: str = "all",
    limit: int | None = None,
) -> list[dict]:
    """Load and merge questions from all (or selected) datasets.

    Args:
        dataset_filter: 'all' or a specific dataset key
        limit: max questions per dataset (None = no limit)

    Returns:
        Merged list of question dicts, each tagged with _dataset key.
    """
    all_questions: list[dict] = []
    keys = list(DATASET_CONFIGS.keys()) if dataset_filter == "all" else [dataset_filter]

    for key in keys:
        name, questions = load_dataset(key)
        if limit:
            questions = questions[:limit]
        print(f"  {name}: {len(questions)}문항 로드됨")
        all_questions.extend(questions)

    return all_questions


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _new_cat_stats() -> dict:
    return {"total": 0, "success": 0, "confidence_sum": 0.0, "score_sum": 0.0, "sim_sum": 0.0, "rouge_sum": 0.0}


def _new_diff_stats() -> dict:
    return {"total": 0, "success": 0}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    model_name: str | None = None,
    limit: int | None = None,
    dataset_filter: str = "all",
):
    """Run unified benchmark across all golden datasets.

    Args:
        model_name: Ollama model name override
        limit: Max questions per dataset
        dataset_filter: 'all' or specific dataset key
    """
    print(f"\n{'='*70}")
    print(f"통합 RAG 벤치마크 테스트")
    print(f"{'='*70}")
    print(f"대상 데이터셋: {dataset_filter}")
    print()

    # Load datasets
    questions = load_all_datasets(dataset_filter=dataset_filter, limit=limit)
    if not questions:
        print("로드된 질문이 없습니다. 종료합니다.")
        return []

    total_count = len(questions)
    print(f"\n총 질문 수: {total_count}")

    # Count per dataset
    ds_counts: dict[str, int] = {}
    for q in questions:
        ds = q["_dataset"]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
    for ds_key, cnt in ds_counts.items():
        ds_name = DATASET_CONFIGS[ds_key]["name"]
        print(f"  {ds_name}: {cnt}문항")

    # Count per category
    cat_counts: dict[str, int] = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    cat_summary = ", ".join(f"{cat}({cnt})" for cat, cnt in sorted(cat_counts.items()))
    print(f"카테고리: {cat_summary}")
    print(f"{'='*70}\n")

    # ---- Initialize RAG components ----
    settings.default_llm_provider = "ollama"
    llm = create_llm(model=model_name) if model_name else create_llm()
    actual_model = model_name or settings.ollama_model
    print(f"사용 모델: {actual_model}")

    retriever = HybridRetriever()
    rag_chain = RAGChain(retriever=retriever, llm=llm)

    # Initialize evaluator
    print("평가 엔진 초기화 중 (bge-m3 임베딩 로드)...")
    evaluator = AnswerEvaluator()
    print("평가 엔진 준비 완료.\n")

    # ---- Stats accumulators ----
    results: list[BenchmarkResult] = []

    # Per-dataset stats
    dataset_stats: dict[str, dict] = {
        key: {"name": cfg["name"], "total": 0, "success": 0, "confidence_sum": 0.0, "score_sum": 0.0}
        for key, cfg in DATASET_CONFIGS.items()
        if key in ds_counts
    }

    # Per-category stats (across all datasets)
    category_stats: dict[str, dict] = {}
    for cat in cat_counts:
        category_stats[cat] = _new_cat_stats()

    # Per-difficulty stats
    difficulty_stats: dict[str, dict] = {}

    total_start = time.time()

    for i, q in enumerate(questions, 1):
        ds_key = q["_dataset"]
        ds_name = DATASET_CONFIGS[ds_key]["name"]
        print(
            f"[{i:03d}/{total_count}] [{ds_name}] Q{q['id']}: {q['question'][:45]}...",
            end=" ",
            flush=True,
        )

        start_time = time.time()
        try:
            # Run RAG query with retry
            response = None
            last_error = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = await rag_chain.query(q["question"])
                    break
                except Exception as retry_err:
                    last_error = retry_err
                    if attempt < MAX_RETRIES:
                        print(f"(retry {attempt+1}) ", end="", flush=True)
                        await asyncio.sleep(2)
            if response is None:
                raise last_error  # type: ignore[misc]

            latency_ms = (time.time() - start_time) * 1000
            answer = response.content
            confidence = response.confidence
            sources = response.sources

            # Evaluate answer
            eval_result = await evaluator.evaluate(q["answer"], answer)
            cat_score = AnswerEvaluator.compute_category_score(eval_result, q["category"], expected=q["answer"], actual=answer)

            # Success criteria
            threshold = CATEGORY_THRESHOLDS.get(q["category"], DEFAULT_THRESHOLD)
            conf_threshold = CONFIDENCE_THRESHOLDS.get(q["category"], DEFAULT_CONFIDENCE)
            success = cat_score >= threshold and confidence > conf_threshold

            result = BenchmarkResult(
                id=q["id"],
                dataset=ds_key,
                category=q["category"],
                difficulty=q["difficulty"],
                question=q["question"],
                expected_answer=q["answer"],
                actual_answer=answer,
                confidence=confidence,
                sources_count=len(sources),
                latency_ms=latency_ms,
                success=success,
                semantic_similarity=eval_result.semantic_similarity,
                rouge_1=eval_result.rouge_1,
                rouge_l=eval_result.rouge_l,
                composite_score=eval_result.composite_score,
                grade=eval_result.grade,
                length_penalty=eval_result.length_penalty,
                category_score=cat_score,
            )
            results.append(result)

            # Update stats
            cat = q["category"]
            diff = q["difficulty"]

            # Category stats
            category_stats[cat]["total"] += 1
            category_stats[cat]["confidence_sum"] += confidence
            category_stats[cat]["score_sum"] += eval_result.composite_score
            category_stats[cat]["sim_sum"] += eval_result.semantic_similarity
            category_stats[cat]["rouge_sum"] += eval_result.rouge_l

            # Difficulty stats
            if diff not in difficulty_stats:
                difficulty_stats[diff] = _new_diff_stats()
            difficulty_stats[diff]["total"] += 1

            # Dataset stats
            if ds_key in dataset_stats:
                dataset_stats[ds_key]["total"] += 1
                dataset_stats[ds_key]["confidence_sum"] += confidence
                dataset_stats[ds_key]["score_sum"] += eval_result.composite_score

            if success:
                category_stats[cat]["success"] += 1
                difficulty_stats[diff]["success"] += 1
                if ds_key in dataset_stats:
                    dataset_stats[ds_key]["success"] += 1

            status = "OK" if success else "FAIL"
            print(
                f"{status} conf={confidence:.2f} "
                f"score={eval_result.composite_score:.2f}({eval_result.grade}) "
                f"cat_score={cat_score:.2f} "
                f"len_pen={eval_result.length_penalty:.2f} "
                f"rouge_l={eval_result.rouge_l:.2f} "
                f"latency={latency_ms:.0f}ms"
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"FAIL ERROR: {str(e)[:50]}")

            result = BenchmarkResult(
                id=q["id"],
                dataset=ds_key,
                category=q["category"],
                difficulty=q["difficulty"],
                question=q["question"],
                expected_answer=q["answer"],
                actual_answer=f"ERROR: {str(e)}",
                confidence=0.0,
                sources_count=0,
                latency_ms=latency_ms,
                success=False,
            )
            results.append(result)

            cat = q["category"]
            diff = q["difficulty"]
            category_stats[cat]["total"] += 1
            if diff not in difficulty_stats:
                difficulty_stats[diff] = _new_diff_stats()
            difficulty_stats[diff]["total"] += 1
            if ds_key in dataset_stats:
                dataset_stats[ds_key]["total"] += 1

    total_time = time.time() - total_start

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"통합 벤치마크 결과 요약")
    print(f"{'='*70}")

    total_success = sum(1 for r in results if r.success)
    total_questions = len(results)
    avg_confidence = sum(r.confidence for r in results) / total_questions if total_questions else 0
    avg_latency = sum(r.latency_ms for r in results) / total_questions if total_questions else 0

    # Evaluation summary (non-error results only)
    evaluated = [r for r in results if r.grade]
    eval_summary = AnswerEvaluator.summarize(
        [
            AnswerEvaluation(
                semantic_similarity=r.semantic_similarity,
                rouge_1=r.rouge_1,
                rouge_l=r.rouge_l,
                answer_length=len(r.actual_answer),
                expected_length=len(r.expected_answer),
                length_ratio=round(len(r.actual_answer) / len(r.expected_answer), 2) if len(r.expected_answer) > 0 else 0.0,
                composite_score=r.composite_score,
                grade=r.grade,
                length_penalty=r.length_penalty,
            )
            for r in evaluated
        ]
    ) if evaluated else {}

    # ---- Overall ----
    print(f"\n[전체 통계]")
    print(f"  성공률: {total_success}/{total_questions} ({100*total_success/total_questions:.1f}%)")
    print(f"  평균 신뢰도: {avg_confidence:.3f}")
    print(f"  평균 응답시간: {avg_latency:.0f}ms")
    print(f"  총 소요시간: {total_time:.1f}초")

    if eval_summary:
        print(f"\n[품질 평가 통계]")
        print(f"  평균 종합 점수: {eval_summary['avg_composite']:.4f}")
        print(f"  평균 의미 유사도: {eval_summary['avg_semantic']:.4f}")
        print(f"  평균 ROUGE-1: {eval_summary['avg_rouge_1']:.4f}")
        print(f"  평균 ROUGE-L: {eval_summary['avg_rouge_l']:.4f}")
        gd = eval_summary["grade_distribution"]
        print(f"  등급 분포: A({gd.get('A',0)}) B({gd.get('B',0)}) C({gd.get('C',0)}) D({gd.get('D',0)}) F({gd.get('F',0)})")

    # ---- Per-dataset breakdown ----
    print(f"\n[데이터셋별 성적]")
    for ds_key, stats in dataset_stats.items():
        n = stats["total"]
        if n > 0:
            success_rate = 100 * stats["success"] / n
            avg_score = stats["score_sum"] / n
            avg_conf = stats["confidence_sum"] / n
            print(
                f"  {stats['name']:12s}: {stats['success']:3d}/{n:3d} ({success_rate:5.1f}%) "
                f"avg_score={avg_score:.3f} avg_conf={avg_conf:.3f}"
            )

    # ---- Per-category breakdown ----
    print(f"\n[카테고리별 품질]")
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        if stats["total"] > 0:
            n = stats["total"]
            success_rate = 100 * stats["success"] / n
            avg_conf = stats["confidence_sum"] / n
            avg_score = stats["score_sum"] / n
            avg_sim = stats["sim_sum"] / n
            avg_rouge = stats["rouge_sum"] / n
            print(
                f"  {cat:12s}: {stats['success']:3d}/{n:3d} ({success_rate:5.1f}%) "
                f"avg_score={avg_score:.2f} avg_sim={avg_sim:.2f} avg_rouge={avg_rouge:.2f} avg_conf={avg_conf:.3f}"
            )

    # ---- Per-difficulty breakdown ----
    print(f"\n[난이도별 통계]")
    for diff in ["easy", "medium", "hard"]:
        stats = difficulty_stats.get(diff)
        if stats and stats["total"] > 0:
            success_rate = 100 * stats["success"] / stats["total"]
            print(f"  {diff:8s}: {stats['success']:3d}/{stats['total']:3d} ({success_rate:5.1f}%)")

    # ---- Failed questions ----
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n[실패한 질문 ({len(failed)}개)]")
        for r in failed[:20]:
            ds_name = DATASET_CONFIGS.get(r.dataset, {}).get("name", r.dataset)
            threshold = CATEGORY_THRESHOLDS.get(r.category, DEFAULT_THRESHOLD)
            print(f"  [{ds_name}] Q{r.id} ({r.category}/{r.difficulty}): {r.question[:40]}...")
            print(
                f"       conf={r.confidence:.2f} score={r.composite_score:.2f}({r.grade}) "
                f"cat_score={r.category_score:.2f}(thr={threshold:.2f}) "
                f"len_pen={r.length_penalty:.2f} "
                f"sim={r.semantic_similarity:.2f} rouge_l={r.rouge_l:.2f}"
            )

    # ---- Low quality answers ----
    low_quality = [r for r in results if r.grade in ("D", "F") and not r.success]
    if low_quality:
        print(f"\n[저품질 답변 (D/F 등급, {len(low_quality)}개)]")
        for r in low_quality[:10]:
            ds_name = DATASET_CONFIGS.get(r.dataset, {}).get("name", r.dataset)
            print(f"  [{ds_name}] Q{r.id}: score={r.composite_score:.2f}({r.grade})")
            print(f"       기대: {r.expected_answer[:60]}...")
            print(f"       실제: {r.actual_answer[:60]}...")

    print(f"\n{'='*70}")

    # ==================================================================
    # Save results to JSON
    # ==================================================================
    model_suffix = f"_{actual_model.replace(':', '_').replace('/', '_')}" if model_name else ""
    ds_suffix = f"_{dataset_filter}" if dataset_filter != "all" else ""
    output_path = DATA_DIR / f"benchmark_results_all{ds_suffix}{model_suffix}.json"

    # Build per-dataset stats for JSON
    dataset_stats_json = {}
    for ds_key, stats in dataset_stats.items():
        n = stats["total"]
        dataset_stats_json[ds_key] = {
            "name": stats["name"],
            "total": n,
            "success": stats["success"],
            "success_rate": round(stats["success"] / n, 4) if n > 0 else 0.0,
            "avg_confidence": round(stats["confidence_sum"] / n, 4) if n > 0 else 0.0,
            "avg_composite_score": round(stats["score_sum"] / n, 4) if n > 0 else 0.0,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_questions": total_questions,
                    "success_count": total_success,
                    "success_rate": round(total_success / total_questions, 4) if total_questions else 0,
                    "avg_confidence": round(avg_confidence, 4),
                    "avg_latency_ms": round(avg_latency, 1),
                    "total_time_sec": round(total_time, 1),
                    "avg_composite_score": eval_summary.get("avg_composite", 0),
                    "avg_semantic_similarity": eval_summary.get("avg_semantic", 0),
                    "avg_rouge_1": eval_summary.get("avg_rouge_1", 0),
                    "avg_rouge_l": eval_summary.get("avg_rouge_l", 0),
                    "grade_distribution": eval_summary.get("grade_distribution", {}),
                    "model": actual_model,
                    "dataset_filter": dataset_filter,
                },
                "dataset_stats": dataset_stats_json,
                "category_stats": {
                    cat: {
                        "total": s["total"],
                        "success": s["success"],
                        "success_rate": round(s["success"] / s["total"], 4) if s["total"] > 0 else 0.0,
                        "avg_confidence": round(s["confidence_sum"] / s["total"], 4) if s["total"] > 0 else 0.0,
                        "avg_composite_score": round(s["score_sum"] / s["total"], 4) if s["total"] > 0 else 0.0,
                        "avg_semantic_similarity": round(s["sim_sum"] / s["total"], 4) if s["total"] > 0 else 0.0,
                        "avg_rouge_l": round(s["rouge_sum"] / s["total"], 4) if s["total"] > 0 else 0.0,
                    }
                    for cat, s in category_stats.items()
                },
                "difficulty_stats": {
                    diff: {
                        "total": s["total"],
                        "success": s["success"],
                        "success_rate": round(s["success"] / s["total"], 4) if s["total"] > 0 else 0.0,
                    }
                    for diff, s in difficulty_stats.items()
                },
                "results": [
                    {
                        "id": r.id,
                        "dataset": r.dataset,
                        "category": r.category,
                        "difficulty": r.difficulty,
                        "question": r.question,
                        "expected_answer": r.expected_answer,
                        "actual_answer": r.actual_answer,
                        "confidence": r.confidence,
                        "sources_count": r.sources_count,
                        "latency_ms": round(r.latency_ms, 1),
                        "success": r.success,
                        "semantic_similarity": r.semantic_similarity,
                        "rouge_1": r.rouge_1,
                        "rouge_l": r.rouge_l,
                        "composite_score": r.composite_score,
                        "grade": r.grade,
                        "length_penalty": r.length_penalty,
                        "category_score": r.category_score,
                    }
                    for r in results
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"상세 결과 저장됨: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="통합 RAG 벤치마크 (내부규정 + 홍보물 + 출장보고서 + ALIO공시)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ollama model name (e.g. qwen2.5:14b)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of questions per dataset",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        choices=["internal_rules", "brochure", "travel", "alio", "all"],
        help="Run specific dataset only (default: all)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            model_name=args.model,
            limit=args.limit,
            dataset_filter=args.dataset,
        )
    )
