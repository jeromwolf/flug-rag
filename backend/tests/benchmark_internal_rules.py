"""한국가스기술공사 내부규정 RAG 벤치마크 테스트 스크립트.

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


# Category-specific success thresholds (tuned per question type)
CATEGORY_THRESHOLDS = {
    "factual": 0.45,
    "inference": 0.45,
    "multi_hop": 0.43,
    "negative": 0.40,
}
DEFAULT_THRESHOLD = 0.50

# Confidence thresholds per category
# negative: low confidence is expected (retriever finds no direct match)
# multi_hop: cross-regulation queries often get low retrieval scores
CONFIDENCE_THRESHOLDS = {
    "factual": 0.15,
    "inference": 0.15,
    "multi_hop": 0.05,
    "negative": 0.0,  # no confidence requirement for negative questions
}
DEFAULT_CONFIDENCE = 0.15

MAX_RETRIES = 1  # retry once on ERROR


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
    # 답변 품질 평가
    semantic_similarity: float = 0.0
    rouge_1: float = 0.0
    rouge_l: float = 0.0
    composite_score: float = 0.0
    grade: str = ""
    length_penalty: float = 1.0
    category_score: float = 0.0


async def run_benchmark(model_name: str | None = None, limit: int | None = None):
    """Run full 60-question benchmark with answer accuracy evaluation."""
    # Load golden dataset
    dataset_path = Path(__file__).parent / "golden_dataset_internal_rules.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    if limit:
        questions = questions[:limit]
    print(f"\n{'='*70}")
    print(f"한국가스기술공사 내부규정 RAG 벤치마크 테스트")
    print(f"{'='*70}")
    print(f"총 질문 수: {len(questions)}")
    print(
        f"카테고리: factual({dataset['dataset_info']['categories']['factual']}), "
        f"inference({dataset['dataset_info']['categories']['inference']}), "
        f"multi_hop({dataset['dataset_info']['categories']['multi_hop']}), "
        f"negative({dataset['dataset_info']['categories']['negative']})"
    )
    print(f"{'='*70}\n")

    # Initialize RAG components
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

    results: list[BenchmarkResult] = []
    category_stats: dict[str, dict] = {
        "factual": {"total": 0, "success": 0, "confidence_sum": 0.0, "score_sum": 0.0, "sim_sum": 0.0, "rouge_sum": 0.0},
        "inference": {"total": 0, "success": 0, "confidence_sum": 0.0, "score_sum": 0.0, "sim_sum": 0.0, "rouge_sum": 0.0},
        "multi_hop": {"total": 0, "success": 0, "confidence_sum": 0.0, "score_sum": 0.0, "sim_sum": 0.0, "rouge_sum": 0.0},
        "negative": {"total": 0, "success": 0, "confidence_sum": 0.0, "score_sum": 0.0, "sim_sum": 0.0, "rouge_sum": 0.0},
    }
    difficulty_stats: dict[str, dict] = {
        "easy": {"total": 0, "success": 0},
        "medium": {"total": 0, "success": 0},
        "hard": {"total": 0, "success": 0},
    }

    total_start = time.time()

    for i, q in enumerate(questions, 1):
        print(f"[{i:02d}/{len(questions)}] Q{q['id']}: {q['question'][:50]}...", end=" ", flush=True)

        start_time = time.time()
        try:
            # Run RAG query with retry on failure
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
                raise last_error
            latency_ms = (time.time() - start_time) * 1000

            answer = response.content
            confidence = response.confidence
            sources = response.sources

            # 답변 평가 (골든 답변 vs 실제 답변)
            eval_result = await evaluator.evaluate(q["answer"], answer)

            # Category-specific composite score
            cat_score = AnswerEvaluator.compute_category_score(eval_result, q["category"], expected=q["answer"], actual=answer)

            # 성공 기준: 카테고리별 임계값 + 카테고리별 confidence 기준
            threshold = CATEGORY_THRESHOLDS.get(q["category"], DEFAULT_THRESHOLD)
            conf_threshold = CONFIDENCE_THRESHOLDS.get(q["category"], DEFAULT_CONFIDENCE)
            success = cat_score >= threshold and confidence > conf_threshold

            result = BenchmarkResult(
                id=q["id"],
                category=q["category"],
                difficulty=q["difficulty"],
                question=q["question"],
                expected_answer=q["answer"],
                actual_answer=answer,  # 전체 답변 저장 (잘림 없음)
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
            category_stats[cat]["total"] += 1
            category_stats[cat]["confidence_sum"] += confidence
            category_stats[cat]["score_sum"] += eval_result.composite_score
            category_stats[cat]["sim_sum"] += eval_result.semantic_similarity
            category_stats[cat]["rouge_sum"] += eval_result.rouge_l
            difficulty_stats[diff]["total"] += 1

            if success:
                category_stats[cat]["success"] += 1
                difficulty_stats[diff]["success"] += 1

            status = "✓" if success else "✗"
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
            print(f"✗ ERROR: {str(e)[:50]}")

            result = BenchmarkResult(
                id=q["id"],
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
            difficulty_stats[diff]["total"] += 1

    total_time = time.time() - total_start

    # ======================== Summary ========================
    print(f"\n{'='*70}")
    print(f"벤치마크 결과 요약")
    print(f"{'='*70}")

    # Overall stats
    total_success = sum(1 for r in results if r.success)
    total_questions = len(results)
    avg_confidence = sum(r.confidence for r in results) / total_questions
    avg_latency = sum(r.latency_ms for r in results) / total_questions

    # 평가 통계 (에러가 아닌 결과만)
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
            )
            for r in evaluated
        ]
    ) if evaluated else {}

    print(f"\n[전체 통계]")
    print(f"  성공률: {total_success}/{total_questions} ({100*total_success/total_questions:.1f}%)")
    print(f"  평균 신뢰도: {avg_confidence:.3f}")
    print(f"  평균 응답시간: {avg_latency:.0f}ms")
    print(f"  총 소요시간: {total_time:.1f}초")

    # 품질 평가 통계
    if eval_summary:
        print(f"\n[품질 평가 통계]")
        print(f"  평균 종합 점수: {eval_summary['avg_composite']:.4f}")
        print(f"  평균 의미 유사도: {eval_summary['avg_semantic']:.4f}")
        print(f"  평균 ROUGE-1: {eval_summary['avg_rouge_1']:.4f}")
        print(f"  평균 ROUGE-L: {eval_summary['avg_rouge_l']:.4f}")
        gd = eval_summary["grade_distribution"]
        print(f"  등급 분포: A({gd.get('A',0)}) B({gd.get('B',0)}) C({gd.get('C',0)}) D({gd.get('D',0)}) F({gd.get('F',0)})")

    # Category stats
    print(f"\n[카테고리별 품질]")
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            n = stats["total"]
            success_rate = 100 * stats["success"] / n
            avg_conf = stats["confidence_sum"] / n
            avg_score = stats["score_sum"] / n
            avg_sim = stats["sim_sum"] / n
            avg_rouge = stats["rouge_sum"] / n
            print(
                f"  {cat:12s}: {stats['success']:2d}/{n:2d} ({success_rate:5.1f}%) "
                f"avg_score={avg_score:.2f} avg_sim={avg_sim:.2f} avg_rouge={avg_rouge:.2f} avg_conf={avg_conf:.3f}"
            )

    # Difficulty stats
    print(f"\n[난이도별 통계]")
    for diff, stats in difficulty_stats.items():
        if stats["total"] > 0:
            success_rate = 100 * stats["success"] / stats["total"]
            print(f"  {diff:8s}: {stats['success']:2d}/{stats['total']:2d} ({success_rate:5.1f}%)")

    # Failed questions
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n[실패한 질문 ({len(failed)}개)]")
        for r in failed[:10]:
            threshold = CATEGORY_THRESHOLDS.get(r.category, DEFAULT_THRESHOLD)
            print(f"  Q{r.id} ({r.category}/{r.difficulty}): {r.question[:40]}...")
            print(
                f"       conf={r.confidence:.2f} score={r.composite_score:.2f}({r.grade}) "
                f"cat_score={r.category_score:.2f}(thr={threshold:.2f}) "
                f"len_pen={r.length_penalty:.2f} "
                f"sim={r.semantic_similarity:.2f} rouge_l={r.rouge_l:.2f}"
            )

    # Low-scoring questions (grade D or F)
    low_quality = [r for r in results if r.grade in ("D", "F") and r.success is False]
    if low_quality:
        print(f"\n[저품질 답변 (D/F 등급, {len(low_quality)}개)]")
        for r in low_quality[:5]:
            print(f"  Q{r.id}: score={r.composite_score:.2f}({r.grade})")
            print(f"       기대: {r.expected_answer[:60]}...")
            print(f"       실제: {r.actual_answer[:60]}...")

    print(f"\n{'='*70}")

    # Save detailed results to JSON
    model_suffix = f"_{actual_model.replace(':', '_').replace('/', '_')}" if model_name else ""
    output_path = (
        Path(__file__).parent.parent
        / f"data/benchmark_results_internal_rules{model_suffix}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_questions": total_questions,
                    "success_count": total_success,
                    "success_rate": round(total_success / total_questions, 4),
                    "avg_confidence": round(avg_confidence, 4),
                    "avg_latency_ms": round(avg_latency, 1),
                    "total_time_sec": round(total_time, 1),
                    "avg_composite_score": eval_summary.get("avg_composite", 0),
                    "avg_semantic_similarity": eval_summary.get("avg_semantic", 0),
                    "avg_rouge_1": eval_summary.get("avg_rouge_1", 0),
                    "avg_rouge_l": eval_summary.get("avg_rouge_l", 0),
                    "grade_distribution": eval_summary.get("grade_distribution", {}),
                },
                "category_stats": category_stats,
                "difficulty_stats": difficulty_stats,
                "results": [
                    {
                        "id": r.id,
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="한국가스기술공사 내부규정 RAG 벤치마크")
    parser.add_argument("--model", "-m", type=str, default=None, help="Ollama model name (e.g. qwen2.5:14b)")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()
    asyncio.run(run_benchmark(model_name=args.model, limit=args.limit))
