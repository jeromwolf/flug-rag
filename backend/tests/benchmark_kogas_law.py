"""한국가스공사법 RAG 벤치마크 테스트 스크립트."""

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
from rag.retriever import HybridRetriever


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


async def run_benchmark():
    """Run full 50-question benchmark."""
    # Load golden dataset
    dataset_path = Path(__file__).parent.parent / "data/sample_dataset/한국가스공사법/golden_dataset_kogas_law.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    print(f"\n{'='*70}")
    print(f"한국가스공사법 RAG 벤치마크 테스트")
    print(f"{'='*70}")
    print(f"총 질문 수: {len(questions)}")
    print(f"카테고리: factual({dataset['dataset_info']['categories']['factual']}), "
          f"inference({dataset['dataset_info']['categories']['inference']}), "
          f"multi_hop({dataset['dataset_info']['categories']['multi_hop']}), "
          f"negative({dataset['dataset_info']['categories']['negative']})")
    print(f"{'='*70}\n")

    # Initialize RAG components
    settings.default_llm_provider = "ollama"
    llm = create_llm()
    retriever = HybridRetriever()
    rag_chain = RAGChain(retriever=retriever, llm=llm)

    results: list[BenchmarkResult] = []
    category_stats = {
        "factual": {"total": 0, "success": 0, "confidence_sum": 0},
        "inference": {"total": 0, "success": 0, "confidence_sum": 0},
        "multi_hop": {"total": 0, "success": 0, "confidence_sum": 0},
        "negative": {"total": 0, "success": 0, "confidence_sum": 0},
    }
    difficulty_stats = {
        "easy": {"total": 0, "success": 0},
        "medium": {"total": 0, "success": 0},
        "hard": {"total": 0, "success": 0},
    }

    total_start = time.time()

    for i, q in enumerate(questions, 1):
        print(f"[{i:02d}/50] Q{q['id']}: {q['question'][:50]}...", end=" ", flush=True)

        start_time = time.time()
        try:
            # Run RAG query
            response = await rag_chain.query(q["question"])
            latency_ms = (time.time() - start_time) * 1000

            # Extract results (RAGResponse is a dataclass)
            answer = response.content
            confidence = response.confidence
            sources = response.sources

            # Determine success (confidence > 0.3 and answer is not empty)
            success = confidence > 0.3 and len(answer) > 20

            result = BenchmarkResult(
                id=q["id"],
                category=q["category"],
                difficulty=q["difficulty"],
                question=q["question"],
                expected_answer=q["answer"],
                actual_answer=answer[:200] + "..." if len(answer) > 200 else answer,
                confidence=confidence,
                sources_count=len(sources),
                latency_ms=latency_ms,
                success=success,
            )
            results.append(result)

            # Update stats
            cat = q["category"]
            diff = q["difficulty"]
            category_stats[cat]["total"] += 1
            category_stats[cat]["confidence_sum"] += confidence
            difficulty_stats[diff]["total"] += 1

            if success:
                category_stats[cat]["success"] += 1
                difficulty_stats[diff]["success"] += 1

            status = "✓" if success else "✗"
            print(f"{status} conf={confidence:.2f}, latency={latency_ms:.0f}ms, sources={len(sources)}")

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

    # Print summary
    print(f"\n{'='*70}")
    print(f"벤치마크 결과 요약")
    print(f"{'='*70}")

    # Overall stats
    total_success = sum(1 for r in results if r.success)
    total_questions = len(results)
    avg_confidence = sum(r.confidence for r in results) / total_questions
    avg_latency = sum(r.latency_ms for r in results) / total_questions

    print(f"\n[전체 통계]")
    print(f"  성공률: {total_success}/{total_questions} ({100*total_success/total_questions:.1f}%)")
    print(f"  평균 신뢰도: {avg_confidence:.3f}")
    print(f"  평균 응답시간: {avg_latency:.0f}ms")
    print(f"  총 소요시간: {total_time:.1f}초")

    # Category stats
    print(f"\n[카테고리별 통계]")
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            success_rate = 100 * stats["success"] / stats["total"]
            avg_conf = stats["confidence_sum"] / stats["total"]
            print(f"  {cat:12s}: {stats['success']:2d}/{stats['total']:2d} ({success_rate:5.1f}%) | avg_conf={avg_conf:.3f}")

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
        for r in failed[:10]:  # Show first 10 failures
            print(f"  Q{r.id} ({r.category}/{r.difficulty}): {r.question[:40]}...")
            print(f"       conf={r.confidence:.2f}, sources={r.sources_count}")

    print(f"\n{'='*70}")

    # Save detailed results to JSON
    output_path = Path(__file__).parent.parent / "data/sample_dataset/한국가스공사법/benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": total_questions,
                "success_count": total_success,
                "success_rate": total_success / total_questions,
                "avg_confidence": avg_confidence,
                "avg_latency_ms": avg_latency,
                "total_time_sec": total_time,
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
                    "latency_ms": r.latency_ms,
                    "success": r.success,
                }
                for r in results
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"상세 결과 저장됨: {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_benchmark())
