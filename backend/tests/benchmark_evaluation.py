"""RAG 블라인드 평가용 벤치마크 스크립트.

golden_dataset_evaluation.json (240문항)을 대상으로 두 가지 모드를 지원한다.
  --mode local (기본): RAGChain 직접 호출 (benchmark_all.py 방식)
  --mode api  : RunPod 등 원격 API 서버에 HTTP 요청

사용 예)
  # 로컬 모드
  python tests/benchmark_evaluation.py --model qwen2.5:14b --limit 10

  # API 모드 (RunPod)
  python tests/benchmark_evaluation.py \\
    --mode api \\
    --api-url https://xxx.proxy.runpod.net \\
    --api-user admin \\
    --api-password admin123 \\
    --limit 20
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"

DATASET_PATH = TESTS_DIR / "golden_dataset_evaluation.json"
OUTPUT_PATH = DATA_DIR / "benchmark_results_evaluation.json"


# ---------------------------------------------------------------------------
# Thresholds (same as benchmark_all.py)
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

def load_dataset(limit: int | None = None) -> list[dict]:
    """Load golden_dataset_evaluation.json.

    Maps 'expected_answer' field so the rest of the code can treat it
    uniformly as q['answer'].
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"데이터셋 파일이 없습니다: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset.get("questions", [])

    # Normalize: expose expected_answer as 'answer' for evaluator calls,
    # while keeping original field intact.
    for q in questions:
        q["answer"] = q.get("expected_answer", "")
        # Normalize source fields
        if "source_regulation" not in q and "source_document" in q:
            q["source_regulation"] = q["source_document"]

    if limit:
        questions = questions[:limit]

    return questions


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _new_cat_stats() -> dict:
    return {"total": 0, "success": 0, "confidence_sum": 0.0, "score_sum": 0.0, "sim_sum": 0.0, "rouge_sum": 0.0}


def _new_diff_stats() -> dict:
    return {"total": 0, "success": 0}


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

async def api_login(session, base_url: str, username: str, password: str) -> str:
    """POST /api/auth/login and return JWT access token."""
    import aiohttp

    url = f"{base_url.rstrip('/')}/api/auth/login"
    payload = {"username": username, "password": password}
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"로그인 실패 (HTTP {resp.status}): {body[:200]}")
        data = await resp.json()

    # Support both {access_token: ...} and {token: ...} shapes
    token = data.get("access_token") or data.get("token")
    if not token:
        raise RuntimeError(f"응답에서 토큰을 찾을 수 없음: {list(data.keys())}")
    return token


async def api_query(session, base_url: str, token: str, question: str) -> tuple[str, float, list]:
    """POST /api/chat and return (content, confidence, sources)."""
    url = f"{base_url.rstrip('/')}/api/chat"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"message": question, "mode": "auto"}

    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"API 응답 오류 (HTTP {resp.status}): {body[:200]}")
        data = await resp.json()

    content = data.get("content") or data.get("answer") or data.get("message") or ""
    confidence = float(data.get("confidence", 0.0))
    sources = data.get("sources") or []
    return content, confidence, sources


# ---------------------------------------------------------------------------
# Local RAG query helper
# ---------------------------------------------------------------------------

async def local_query(rag_chain, question: str) -> tuple[str, float, list]:
    """Query via local RAGChain. Returns (content, confidence, sources)."""
    response = await rag_chain.query(question)
    return response.content, response.confidence, response.sources


# ---------------------------------------------------------------------------
# Core benchmark loop (shared between modes)
# ---------------------------------------------------------------------------

async def run_benchmark_loop(
    questions: list[dict],
    query_fn,  # async callable(question: str) -> (content, confidence, sources)
    evaluator,
    model_label: str,
    mode: str,
) -> list[BenchmarkResult]:
    """Evaluate all questions using query_fn and evaluator.

    Args:
        questions: list of question dicts (must have 'answer' key).
        query_fn: async function that returns (content, confidence, sources).
        evaluator: AnswerEvaluator instance.
        model_label: display string for model/mode.
        mode: 'local' or 'api'.

    Returns:
        List of BenchmarkResult.
    """
    from rag.evaluator import AnswerEvaluator

    total_count = len(questions)
    results: list[BenchmarkResult] = []

    # Aggregate stats
    cat_counts: dict[str, int] = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1

    category_stats: dict[str, dict] = {cat: _new_cat_stats() for cat in cat_counts}
    difficulty_stats: dict[str, dict] = {}

    total_start = time.time()

    for i, q in enumerate(questions, 1):
        print(
            f"[{i:03d}/{total_count}] Q{q['id']} ({q['category']}): {q['question'][:45]}...",
            end=" ",
            flush=True,
        )

        start_time = time.time()
        try:
            # Query with retry
            content = None
            confidence = 0.0
            sources = []
            last_error = None

            for attempt in range(MAX_RETRIES + 1):
                try:
                    content, confidence, sources = await query_fn(q["question"])
                    break
                except Exception as retry_err:
                    last_error = retry_err
                    if attempt < MAX_RETRIES:
                        print(f"(retry {attempt+1}) ", end="", flush=True)
                        await asyncio.sleep(2)

            if content is None:
                raise last_error  # type: ignore[misc]

            latency_ms = (time.time() - start_time) * 1000

            # Evaluate
            eval_result = await evaluator.evaluate(q["answer"], content)
            cat_score = AnswerEvaluator.compute_category_score(
                eval_result, q["category"], expected=q["answer"], actual=content
            )

            # Success criteria
            threshold = CATEGORY_THRESHOLDS.get(q["category"], DEFAULT_THRESHOLD)
            conf_threshold = CONFIDENCE_THRESHOLDS.get(q["category"], DEFAULT_CONFIDENCE)
            success = cat_score >= threshold and confidence > conf_threshold

            result = BenchmarkResult(
                id=q["id"],
                category=q["category"],
                difficulty=q["difficulty"],
                question=q["question"],
                expected_answer=q["answer"],
                actual_answer=content,
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

            if diff not in difficulty_stats:
                difficulty_stats[diff] = _new_diff_stats()
            difficulty_stats[diff]["total"] += 1

            if success:
                category_stats[cat]["success"] += 1
                difficulty_stats[diff]["success"] += 1

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
            print(f"FAIL ERROR: {str(e)[:60]}")

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
            if diff not in difficulty_stats:
                difficulty_stats[diff] = _new_diff_stats()
            difficulty_stats[diff]["total"] += 1

    total_time = time.time() - total_start

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"RAG 블라인드 평가 벤치마크 결과 ({mode.upper()} 모드)")
    print(f"{'='*70}")

    total_success = sum(1 for r in results if r.success)
    total_questions = len(results)
    avg_confidence = sum(r.confidence for r in results) / total_questions if total_questions else 0
    avg_latency = sum(r.latency_ms for r in results) / total_questions if total_questions else 0

    # Evaluation summary
    from rag.evaluator import AnswerEvaluation

    evaluated = [r for r in results if r.grade]
    eval_summary = AnswerEvaluator.summarize(
        [
            AnswerEvaluation(
                semantic_similarity=r.semantic_similarity,
                rouge_1=r.rouge_1,
                rouge_l=r.rouge_l,
                answer_length=len(r.actual_answer),
                expected_length=len(r.expected_answer),
                length_ratio=round(len(r.actual_answer) / len(r.expected_answer), 2)
                if len(r.expected_answer) > 0
                else 0.0,
                composite_score=r.composite_score,
                grade=r.grade,
                length_penalty=r.length_penalty,
            )
            for r in evaluated
        ]
    ) if evaluated else {}

    print(f"\n[전체 통계]")
    print(f"  모드: {mode.upper()} | 모델/서버: {model_label}")
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

    print(f"\n[난이도별 통계]")
    for diff in ["easy", "medium", "hard"]:
        stats = difficulty_stats.get(diff)
        if stats and stats["total"] > 0:
            success_rate = 100 * stats["success"] / stats["total"]
            print(f"  {diff:8s}: {stats['success']:3d}/{stats['total']:3d} ({success_rate:5.1f}%)")

    # Failed questions
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n[실패한 질문 ({len(failed)}개)]")
        for r in failed[:20]:
            threshold = CATEGORY_THRESHOLDS.get(r.category, DEFAULT_THRESHOLD)
            print(f"  Q{r.id} ({r.category}/{r.difficulty}): {r.question[:40]}...")
            print(
                f"       conf={r.confidence:.2f} score={r.composite_score:.2f}({r.grade}) "
                f"cat_score={r.category_score:.2f}(thr={threshold:.2f}) "
                f"len_pen={r.length_penalty:.2f} "
                f"sim={r.semantic_similarity:.2f} rouge_l={r.rouge_l:.2f}"
            )

    # Low quality answers
    low_quality = [r for r in results if r.grade in ("D", "F") and not r.success]
    if low_quality:
        print(f"\n[저품질 답변 (D/F 등급, {len(low_quality)}개)]")
        for r in low_quality[:10]:
            print(f"  Q{r.id}: score={r.composite_score:.2f}({r.grade})")
            print(f"       기대: {r.expected_answer[:60]}...")
            print(f"       실제: {r.actual_answer[:60]}...")

    print(f"\n{'='*70}")

    # ----------------------------------------------------------------------
    # Save JSON results
    # ----------------------------------------------------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
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
                    "model_label": model_label,
                    "mode": mode,
                },
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

    print(f"상세 결과 저장됨: {OUTPUT_PATH}")
    return results


# ---------------------------------------------------------------------------
# Main entry points per mode
# ---------------------------------------------------------------------------

async def run_local(model_name: str | None, limit: int | None):
    """Local RAGChain mode."""
    from config.settings import settings
    from core.llm import create_llm
    from rag.chain import RAGChain
    from rag.evaluator import AnswerEvaluator
    from rag.retriever import HybridRetriever

    print(f"\n{'='*70}")
    print("RAG 블라인드 평가 벤치마크 (LOCAL 모드)")
    print(f"{'='*70}")

    questions = load_dataset(limit=limit)
    total_count = len(questions)
    cat_counts: dict[str, int] = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    cat_summary = ", ".join(f"{c}({n})" for c, n in sorted(cat_counts.items()))
    print(f"총 질문 수: {total_count} ({cat_summary})")

    settings.default_llm_provider = "ollama"
    llm = create_llm(model=model_name) if model_name else create_llm()
    actual_model = model_name or settings.ollama_model
    print(f"사용 모델: {actual_model}")

    retriever = HybridRetriever()
    rag_chain = RAGChain(retriever=retriever, llm=llm)

    print("평가 엔진 초기화 중 (bge-m3 임베딩 로드)...")
    evaluator = AnswerEvaluator()
    print("평가 엔진 준비 완료.\n")

    async def query_fn(question: str):
        return await local_query(rag_chain, question)

    await run_benchmark_loop(
        questions=questions,
        query_fn=query_fn,
        evaluator=evaluator,
        model_label=actual_model,
        mode="local",
    )


async def run_api(
    api_url: str,
    api_user: str,
    api_password: str,
    limit: int | None,
):
    """HTTP API mode (for RunPod or any remote server)."""
    try:
        import aiohttp
    except ImportError:
        print("aiohttp가 설치되어 있지 않습니다. 설치: pip install aiohttp")
        sys.exit(1)

    from rag.evaluator import AnswerEvaluator

    print(f"\n{'='*70}")
    print("RAG 블라인드 평가 벤치마크 (API 모드)")
    print(f"{'='*70}")
    print(f"서버: {api_url}")
    print(f"사용자: {api_user}")

    questions = load_dataset(limit=limit)
    total_count = len(questions)
    cat_counts: dict[str, int] = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    cat_summary = ", ".join(f"{c}({n})" for c, n in sorted(cat_counts.items()))
    print(f"총 질문 수: {total_count} ({cat_summary})")

    print("평가 엔진 초기화 중 (bge-m3 임베딩 로드)...")
    evaluator = AnswerEvaluator()
    print("평가 엔진 준비 완료.")

    async with aiohttp.ClientSession() as session:
        # Login
        print(f"\n서버 로그인 중...")
        token = await api_login(session, api_url, api_user, api_password)
        print("로그인 성공.\n")

        async def query_fn(question: str):
            return await api_query(session, api_url, token, question)

        await run_benchmark_loop(
            questions=questions,
            query_fn=query_fn,
            evaluator=evaluator,
            model_label=api_url,
            mode="api",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG 블라인드 평가 벤치마크 (golden_dataset_evaluation.json)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "api"],
        help="실행 모드: local (RAGChain 직접) 또는 api (HTTP 원격 서버). 기본: local",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="[local 모드] Ollama 모델명 (예: qwen2.5:14b)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="평가할 최대 질문 수 (기본: 전체)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="[api 모드] 서버 기본 URL (예: https://xxx.proxy.runpod.net)",
    )
    parser.add_argument(
        "--api-user",
        type=str,
        default="admin",
        help="[api 모드] 로그인 사용자명 (기본: admin)",
    )
    parser.add_argument(
        "--api-password",
        type=str,
        default="admin123",
        help="[api 모드] 로그인 비밀번호 (기본: admin123)",
    )

    args = parser.parse_args()

    if args.mode == "api":
        if not args.api_url:
            parser.error("--api-url 이 필요합니다 (api 모드).")
        asyncio.run(
            run_api(
                api_url=args.api_url,
                api_user=args.api_user,
                api_password=args.api_password,
                limit=args.limit,
            )
        )
    else:
        asyncio.run(
            run_local(
                model_name=args.model,
                limit=args.limit,
            )
        )
