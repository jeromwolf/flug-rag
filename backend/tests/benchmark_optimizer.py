"""RAG 파라미터 최적화 벤치마크.

다양한 RAG 설정 조합을 테스트하여 최적 구성을 찾아낸다.
테스트 항목: vector/bm25 비중, retrieval_top_k, rerank_top_n, chunk_size, temperature 등.
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from core.llm import create_llm
from rag.chain import RAGChain
from rag.evaluator import AnswerEvaluation, AnswerEvaluator
from rag.prompt import PromptManager
from rag.quality import QualityController
from rag.retriever import HybridRetriever


@dataclass
class ParamConfig:
    """A single parameter configuration to test."""
    name: str
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    temperature: float = 0.7
    chunk_size: int = 800
    chunk_overlap: int = 80
    # New tuning parameters
    score_threshold: float = 0.0
    context_max_chunks: int = 0
    max_tokens: int = 2048
    use_rerank: bool = True
    query_expansion: bool = False  # HyDE
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # Advanced RAG techniques
    multi_query: bool = False
    multi_query_count: int = 3
    self_rag: bool = False
    agentic_rag: bool = False


@dataclass
class ConfigResult:
    """Result of testing a single configuration."""
    config: ParamConfig
    avg_composite: float = 0.0
    avg_semantic: float = 0.0
    avg_rouge_l: float = 0.0
    avg_confidence: float = 0.0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    grade_distribution: dict = field(default_factory=dict)
    total_questions: int = 0
    total_time_sec: float = 0.0


# ====== Parameter Grid Definition ======

# Predefined parameter grids for different optimization targets
PARAM_GRIDS = {
    # Quick test: retrieval weights only (4 combos)
    "weights": [
        ParamConfig(name="v0.5_b0.5", vector_weight=0.5, bm25_weight=0.5),
        ParamConfig(name="v0.6_b0.4", vector_weight=0.6, bm25_weight=0.4),
        ParamConfig(name="v0.7_b0.3", vector_weight=0.7, bm25_weight=0.3),
        ParamConfig(name="v0.8_b0.2", vector_weight=0.8, bm25_weight=0.2),
    ],

    # Retrieval depth (4 combos)
    "retrieval": [
        ParamConfig(name="k10_r3", retrieval_top_k=10, rerank_top_n=3),
        ParamConfig(name="k15_r5", retrieval_top_k=15, rerank_top_n=5),
        ParamConfig(name="k20_r5", retrieval_top_k=20, rerank_top_n=5),
        ParamConfig(name="k30_r7", retrieval_top_k=30, rerank_top_n=7),
    ],

    # Temperature (4 combos)
    "temperature": [
        ParamConfig(name="temp0.1", temperature=0.1),
        ParamConfig(name="temp0.3", temperature=0.3),
        ParamConfig(name="temp0.5", temperature=0.5),
        ParamConfig(name="temp0.7", temperature=0.7),
    ],

    # Combined optimization (top candidates from each category)
    "combined": [
        ParamConfig(name="baseline", vector_weight=0.7, bm25_weight=0.3, retrieval_top_k=20, rerank_top_n=5, temperature=0.7),
        ParamConfig(name="semantic_heavy", vector_weight=0.8, bm25_weight=0.2, retrieval_top_k=20, rerank_top_n=5, temperature=0.3),
        ParamConfig(name="balanced", vector_weight=0.6, bm25_weight=0.4, retrieval_top_k=20, rerank_top_n=5, temperature=0.3),
        ParamConfig(name="deep_retrieval", vector_weight=0.7, bm25_weight=0.3, retrieval_top_k=30, rerank_top_n=7, temperature=0.3),
        ParamConfig(name="precise", vector_weight=0.8, bm25_weight=0.2, retrieval_top_k=15, rerank_top_n=3, temperature=0.1),
        ParamConfig(name="wide_retrieval", vector_weight=0.5, bm25_weight=0.5, retrieval_top_k=30, rerank_top_n=7, temperature=0.5),
    ],

    # Score threshold (4 combos)
    "threshold": [
        ParamConfig(name="no_threshold", score_threshold=0.0),
        ParamConfig(name="thresh_0.1", score_threshold=0.1),
        ParamConfig(name="thresh_0.3", score_threshold=0.3),
        ParamConfig(name="thresh_0.5", score_threshold=0.5),
    ],

    # Context window size (4 combos)
    "context": [
        ParamConfig(name="ctx_3", context_max_chunks=3, rerank_top_n=5),
        ParamConfig(name="ctx_5", context_max_chunks=5, rerank_top_n=7),
        ParamConfig(name="ctx_7", context_max_chunks=7, rerank_top_n=10),
        ParamConfig(name="ctx_all", context_max_chunks=0, rerank_top_n=5),
    ],

    # Max tokens (4 combos)
    "tokens": [
        ParamConfig(name="tok_512", max_tokens=512),
        ParamConfig(name="tok_1024", max_tokens=1024),
        ParamConfig(name="tok_2048", max_tokens=2048),
        ParamConfig(name="tok_4096", max_tokens=4096),
    ],

    # HyDE query expansion (2 combos)
    "hyde": [
        ParamConfig(name="no_hyde", query_expansion=False),
        ParamConfig(name="with_hyde", query_expansion=True),
    ],

    # BM25 parameters (4 combos)
    "bm25": [
        ParamConfig(name="bm25_default", bm25_k1=1.5, bm25_b=0.75),
        ParamConfig(name="bm25_low_k1", bm25_k1=1.0, bm25_b=0.75),
        ParamConfig(name="bm25_high_k1", bm25_k1=2.0, bm25_b=0.75),
        ParamConfig(name="bm25_low_b", bm25_k1=1.5, bm25_b=0.5),
    ],

    # Reranking (2 combos)
    "rerank": [
        ParamConfig(name="with_rerank", use_rerank=True),
        ParamConfig(name="no_rerank", use_rerank=False),
    ],

    # Multi-query (2 combos)
    "multi_query": [
        ParamConfig(name="no_multi_query", multi_query=False),
        ParamConfig(name="with_multi_query", multi_query=True, multi_query_count=3),
    ],

    # Self-RAG (2 combos)
    "self_rag": [
        ParamConfig(name="no_self_rag", self_rag=False),
        ParamConfig(name="with_self_rag", self_rag=True),
    ],

    # Agentic RAG (2 combos)
    "agentic": [
        ParamConfig(name="no_agentic", agentic_rag=False),
        ParamConfig(name="with_agentic", agentic_rag=True),
    ],

    # All advanced techniques combined (4 combos)
    "advanced": [
        ParamConfig(name="baseline", multi_query=False, self_rag=False, query_expansion=False),
        ParamConfig(name="hyde_only", query_expansion=True),
        ParamConfig(name="multi_query_only", multi_query=True),
        ParamConfig(name="all_advanced", multi_query=True, self_rag=True, query_expansion=True, temperature=0.3),
    ],

    # Full grid search (generates many combos - use with sample_size)
    "full": [],  # Generated dynamically
}


def generate_full_grid() -> list[ParamConfig]:
    """Generate full grid search parameter combinations."""
    configs = []
    weights = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    top_ks = [15, 20, 30]
    rerank_ns = [3, 5, 7]
    temps = [0.1, 0.3, 0.5]

    for (vw, bw), k, r, t in product(weights, top_ks, rerank_ns, temps):
        if r >= k:  # rerank_top_n must be < retrieval_top_k
            continue
        name = f"v{vw}_b{bw}_k{k}_r{r}_t{t}"
        configs.append(ParamConfig(
            name=name,
            vector_weight=vw,
            bm25_weight=bw,
            retrieval_top_k=k,
            rerank_top_n=r,
            temperature=t,
        ))
    return configs


async def test_config(
    config: ParamConfig,
    questions: list[dict],
    evaluator: AnswerEvaluator,
    llm,
    sample_indices: list[int] | None = None,
) -> ConfigResult:
    """Test a single configuration against the question set."""

    # Apply config to retriever
    retriever = HybridRetriever(
        vector_weight=config.vector_weight,
        bm25_weight=config.bm25_weight,
        top_k=config.retrieval_top_k,
        rerank_top_n=config.rerank_top_n,
        score_threshold=config.score_threshold,
        use_rerank=config.use_rerank,
        bm25_k1=config.bm25_k1,
        bm25_b=config.bm25_b,
    )

    # Temporarily apply settings for this config
    import config.settings as cfg_module
    original_context_max = cfg_module.settings.context_max_chunks
    original_max_tokens = cfg_module.settings.llm_max_tokens
    original_hyde = cfg_module.settings.query_expansion_enabled
    original_multi_query = cfg_module.settings.multi_query_enabled
    original_multi_query_count = cfg_module.settings.multi_query_count
    original_self_rag = cfg_module.settings.self_rag_enabled
    original_agentic = cfg_module.settings.agentic_rag_enabled
    cfg_module.settings.context_max_chunks = config.context_max_chunks
    cfg_module.settings.llm_max_tokens = config.max_tokens
    cfg_module.settings.query_expansion_enabled = config.query_expansion
    cfg_module.settings.multi_query_enabled = config.multi_query
    cfg_module.settings.multi_query_count = config.multi_query_count
    cfg_module.settings.self_rag_enabled = config.self_rag
    cfg_module.settings.agentic_rag_enabled = config.agentic_rag

    try:
        rag_chain = RAGChain(retriever=retriever, llm=llm)

        # Select questions to test
        if sample_indices is not None:
            test_questions = [questions[i] for i in sample_indices if i < len(questions)]
        else:
            test_questions = questions

        scores = []
        sims = []
        rouges = []
        confs = []
        latencies = []
        successes = 0
        grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

        start_time = time.time()

        for q in test_questions:
            try:
                t0 = time.time()
                response = await rag_chain.query(
                    q["question"],
                    temperature=config.temperature,
                )
                latency = (time.time() - t0) * 1000

                answer = response.content
                confidence = response.confidence

                eval_result = await evaluator.evaluate(q["answer"], answer)

                scores.append(eval_result.composite_score)
                sims.append(eval_result.semantic_similarity)
                rouges.append(eval_result.rouge_l)
                confs.append(confidence)
                latencies.append(latency)
                grades[eval_result.grade] = grades.get(eval_result.grade, 0) + 1

                if eval_result.composite_score >= 0.5 and confidence > 0.3:
                    successes += 1

            except Exception as e:
                scores.append(0.0)
                sims.append(0.0)
                rouges.append(0.0)
                confs.append(0.0)
                latencies.append(0.0)
                grades["F"] += 1

        total_time = time.time() - start_time
        n = len(test_questions)

        return ConfigResult(
            config=config,
            avg_composite=round(sum(scores) / n, 4) if n else 0,
            avg_semantic=round(sum(sims) / n, 4) if n else 0,
            avg_rouge_l=round(sum(rouges) / n, 4) if n else 0,
            avg_confidence=round(sum(confs) / n, 4) if n else 0,
            success_rate=round(successes / n, 4) if n else 0,
            avg_latency_ms=round(sum(latencies) / n, 1) if n else 0,
            grade_distribution=grades,
            total_questions=n,
            total_time_sec=round(total_time, 1),
        )
    finally:
        # Restore original settings
        cfg_module.settings.context_max_chunks = original_context_max
        cfg_module.settings.llm_max_tokens = original_max_tokens
        cfg_module.settings.query_expansion_enabled = original_hyde
        cfg_module.settings.multi_query_enabled = original_multi_query
        cfg_module.settings.multi_query_count = original_multi_query_count
        cfg_module.settings.self_rag_enabled = original_self_rag
        cfg_module.settings.agentic_rag_enabled = original_agentic


def select_sample_indices(questions: list[dict], sample_size: int) -> list[int]:
    """Select a representative sample of question indices.

    Ensures proportional representation from each category and difficulty.
    """
    if sample_size >= len(questions):
        return list(range(len(questions)))

    # Group by category
    category_indices: dict[str, list[int]] = {}
    for i, q in enumerate(questions):
        cat = q.get("category", "unknown")
        category_indices.setdefault(cat, []).append(i)

    # Take proportional samples from each category
    selected = []
    for cat, indices in category_indices.items():
        n_take = max(1, round(sample_size * len(indices) / len(questions)))
        # Spread evenly across the category
        step = max(1, len(indices) // n_take)
        selected.extend(indices[::step][:n_take])

    # Trim to exact sample_size
    return sorted(selected[:sample_size])


async def run_optimizer(
    grid_name: str = "combined",
    sample_size: int = 15,
):
    """Run parameter optimization benchmark.

    Args:
        grid_name: Which parameter grid to test.
                   Options: "weights", "retrieval", "temperature", "combined", "full"
        sample_size: Number of questions to test per config (0 = all 50).
    """
    # Load golden dataset
    dataset_path = (
        Path(__file__).parent.parent
        / "data/sample_dataset/한국가스공사법/golden_dataset_kogas_law.json"
    )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]

    # Get parameter grid
    if grid_name == "full":
        configs = generate_full_grid()
    elif grid_name in PARAM_GRIDS:
        configs = PARAM_GRIDS[grid_name]
    else:
        print(f"Unknown grid: {grid_name}. Available: {list(PARAM_GRIDS.keys())}")
        return

    print(f"\n{'='*70}")
    print(f"RAG 파라미터 최적화 벤치마크")
    print(f"{'='*70}")
    print(f"파라미터 그리드: {grid_name} ({len(configs)}개 조합)")
    print(f"테스트 질문 수: {sample_size if sample_size > 0 else len(questions)}개 / 전체 {len(questions)}개")
    print(f"{'='*70}\n")

    # Initialize shared components
    settings.default_llm_provider = "ollama"
    llm = create_llm()

    print("평가 엔진 초기화 중...")
    evaluator = AnswerEvaluator()
    print("평가 엔진 준비 완료.\n")

    # Select sample indices
    sample_indices = None
    if 0 < sample_size < len(questions):
        sample_indices = select_sample_indices(questions, sample_size)
        print(f"샘플 질문 인덱스: {sample_indices}\n")

    # Test each configuration
    results: list[ConfigResult] = []
    total_start = time.time()

    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] 테스트: {config.name}", end=" ", flush=True)
        print(f"(v={config.vector_weight} b={config.bm25_weight} "
              f"k={config.retrieval_top_k} r={config.rerank_top_n} "
              f"t={config.temperature})")

        result = await test_config(config, questions, evaluator, llm, sample_indices)
        results.append(result)

        gd = result.grade_distribution
        print(f"  -> score={result.avg_composite:.3f} sim={result.avg_semantic:.3f} "
              f"rouge_l={result.avg_rouge_l:.3f} success={result.success_rate:.1%} "
              f"latency={result.avg_latency_ms:.0f}ms "
              f"[A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} "
              f"D:{gd.get('D',0)} F:{gd.get('F',0)}]")
        print()

    total_time = time.time() - total_start

    # ====== Rankings ======
    print(f"\n{'='*70}")
    print(f"최적화 결과 (종합 점수 기준 정렬)")
    print(f"{'='*70}\n")

    # Sort by composite score (primary), then by success rate (secondary)
    ranked = sorted(results, key=lambda r: (r.avg_composite, r.success_rate), reverse=True)

    print(f"{'순위':>4s}  {'구성':20s}  {'종합':>6s}  {'유사도':>6s}  {'ROUGE-L':>7s}  "
          f"{'성공률':>6s}  {'지연(ms)':>8s}  {'등급분포':20s}")
    print(f"{'-'*4}  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*20}")

    for rank, r in enumerate(ranked, 1):
        gd = r.grade_distribution
        grade_str = f"A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} D:{gd.get('D',0)} F:{gd.get('F',0)}"
        marker = " *" if rank == 1 else ""
        print(f"{rank:4d}  {r.config.name:20s}  {r.avg_composite:6.4f}  {r.avg_semantic:6.4f}  "
              f"{r.avg_rouge_l:7.4f}  {r.success_rate:5.1%}  {r.avg_latency_ms:8.0f}  "
              f"{grade_str}{marker}")

    # Best configuration
    best = ranked[0]
    print(f"\n{'='*70}")
    print(f"* 최적 구성: {best.config.name}")
    print(f"{'='*70}")
    print(f"  vector_weight:    {best.config.vector_weight}")
    print(f"  bm25_weight:      {best.config.bm25_weight}")
    print(f"  retrieval_top_k:  {best.config.retrieval_top_k}")
    print(f"  rerank_top_n:     {best.config.rerank_top_n}")
    print(f"  temperature:      {best.config.temperature}")
    print(f"  score_threshold:  {best.config.score_threshold}")
    print(f"  context_max:      {best.config.context_max_chunks}")
    print(f"  max_tokens:       {best.config.max_tokens}")
    print(f"  use_rerank:       {best.config.use_rerank}")
    print(f"  query_expansion:  {best.config.query_expansion}")
    print(f"  bm25_k1:          {best.config.bm25_k1}")
    print(f"  bm25_b:           {best.config.bm25_b}")
    print(f"  multi_query:      {best.config.multi_query}")
    print(f"  self_rag:         {best.config.self_rag}")
    print(f"  agentic_rag:      {best.config.agentic_rag}")
    print(f"  ---")
    print(f"  종합 점수:     {best.avg_composite:.4f}")
    print(f"  의미 유사도:   {best.avg_semantic:.4f}")
    print(f"  ROUGE-L:       {best.avg_rouge_l:.4f}")
    print(f"  성공률:        {best.success_rate:.1%}")
    print(f"  평균 지연:     {best.avg_latency_ms:.0f}ms")

    # Recommend .env settings
    print(f"\n[추천 .env 설정]")
    print(f"  VECTOR_WEIGHT={best.config.vector_weight}")
    print(f"  BM25_WEIGHT={best.config.bm25_weight}")
    print(f"  RETRIEVAL_TOP_K={best.config.retrieval_top_k}")
    print(f"  RERANK_TOP_N={best.config.rerank_top_n}")
    print(f"  RETRIEVAL_SCORE_THRESHOLD={best.config.score_threshold}")
    print(f"  CONTEXT_MAX_CHUNKS={best.config.context_max_chunks}")
    print(f"  LLM_MAX_TOKENS={best.config.max_tokens}")
    print(f"  USE_RERANK={str(best.config.use_rerank).lower()}")
    print(f"  QUERY_EXPANSION_ENABLED={str(best.config.query_expansion).lower()}")
    print(f"  BM25_K1={best.config.bm25_k1}")
    print(f"  BM25_B={best.config.bm25_b}")
    print(f"  MULTI_QUERY_ENABLED={str(best.config.multi_query).lower()}")
    print(f"  SELF_RAG_ENABLED={str(best.config.self_rag).lower()}")
    print(f"  AGENTIC_RAG_ENABLED={str(best.config.agentic_rag).lower()}")

    print(f"\n총 소요 시간: {total_time:.1f}초")
    print(f"{'='*70}")

    # Save results to JSON
    output_path = (
        Path(__file__).parent.parent
        / "data/sample_dataset/한국가스공사법/optimizer_results.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "grid_name": grid_name,
                "sample_size": sample_size,
                "total_configs": len(configs),
                "total_time_sec": round(total_time, 1),
                "best_config": {
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
                    "avg_composite": best.avg_composite,
                    "avg_semantic": best.avg_semantic,
                    "avg_rouge_l": best.avg_rouge_l,
                    "success_rate": best.success_rate,
                },
                "rankings": [
                    {
                        "rank": rank,
                        "name": r.config.name,
                        "vector_weight": r.config.vector_weight,
                        "bm25_weight": r.config.bm25_weight,
                        "retrieval_top_k": r.config.retrieval_top_k,
                        "rerank_top_n": r.config.rerank_top_n,
                        "temperature": r.config.temperature,
                        "score_threshold": r.config.score_threshold,
                        "context_max_chunks": r.config.context_max_chunks,
                        "max_tokens": r.config.max_tokens,
                        "use_rerank": r.config.use_rerank,
                        "query_expansion": r.config.query_expansion,
                        "bm25_k1": r.config.bm25_k1,
                        "bm25_b": r.config.bm25_b,
                        "multi_query": r.config.multi_query,
                        "self_rag": r.config.self_rag,
                        "agentic_rag": r.config.agentic_rag,
                        "avg_composite": r.avg_composite,
                        "avg_semantic": r.avg_semantic,
                        "avg_rouge_l": r.avg_rouge_l,
                        "success_rate": r.success_rate,
                        "avg_latency_ms": r.avg_latency_ms,
                        "grade_distribution": r.grade_distribution,
                        "total_questions": r.total_questions,
                        "total_time_sec": r.total_time_sec,
                    }
                    for rank, r in enumerate(ranked, 1)
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"상세 결과 저장됨: {output_path}")

    return ranked


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 파라미터 최적화 벤치마크")
    parser.add_argument(
        "--grid", "-g",
        default="combined",
        choices=["weights", "retrieval", "temperature", "combined", "threshold", "context", "tokens", "hyde", "bm25", "rerank", "multi_query", "self_rag", "agentic", "advanced", "full"],
        help="파라미터 그리드 선택 (default: combined)",
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=15,
        help="구성당 테스트 질문 수 (0=전체 50개, default: 15)",
    )
    args = parser.parse_args()

    asyncio.run(run_optimizer(grid_name=args.grid, sample_size=args.sample))
