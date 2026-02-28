"""Parametric RAG benchmark across ALL 4 golden datasets.

Extends benchmark_optimizer.py to test parameter configurations against
internal_rules, brochure, travel, and alio datasets simultaneously.

Phases:
  speed    - Latency-focused configurations
  quality  - Quality-focused configurations
  advanced - Advanced RAG techniques (multi-query, HyDE, self-RAG)
  optimal  - Best-candidate configurations combining speed + quality
  all      - Run all phases sequentially

Usage:
  python tests/benchmark_parametric.py --phase speed
  python tests/benchmark_parametric.py --phase quality --sample 10
  python tests/benchmark_parametric.py --phase all --sample 0  # full dataset
  python tests/benchmark_parametric.py --phase speed --model qwen2.5:14b
"""

import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_all import (
    CATEGORY_THRESHOLDS,
    CONFIDENCE_THRESHOLDS,
    DEFAULT_CONFIDENCE,
    DEFAULT_THRESHOLD,
    DATASET_CONFIGS,
    load_dataset,
)
from benchmark_optimizer import ParamConfig, ConfigResult, test_config

from config.settings import settings
from core.llm import create_llm
from rag.evaluator import AnswerEvaluator

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

SPEED_TESTS: dict[str, ParamConfig] = {
    "speed_baseline": ParamConfig(
        name="speed_baseline",
        retrieval_top_k=20, rerank_top_n=5, max_tokens=2048,
    ),
    "speed_light": ParamConfig(
        name="speed_light",
        retrieval_top_k=10, rerank_top_n=3, max_tokens=1024,
    ),
    "speed_minimal": ParamConfig(
        name="speed_minimal",
        retrieval_top_k=5, rerank_top_n=3, max_tokens=512,
    ),
    "speed_no_rerank": ParamConfig(
        name="speed_no_rerank",
        retrieval_top_k=20, rerank_top_n=5, use_rerank=False, max_tokens=2048,
    ),
    "speed_fast_rerank": ParamConfig(
        name="speed_fast_rerank",
        retrieval_top_k=15, rerank_top_n=3, max_tokens=1024,
    ),
}

QUALITY_TESTS: dict[str, ParamConfig] = {
    "quality_baseline": ParamConfig(name="quality_baseline"),
    "quality_deep": ParamConfig(
        name="quality_deep",
        retrieval_top_k=30, rerank_top_n=7,
    ),
    "quality_semantic": ParamConfig(
        name="quality_semantic",
        vector_weight=0.8, bm25_weight=0.2,
    ),
    "quality_keyword": ParamConfig(
        name="quality_keyword",
        vector_weight=0.5, bm25_weight=0.5,
    ),
    "quality_temp_low": ParamConfig(
        name="quality_temp_low",
        temperature=0.1,
    ),
    "quality_temp_mid": ParamConfig(
        name="quality_temp_mid",
        temperature=0.3,
    ),
    "quality_temp_high": ParamConfig(
        name="quality_temp_high",
        temperature=0.5,
    ),
}

ADVANCED_TESTS: dict[str, ParamConfig] = {
    "adv_baseline": ParamConfig(name="adv_baseline"),
    "adv_multi_query": ParamConfig(
        name="adv_multi_query",
        multi_query=True, multi_query_count=3,
    ),
    "adv_hyde": ParamConfig(
        name="adv_hyde",
        query_expansion=True,
    ),
    "adv_self_rag": ParamConfig(
        name="adv_self_rag",
        self_rag=True,
    ),
    "adv_combined": ParamConfig(
        name="adv_combined",
        multi_query=True, query_expansion=True, self_rag=True, temperature=0.3,
    ),
}

OPTIMAL_CANDIDATES: dict[str, ParamConfig] = {
    "opt_current": ParamConfig(name="opt_current"),
    "opt_speed_quality": ParamConfig(
        name="opt_speed_quality",
        retrieval_top_k=15, rerank_top_n=5, temperature=0.3, max_tokens=1024,
    ),
    "opt_max_quality": ParamConfig(
        name="opt_max_quality",
        retrieval_top_k=30, rerank_top_n=7, temperature=0.1,
        vector_weight=0.8, bm25_weight=0.2,
    ),
    "opt_balanced": ParamConfig(
        name="opt_balanced",
        retrieval_top_k=20, rerank_top_n=5, temperature=0.3,
        max_tokens=1024, vector_weight=0.7, bm25_weight=0.3,
    ),
}

PHASE_MAP: dict[str, dict[str, ParamConfig]] = {
    "speed": SPEED_TESTS,
    "quality": QUALITY_TESTS,
    "advanced": ADVANCED_TESTS,
    "optimal": OPTIMAL_CANDIDATES,
}


# ---------------------------------------------------------------------------
# Representative sample selection
# ---------------------------------------------------------------------------

def select_representative_sample(
    all_questions: list[dict],
    per_dataset: int = 5,
) -> list[dict]:
    """Select a deterministic, balanced sample across datasets and categories.

    Target distribution for per_dataset=5 from each of 4 datasets (20 total):
      ~8 factual, ~5 inference, ~5 multi_hop, ~2 negative

    Uses deterministic selection (sorted by id, spread evenly) so the same
    questions are chosen every run for reproducibility.
    """
    if per_dataset <= 0:
        return all_questions

    # Group questions by dataset
    by_dataset: dict[str, list[dict]] = {}
    for q in all_questions:
        ds = q.get("_dataset", "unknown")
        by_dataset.setdefault(ds, [])
        by_dataset[ds].append(q)

    selected: list[dict] = []

    for ds_key in sorted(by_dataset.keys()):
        ds_questions = sorted(by_dataset[ds_key], key=lambda q: q["id"])
        n_pick = min(per_dataset, len(ds_questions))

        if n_pick >= len(ds_questions):
            selected.extend(ds_questions)
            continue

        # Group by category within this dataset
        by_cat: dict[str, list[dict]] = {}
        for q in ds_questions:
            cat = q.get("category", "unknown")
            by_cat.setdefault(cat, [])
            by_cat[cat].append(q)

        # Target category distribution for 5 picks:
        #   factual: 2, inference: 1, multi_hop: 1, negative: 1
        # Adjust if fewer categories are available
        target_dist = {
            "factual": 2,
            "inference": 1,
            "multi_hop": 1,
            "negative": 1,
        }

        # First pass: fill from target distribution
        ds_selected: list[dict] = []
        remaining_budget = n_pick

        for cat in ["factual", "inference", "multi_hop", "negative"]:
            cat_qs = by_cat.get(cat, [])
            if not cat_qs:
                continue
            target = min(target_dist.get(cat, 1), len(cat_qs), remaining_budget)
            if target <= 0:
                continue

            # Pick evenly spaced questions for difficulty balance
            # Sort by difficulty: easy -> medium -> hard
            diff_order = {"easy": 0, "medium": 1, "hard": 2}
            cat_qs_sorted = sorted(cat_qs, key=lambda q: (diff_order.get(q.get("difficulty", "medium"), 1), q["id"]))

            step = max(1, len(cat_qs_sorted) // target)
            picks = cat_qs_sorted[::step][:target]
            ds_selected.extend(picks)
            remaining_budget -= len(picks)

        # Second pass: fill remaining budget from any category
        if remaining_budget > 0:
            already_ids = {q["id"] for q in ds_selected}
            extras = [q for q in ds_questions if q["id"] not in already_ids]
            ds_selected.extend(extras[:remaining_budget])

        selected.extend(ds_selected[:n_pick])

    return selected


# ---------------------------------------------------------------------------
# Load all datasets with tagging
# ---------------------------------------------------------------------------

def load_all_questions() -> list[dict]:
    """Load and merge questions from all 4 golden datasets."""
    all_questions: list[dict] = []
    for key in sorted(DATASET_CONFIGS.keys()):
        name, questions = load_dataset(key)
        if questions:
            print(f"  {name}: {len(questions)} questions loaded")
            all_questions.extend(questions)
    return all_questions


# ---------------------------------------------------------------------------
# Per-question evaluation (with retrieval/generation timing)
# ---------------------------------------------------------------------------

async def evaluate_question(
    q: dict,
    rag_chain,
    evaluator: AnswerEvaluator,
    temperature: float | None = None,
) -> dict:
    """Evaluate a single question with timing breakdown.

    Returns a dict with evaluation metrics and timing info.
    """
    result = {
        "id": q["id"],
        "dataset": q.get("_dataset", "unknown"),
        "category": q.get("category", "unknown"),
        "difficulty": q.get("difficulty", "medium"),
        "question": q["question"],
        "success": False,
        "composite_score": 0.0,
        "semantic_similarity": 0.0,
        "rouge_l": 0.0,
        "confidence": 0.0,
        "grade": "F",
        "total_latency_ms": 0.0,
        "retrieval_ms": 0.0,
        "generation_ms": 0.0,
    }

    try:
        t_start = time.time()

        # Retrieval phase: time the retriever separately if accessible
        retriever = rag_chain.retriever
        t_ret_start = time.time()
        docs = await retriever.retrieve(q["question"])
        t_ret_end = time.time()
        result["retrieval_ms"] = round((t_ret_end - t_ret_start) * 1000, 1)

        # Generation phase
        t_gen_start = time.time()
        response = await rag_chain.query(q["question"], temperature=temperature)
        t_gen_end = time.time()
        # Generation time = total query time minus retrieval
        # (query internally also retrieves, so generation = query_time - retrieval overhead)
        total_query_ms = (t_gen_end - t_gen_start) * 1000
        result["generation_ms"] = round(total_query_ms - result["retrieval_ms"], 1)
        if result["generation_ms"] < 0:
            # If retrieval is done inside query(), just split total time
            result["generation_ms"] = round(total_query_ms * 0.6, 1)
            result["retrieval_ms"] = round(total_query_ms * 0.4, 1)

        result["total_latency_ms"] = round((t_gen_end - t_start) * 1000, 1)

        answer = response.content
        confidence = response.confidence
        result["confidence"] = confidence

        # Evaluate
        eval_result = await evaluator.evaluate(q["answer"], answer)
        cat_score = AnswerEvaluator.compute_category_score(eval_result, q["category"], expected=q["answer"], actual=answer)

        result["composite_score"] = eval_result.composite_score
        result["semantic_similarity"] = eval_result.semantic_similarity
        result["rouge_l"] = eval_result.rouge_l
        result["grade"] = eval_result.grade
        result["category_score"] = cat_score

        # Success criteria (same as benchmark_all.py)
        threshold = CATEGORY_THRESHOLDS.get(q["category"], DEFAULT_THRESHOLD)
        conf_threshold = CONFIDENCE_THRESHOLDS.get(q["category"], DEFAULT_CONFIDENCE)
        result["success"] = cat_score >= threshold and confidence > conf_threshold

    except Exception as e:
        result["total_latency_ms"] = round((time.time() - t_start) * 1000, 1)
        result["error"] = str(e)[:200]

    return result


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

async def run_phase(
    phase_name: str,
    configs: dict[str, ParamConfig],
    questions: list[dict],
    evaluator: AnswerEvaluator,
    llm,
    model_name: str,
) -> list[ConfigResult]:
    """Run a single benchmark phase against all configs."""

    config_list = list(configs.values())

    print(f"\n{'='*80}")
    print(f"  Phase: {phase_name.upper()}")
    print(f"  Configurations: {len(config_list)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Model: {model_name}")
    print(f"{'='*80}\n")

    # Show question distribution
    ds_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    for q in questions:
        ds = q.get("_dataset", "unknown")
        cat = q.get("category", "unknown")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("  Dataset distribution:")
    for ds_key in sorted(ds_counts.keys()):
        ds_name = DATASET_CONFIGS.get(ds_key, {}).get("name", ds_key)
        print(f"    {ds_name}: {ds_counts[ds_key]}")
    cat_str = ", ".join(f"{c}({n})" for c, n in sorted(cat_counts.items()))
    print(f"  Category distribution: {cat_str}")
    print()

    results: list[ConfigResult] = []
    phase_start = time.time()

    for i, config in enumerate(config_list, 1):
        print(f"  [{i}/{len(config_list)}] {config.name}", flush=True)
        params_str = (
            f"    k={config.retrieval_top_k} r={config.rerank_top_n} "
            f"t={config.temperature} v={config.vector_weight} b={config.bm25_weight} "
            f"tok={config.max_tokens} rerank={config.use_rerank}"
        )
        if config.multi_query:
            params_str += f" multi_q={config.multi_query_count}"
        if config.query_expansion:
            params_str += " hyde=on"
        if config.self_rag:
            params_str += " self_rag=on"
        print(params_str)

        result = await test_config(
            config=config,
            questions=questions,
            evaluator=evaluator,
            llm=llm,
            sample_indices=None,  # We already selected the sample
        )
        results.append(result)

        gd = result.grade_distribution
        print(
            f"    -> composite={result.avg_composite:.4f} "
            f"semantic={result.avg_semantic:.4f} "
            f"rouge_l={result.avg_rouge_l:.4f} "
            f"success={result.success_rate:.1%} "
            f"latency={result.avg_latency_ms:.0f}ms "
            f"[A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} "
            f"D:{gd.get('D',0)} F:{gd.get('F',0)}]"
        )
        print()

    phase_time = time.time() - phase_start

    # ---- Results table ----
    print(f"\n{'='*80}")
    print(f"  Phase {phase_name.upper()} Results (sorted by composite score)")
    print(f"{'='*80}\n")

    ranked = sorted(results, key=lambda r: (r.avg_composite, r.success_rate), reverse=True)

    header = (
        f"{'Rank':>4s}  {'Config':24s}  {'Composite':>9s}  {'Semantic':>8s}  "
        f"{'ROUGE-L':>7s}  {'Success':>7s}  {'Latency':>8s}  {'Grades':20s}"
    )
    print(header)
    print(f"{'-'*4}  {'-'*24}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*20}")

    for rank, r in enumerate(ranked, 1):
        gd = r.grade_distribution
        grade_str = (
            f"A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} "
            f"D:{gd.get('D',0)} F:{gd.get('F',0)}"
        )
        marker = " *" if rank == 1 else ""
        print(
            f"{rank:4d}  {r.config.name:24s}  {r.avg_composite:9.4f}  "
            f"{r.avg_semantic:8.4f}  {r.avg_rouge_l:7.4f}  "
            f"{r.success_rate:6.1%}  {r.avg_latency_ms:7.0f}ms  "
            f"{grade_str}{marker}"
        )

    # ---- Best config detail ----
    best = ranked[0]
    baseline = ranked[-1]  # often the baseline is last alphabetically, but use worst as comparison
    # Find actual baseline (name containing "baseline" or "current")
    baseline_result = next(
        (r for r in ranked if "baseline" in r.config.name or "current" in r.config.name),
        ranked[-1],
    )

    print(f"\n  Best configuration: {best.config.name}")
    print(f"    composite={best.avg_composite:.4f}  semantic={best.avg_semantic:.4f}  "
          f"rouge_l={best.avg_rouge_l:.4f}  success={best.success_rate:.1%}  "
          f"latency={best.avg_latency_ms:.0f}ms")

    if baseline_result.config.name != best.config.name:
        delta_comp = best.avg_composite - baseline_result.avg_composite
        delta_lat = best.avg_latency_ms - baseline_result.avg_latency_ms
        delta_succ = best.success_rate - baseline_result.success_rate
        sign_comp = "+" if delta_comp >= 0 else ""
        sign_lat = "+" if delta_lat >= 0 else ""
        sign_succ = "+" if delta_succ >= 0 else ""
        print(f"\n  vs baseline ({baseline_result.config.name}):")
        print(f"    composite: {sign_comp}{delta_comp:.4f}")
        print(f"    success:   {sign_succ}{delta_succ:.1%}")
        print(f"    latency:   {sign_lat}{delta_lat:.0f}ms")

    print(f"\n  Phase time: {phase_time:.1f}s")
    print(f"{'='*80}\n")

    return ranked


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    phase_name: str,
    ranked: list[ConfigResult],
    questions: list[dict],
    model_name: str,
    total_time: float,
) -> Path:
    """Save phase results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "data"
    output_path = output_dir / f"parametric_results_{phase_name}_{timestamp}.json"

    # Build question summary
    ds_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    for q in questions:
        ds = q.get("_dataset", "unknown")
        cat = q.get("category", "unknown")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    best = ranked[0] if ranked else None

    data = {
        "meta": {
            "phase": phase_name,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "total_configs": len(ranked),
            "total_questions": len(questions),
            "total_time_sec": round(total_time, 1),
            "dataset_distribution": ds_counts,
            "category_distribution": cat_counts,
        },
        "best_config": {
            "name": best.config.name,
            "avg_composite": best.avg_composite,
            "avg_semantic": best.avg_semantic,
            "avg_rouge_l": best.avg_rouge_l,
            "success_rate": best.success_rate,
            "avg_latency_ms": best.avg_latency_ms,
            "params": asdict(best.config),
        } if best else None,
        "rankings": [
            {
                "rank": rank,
                "name": r.config.name,
                "avg_composite": r.avg_composite,
                "avg_semantic": r.avg_semantic,
                "avg_rouge_l": r.avg_rouge_l,
                "avg_confidence": r.avg_confidence,
                "success_rate": r.success_rate,
                "avg_latency_ms": r.avg_latency_ms,
                "grade_distribution": r.grade_distribution,
                "total_questions": r.total_questions,
                "total_time_sec": r.total_time_sec,
                "params": asdict(r.config),
            }
            for rank, r in enumerate(ranked, 1)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_parametric(
    phase: str = "speed",
    sample_per_dataset: int = 5,
    model_name: str | None = None,
):
    """Run parametric benchmark.

    Args:
        phase: Which phase to run: speed, quality, advanced, optimal, all.
        sample_per_dataset: Questions per dataset (0 = use all).
        model_name: Override LLM model name.
    """

    print(f"\n{'#'*80}")
    print(f"  Parametric RAG Benchmark")
    print(f"  Phase: {phase}")
    sample_total = sample_per_dataset * 4 if sample_per_dataset > 0 else "all"
    print(f"  Sample: {sample_per_dataset}/dataset ({sample_total} total)")
    print(f"{'#'*80}\n")

    # ---- Load questions ----
    print("Loading golden datasets...")
    all_questions = load_all_questions()
    if not all_questions:
        print("No questions loaded. Exiting.")
        return

    total_available = len(all_questions)
    print(f"Total available: {total_available} questions\n")

    # ---- Select sample ----
    if sample_per_dataset > 0:
        questions = select_representative_sample(all_questions, per_dataset=sample_per_dataset)
        print(f"Selected representative sample: {len(questions)} questions")
        # Show what was selected
        for q in questions:
            ds_name = DATASET_CONFIGS.get(q.get("_dataset", ""), {}).get("name", q.get("_dataset", "?"))
            print(f"  [{ds_name}] Q{q['id']} ({q['category']}/{q['difficulty']}): {q['question'][:50]}...")
        print()
    else:
        questions = all_questions
        print(f"Using full dataset: {len(questions)} questions\n")

    # ---- Initialize LLM + Evaluator ----
    settings.default_llm_provider = "ollama"
    llm = create_llm(model=model_name) if model_name else create_llm()
    actual_model = model_name or settings.ollama_model
    print(f"Model: {actual_model}")

    print("Initializing evaluator (bge-m3 embeddings)...")
    evaluator = AnswerEvaluator()
    print("Evaluator ready.\n")

    # ---- Determine which phases to run ----
    if phase == "all":
        phases_to_run = ["speed", "quality", "advanced", "optimal"]
    else:
        phases_to_run = [phase]

    # ---- Run phases ----
    all_phase_results: dict[str, list[ConfigResult]] = {}
    grand_start = time.time()

    for phase_key in phases_to_run:
        if phase_key not in PHASE_MAP:
            print(f"Unknown phase: {phase_key}. Skipping.")
            continue

        configs = PHASE_MAP[phase_key]
        phase_start = time.time()

        ranked = await run_phase(
            phase_name=phase_key,
            configs=configs,
            questions=questions,
            evaluator=evaluator,
            llm=llm,
            model_name=actual_model,
        )

        phase_time = time.time() - phase_start
        all_phase_results[phase_key] = ranked

        # Save phase results
        output_path = save_results(
            phase_name=phase_key,
            ranked=ranked,
            questions=questions,
            model_name=actual_model,
            total_time=phase_time,
        )
        print(f"  Results saved: {output_path}\n")

    grand_time = time.time() - grand_start

    # ---- Grand summary (if multiple phases) ----
    if len(all_phase_results) > 1:
        print(f"\n{'#'*80}")
        print(f"  GRAND SUMMARY - All Phases")
        print(f"{'#'*80}\n")

        # Collect all best results
        all_bests: list[tuple[str, ConfigResult]] = []
        for pkey, ranked in all_phase_results.items():
            if ranked:
                all_bests.append((pkey, ranked[0]))

        print(f"{'Phase':>10s}  {'Best Config':24s}  {'Composite':>9s}  "
              f"{'Success':>7s}  {'Latency':>8s}")
        print(f"{'-'*10}  {'-'*24}  {'-'*9}  {'-'*7}  {'-'*8}")

        for pkey, best in all_bests:
            print(
                f"{pkey:>10s}  {best.config.name:24s}  "
                f"{best.avg_composite:9.4f}  {best.success_rate:6.1%}  "
                f"{best.avg_latency_ms:7.0f}ms"
            )

        # Overall best across all phases
        overall_best_phase, overall_best = max(
            all_bests, key=lambda x: (x[1].avg_composite, x[1].success_rate)
        )
        print(f"\n  Overall best: {overall_best.config.name} (phase: {overall_best_phase})")
        print(f"    composite={overall_best.avg_composite:.4f} "
              f"success={overall_best.success_rate:.1%} "
              f"latency={overall_best.avg_latency_ms:.0f}ms")

        # Recommended settings
        bc = overall_best.config
        print(f"\n  Recommended .env settings:")
        print(f"    VECTOR_WEIGHT={bc.vector_weight}")
        print(f"    BM25_WEIGHT={bc.bm25_weight}")
        print(f"    RETRIEVAL_TOP_K={bc.retrieval_top_k}")
        print(f"    RERANK_TOP_N={bc.rerank_top_n}")
        print(f"    LLM_MAX_TOKENS={bc.max_tokens}")
        print(f"    USE_RERANK={str(bc.use_rerank).lower()}")
        print(f"    QUERY_EXPANSION_ENABLED={str(bc.query_expansion).lower()}")
        print(f"    MULTI_QUERY_ENABLED={str(bc.multi_query).lower()}")
        print(f"    SELF_RAG_ENABLED={str(bc.self_rag).lower()}")

    print(f"\nTotal benchmark time: {grand_time:.1f}s")
    print(f"{'#'*80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parametric RAG benchmark across all 4 golden datasets"
    )
    parser.add_argument(
        "--phase", "-p",
        type=str,
        default="speed",
        choices=["speed", "quality", "advanced", "optimal", "all"],
        help="Which phase to run (default: speed)",
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=5,
        help="Questions per dataset (0 = full dataset, default: 5 => 20 total)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ollama model name override (e.g. qwen2.5:14b)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_parametric(
            phase=args.phase,
            sample_per_dataset=args.sample,
            model_name=args.model,
        )
    )
