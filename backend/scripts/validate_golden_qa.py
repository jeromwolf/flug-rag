"""Golden QA validation script.

Loads golden_qa.json and validates that the RAG pipeline retrieves
the correct source documents for each question. Optionally runs full
RAG generation if an LLM is available.

Usage:
    python scripts/validate_golden_qa.py                  # retrieval-only mode
    python scripts/validate_golden_qa.py --full            # full RAG mode (needs LLM)
    python scripts/validate_golden_qa.py --difficulty easy  # filter by difficulty
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- allow imports from the backend package root
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.embeddings import create_embedder
from core.vectorstore import create_vectorstore
from pipeline.ingest import IngestPipeline
from rag.retriever import HybridRetriever

GOLDEN_QA_PATH = BACKEND_DIR / "data" / "golden_qa.json"
SAMPLE_DIR = BACKEND_DIR / "data" / "sample"
VALIDATION_COLLECTION = "validation_test"
VALIDATION_CHROMA_DIR = str(BACKEND_DIR / "data" / "chroma_validation")


@dataclass
class ValidationResult:
    """Result for a single QA validation."""
    qa_id: int
    question: str
    difficulty: str
    category: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    retrieval_recall: float
    retrieval_correct: bool
    latency_ms: int


@dataclass
class ValidationReport:
    """Aggregate validation report."""
    total: int = 0
    correct_retrievals: int = 0
    accuracy: float = 0.0
    avg_recall: float = 0.0
    avg_latency_ms: float = 0.0
    results: list[ValidationResult] = field(default_factory=list)
    by_difficulty: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)


def compute_retrieval_recall(
    expected_sources: list[str],
    retrieved_sources: list[str],
) -> float:
    """Compute recall: fraction of expected source docs found in retrieval."""
    if not expected_sources:
        return 1.0

    found = 0
    for expected in expected_sources:
        expected_stem = Path(expected).stem
        for retrieved in retrieved_sources:
            if expected_stem in retrieved or expected in retrieved:
                found += 1
                break

    return found / len(expected_sources)


async def run_validation(
    difficulty_filter: str | None = None,
    full_mode: bool = False,
) -> ValidationReport:
    """Run golden QA validation."""
    sep = "=" * 72

    # Load golden QA
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    if difficulty_filter:
        golden_qa = [qa for qa in golden_qa if qa["difficulty"] == difficulty_filter]

    print(f"\n{sep}")
    print("  Golden QA Validation")
    print(sep)
    print(f"  Total QA pairs: {len(golden_qa)}")
    if difficulty_filter:
        print(f"  Filter: difficulty={difficulty_filter}")
    print()

    # Create infrastructure
    print("  Setting up vector store and embedder...")
    vectorstore = create_vectorstore(
        collection_name=VALIDATION_COLLECTION,
        persist_dir=VALIDATION_CHROMA_DIR,
    )
    embedder = create_embedder()

    # Clear and ingest all sample documents
    print("  Clearing previous validation data...")
    await vectorstore.clear()

    print("  Ingesting sample documents...")
    pipeline = IngestPipeline(vectorstore=vectorstore, embedder=embedder)
    sample_files = sorted(SAMPLE_DIR.glob("*.txt"))
    print(f"  Found {len(sample_files)} sample documents")

    for fpath in sample_files:
        result = await pipeline.ingest(file_path=fpath)
        status = "OK" if result.status == "completed" else "FAIL"
        print(f"    [{status}] {fpath.name} -> {result.chunk_count} chunks")

    # Create retriever
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        embedder=embedder,
    )

    # Optionally create RAG chain for full mode
    rag_chain = None
    if full_mode:
        try:
            from rag import RAGChain
            rag_chain = RAGChain(retriever=retriever)
            print("  Full RAG mode: LLM connected")
        except Exception as e:
            print(f"  WARNING: Could not initialize RAG chain: {e}")
            print("  Falling back to retrieval-only mode")

    # Run validation
    print(f"\n  Running validation on {len(golden_qa)} QA pairs...")
    print(f"  {'ID':>4} {'Diff':>6} {'Category':>16} {'Recall':>7} {'Status':>8} {'ms':>6}")
    print(f"  {'-'*4} {'-'*6} {'-'*16} {'-'*7} {'-'*8} {'-'*6}")

    report = ValidationReport(total=len(golden_qa))

    for qa in golden_qa:
        qa_id = qa["id"]
        question = qa["question"]
        expected_sources = qa["source_document"]
        difficulty = qa["difficulty"]
        category = qa["category"]

        start = time.time()

        if rag_chain:
            try:
                response = await rag_chain.query(question=question, mode="rag")
                retrieved_sources = [
                    s.get("metadata", {}).get("filename", "")
                    for s in response.sources
                ]
            except Exception:
                retrieved_sources = []
        else:
            try:
                results_list = await retriever.retrieve(query=question)
                retrieved_sources = list({
                    r.metadata.get("filename", "")
                    for r in results_list
                    if r.metadata.get("filename")
                })
            except Exception:
                retrieved_sources = []

        latency_ms = int((time.time() - start) * 1000)

        recall = compute_retrieval_recall(expected_sources, retrieved_sources)
        correct = recall >= 0.5  # At least half of expected sources found

        result = ValidationResult(
            qa_id=qa_id,
            question=question,
            difficulty=difficulty,
            category=category,
            expected_sources=expected_sources,
            retrieved_sources=retrieved_sources,
            retrieval_recall=recall,
            retrieval_correct=correct,
            latency_ms=latency_ms,
        )
        report.results.append(result)

        if correct:
            report.correct_retrievals += 1

        # Aggregate by difficulty
        diff_stats = report.by_difficulty.setdefault(
            difficulty, {"total": 0, "correct": 0, "recall_sum": 0.0}
        )
        diff_stats["total"] += 1
        diff_stats["correct"] += 1 if correct else 0
        diff_stats["recall_sum"] += recall

        # Aggregate by category
        cat_stats = report.by_category.setdefault(
            category, {"total": 0, "correct": 0, "recall_sum": 0.0}
        )
        cat_stats["total"] += 1
        cat_stats["correct"] += 1 if correct else 0
        cat_stats["recall_sum"] += recall

        status = "PASS" if correct else "FAIL"
        print(
            f"  {qa_id:>4} {difficulty:>6} {category:>16} "
            f"{recall:>6.0%} {status:>8} {latency_ms:>5}ms"
        )

    # Compute aggregates
    if report.total > 0:
        report.accuracy = report.correct_retrievals / report.total
        report.avg_recall = (
            sum(r.retrieval_recall for r in report.results) / report.total
        )
        report.avg_latency_ms = (
            sum(r.latency_ms for r in report.results) / report.total
        )

    # Print report
    print(f"\n{sep}")
    print("  VALIDATION REPORT")
    print(sep)
    print(f"  Total QA pairs:        {report.total}")
    print(f"  Correct retrievals:    {report.correct_retrievals}")
    print(f"  Retrieval accuracy:    {report.accuracy:.1%}")
    print(f"  Average recall:        {report.avg_recall:.1%}")
    print(f"  Average latency:       {report.avg_latency_ms:.0f} ms")

    print(f"\n  By Difficulty:")
    for diff, stats in sorted(report.by_difficulty.items()):
        n = stats["total"]
        acc = stats["correct"] / n if n else 0
        avg_r = stats["recall_sum"] / n if n else 0
        print(f"    {diff:>8s}: accuracy={acc:.0%}  avg_recall={avg_r:.1%}  (n={n})")

    print(f"\n  By Category:")
    for cat, stats in sorted(report.by_category.items()):
        n = stats["total"]
        acc = stats["correct"] / n if n else 0
        avg_r = stats["recall_sum"] / n if n else 0
        print(f"    {cat:>16s}: accuracy={acc:.0%}  avg_recall={avg_r:.1%}  (n={n})")

    # Failed items
    failed = [r for r in report.results if not r.retrieval_correct]
    if failed:
        print(f"\n  Failed Retrievals ({len(failed)}):")
        for r in failed:
            print(f"    Q{r.qa_id}: {r.question[:60]}...")
            print(f"      Expected: {r.expected_sources}")
            print(f"      Got:      {r.retrieved_sources}")

    print(sep)

    # Cleanup
    import shutil
    validation_dir = Path(VALIDATION_CHROMA_DIR)
    if validation_dir.exists():
        shutil.rmtree(validation_dir, ignore_errors=True)

    return report


def main():
    difficulty_filter = None
    full_mode = "--full" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--difficulty" and i + 1 < len(sys.argv):
            difficulty_filter = sys.argv[i + 1]

    report = asyncio.run(run_validation(
        difficulty_filter=difficulty_filter,
        full_mode=full_mode,
    ))

    # Exit code based on accuracy
    if report.accuracy >= 0.5:
        print(f"\n  RESULT: PASS (accuracy {report.accuracy:.1%} >= 50%)")
        sys.exit(0)
    else:
        print(f"\n  RESULT: FAIL (accuracy {report.accuracy:.1%} < 50%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
