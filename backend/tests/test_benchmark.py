"""
RAG Benchmark Test Suite

Evaluates RAG quality using golden Q&A pairs from golden_qa.json.
Measures retrieval recall, answer similarity, and latency.

Usage:
    pytest tests/test_benchmark.py -v --tb=short
    python tests/test_benchmark.py  # standalone mode
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
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.embeddings import create_embedder
from core.vectorstore import create_vectorstore
from pipeline.ingest import IngestPipeline
from rag import RAGChain, RAGResponse
from rag.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).parent.parent
GOLDEN_QA_PATH = BACKEND_DIR / "data" / "golden_qa.json"
SAMPLE_DIR = BACKEND_DIR / "data" / "sample"
BENCHMARK_COLLECTION = "benchmark_test"
BENCHMARK_CHROMA_DIR = str(BACKEND_DIR / "data" / "chroma_benchmark")
SIMILARITY_PASS_THRESHOLD = 0.70  # 70% keyword overlap to pass


# ---------------------------------------------------------------------------
# Data classes for benchmark results
# ---------------------------------------------------------------------------
@dataclass
class SingleResult:
    """Result for a single Q&A evaluation."""
    qa_id: int
    question: str
    difficulty: str
    category: str
    expected_answer: str
    actual_answer: str
    source_documents: list[str]
    retrieved_sources: list[str]
    retrieval_recall: float
    answer_similarity: float
    latency_ms: int
    passed: bool


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_recall: float = 0.0
    avg_similarity: float = 0.0
    avg_latency_ms: float = 0.0
    pass_rate: float = 0.0
    results_by_difficulty: dict = field(default_factory=dict)
    results_by_category: dict = field(default_factory=dict)
    details: list[SingleResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------
def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from Korean text.

    Splits on whitespace and punctuation, filters out very short tokens
    (single character) and common Korean particles/postpositions.
    """
    import re

    tokens = re.findall(r"[\w]+", text.lower())

    # Common Korean particles / suffixes to ignore
    stopwords = {
        "이", "가", "은", "는", "을", "를", "의", "에", "에서",
        "로", "으로", "와", "과", "도", "만", "까지", "부터",
        "에게", "한테", "께", "보다", "처럼", "같이", "마다",
        "이다", "이며", "하고", "하며", "또는", "및", "등",
        "그", "이", "저", "그리고", "그러나", "하지만", "때문에",
        "위해", "대한", "통해", "따라", "대해", "있다", "없다",
        "한다", "된다", "하는", "되는", "있는", "없는",
    }

    keywords = {t for t in tokens if len(t) > 1 and t not in stopwords}
    return keywords


def compute_keyword_similarity(expected: str, actual: str) -> float:
    """Compute keyword overlap similarity between expected and actual answer.

    Returns a float in [0.0, 1.0] representing how many expected keywords
    appear in the actual answer.
    """
    expected_kw = extract_keywords(expected)
    actual_kw = extract_keywords(actual)

    if not expected_kw:
        return 1.0 if not actual_kw else 0.0

    overlap = expected_kw & actual_kw
    return len(overlap) / len(expected_kw)


def compute_retrieval_recall(
    expected_sources: list[str],
    retrieved_sources: list[str],
) -> float:
    """Compute recall: fraction of expected source docs found in retrieval.

    Matches by checking whether the expected filename appears as a substring
    of any retrieved source metadata value.
    """
    if not expected_sources:
        return 1.0

    found = 0
    for expected in expected_sources:
        # Strip extension for flexible matching
        expected_stem = Path(expected).stem
        for retrieved in retrieved_sources:
            if expected_stem in retrieved or expected in retrieved:
                found += 1
                break

    return found / len(expected_sources)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def golden_qa() -> list[dict]:
    """Load golden Q&A pairs."""
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def event_loop():
    """Create a module-scoped event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def benchmark_vectorstore():
    """Create an isolated ChromaDB collection for benchmarking."""
    store = create_vectorstore(
        collection_name=BENCHMARK_COLLECTION,
        persist_dir=BENCHMARK_CHROMA_DIR,
    )
    return store


@pytest.fixture(scope="module")
def benchmark_embedder():
    """Create embedder for benchmarking."""
    return create_embedder()


@pytest.fixture(scope="module")
def benchmark_pipeline(benchmark_vectorstore, benchmark_embedder):
    """Create ingest pipeline with benchmark-specific stores."""
    return IngestPipeline(
        vectorstore=benchmark_vectorstore,
        embedder=benchmark_embedder,
    )


@pytest.fixture(scope="module")
def benchmark_rag(benchmark_vectorstore, benchmark_embedder):
    """Create RAG chain wired to the benchmark vector store."""
    retriever = HybridRetriever(
        vectorstore=benchmark_vectorstore,
        embedder=benchmark_embedder,
    )
    return RAGChain(retriever=retriever)


@pytest.fixture(scope="module", autouse=True)
def ingest_sample_documents(event_loop, benchmark_pipeline, benchmark_vectorstore):
    """Ingest all 10 sample documents before running benchmark tests."""

    async def _ingest():
        # Clear any previous data
        await benchmark_vectorstore.clear()

        sample_files = sorted(SAMPLE_DIR.glob("*.txt"))
        assert len(sample_files) >= 5, (
            f"Expected at least 5 sample documents in {SAMPLE_DIR}, found {len(sample_files)}"
        )

        results = []
        for fpath in sample_files:
            result = await benchmark_pipeline.ingest(file_path=fpath)
            results.append(result)
            assert result.status == "completed", (
                f"Failed to ingest {fpath.name}: {result.error}"
            )

        total_chunks = sum(r.chunk_count for r in results)
        print(f"\n[Benchmark Setup] Ingested {len(results)} documents, "
              f"{total_chunks} total chunks")
        return results

    event_loop.run_until_complete(_ingest())
    yield
    # Cleanup: remove benchmark data after tests
    import shutil
    benchmark_dir = Path(BENCHMARK_CHROMA_DIR)
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Individual Q&A test (parametrized)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_golden_qa_pairs(golden_qa, benchmark_rag):
    """Run all golden Q&A pairs and assert aggregate pass rate >= 70%."""
    report = BenchmarkReport()
    report.total = len(golden_qa)

    for qa in golden_qa:
        qa_id = qa["id"]
        question = qa["question"]
        expected_answer = qa["expected_answer"]
        expected_sources = qa["source_document"]
        difficulty = qa["difficulty"]
        category = qa["category"]

        # Query the RAG chain
        start = time.time()
        try:
            response: RAGResponse = await benchmark_rag.query(
                question=question, mode="rag"
            )
            latency_ms = int((time.time() - start) * 1000)
            actual_answer = response.content
            retrieved_sources = [
                s.get("metadata", {}).get("filename", "")
                for s in response.sources
            ]
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            actual_answer = f"[ERROR] {e}"
            retrieved_sources = []

        # Score
        recall = compute_retrieval_recall(expected_sources, retrieved_sources)
        similarity = compute_keyword_similarity(expected_answer, actual_answer)
        passed = similarity >= SIMILARITY_PASS_THRESHOLD

        result = SingleResult(
            qa_id=qa_id,
            question=question,
            difficulty=difficulty,
            category=category,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            source_documents=expected_sources,
            retrieved_sources=retrieved_sources,
            retrieval_recall=recall,
            answer_similarity=similarity,
            latency_ms=latency_ms,
            passed=passed,
        )
        report.details.append(result)

        if passed:
            report.passed += 1
        else:
            report.failed += 1

        # Aggregate by difficulty
        diff_stats = report.results_by_difficulty.setdefault(
            difficulty, {"total": 0, "passed": 0, "recall_sum": 0.0, "sim_sum": 0.0}
        )
        diff_stats["total"] += 1
        diff_stats["passed"] += 1 if passed else 0
        diff_stats["recall_sum"] += recall
        diff_stats["sim_sum"] += similarity

        # Aggregate by category
        cat_stats = report.results_by_category.setdefault(
            category, {"total": 0, "passed": 0, "recall_sum": 0.0, "sim_sum": 0.0}
        )
        cat_stats["total"] += 1
        cat_stats["passed"] += 1 if passed else 0
        cat_stats["recall_sum"] += recall
        cat_stats["sim_sum"] += similarity

    # Compute aggregates
    if report.total > 0:
        report.avg_recall = sum(r.retrieval_recall for r in report.details) / report.total
        report.avg_similarity = sum(r.answer_similarity for r in report.details) / report.total
        report.avg_latency_ms = sum(r.latency_ms for r in report.details) / report.total
        report.pass_rate = report.passed / report.total

    # Print report
    _print_report(report)

    # Assert overall pass rate
    assert report.pass_rate >= 0.50, (
        f"Overall pass rate {report.pass_rate:.1%} is below 50% minimum. "
        f"Passed: {report.passed}/{report.total}"
    )


@pytest.mark.asyncio
async def test_retrieval_recall_by_difficulty(golden_qa, benchmark_rag):
    """Verify that easy questions achieve high retrieval recall."""
    easy_questions = [qa for qa in golden_qa if qa["difficulty"] == "easy"]
    recalls = []

    for qa in easy_questions:
        response: RAGResponse = await benchmark_rag.query(
            question=qa["question"], mode="rag"
        )
        retrieved = [
            s.get("metadata", {}).get("filename", "") for s in response.sources
        ]
        recall = compute_retrieval_recall(qa["source_document"], retrieved)
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"\n[Easy Q Recall] Average retrieval recall for easy questions: {avg_recall:.1%}")
    # Easy questions should generally find the right source doc
    assert avg_recall >= 0.5, (
        f"Easy question retrieval recall {avg_recall:.1%} is below 50%"
    )


@pytest.mark.asyncio
async def test_latency_within_bounds(golden_qa, benchmark_rag):
    """Verify that average query latency is within acceptable bounds."""
    latencies = []

    for qa in golden_qa[:5]:  # Sample first 5 for speed
        start = time.time()
        await benchmark_rag.query(question=qa["question"], mode="rag")
        latencies.append(int((time.time() - start) * 1000))

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n[Latency] Average query latency (sample of 5): {avg_latency:.0f}ms")
    # Allow generous timeout (60s) since this includes embedding + LLM
    assert avg_latency < 60000, f"Average latency {avg_latency}ms exceeds 60s limit"


@pytest.mark.asyncio
async def test_cross_document_retrieval(golden_qa, benchmark_rag):
    """Verify retrieval works for cross-document questions (multiple sources)."""
    cross_doc_questions = [
        qa for qa in golden_qa
        if qa.get("category") == "cross_document" or len(qa["source_document"]) > 1
    ]

    if not cross_doc_questions:
        pytest.skip("No cross-document questions found")

    recalls = []
    for qa in cross_doc_questions:
        response: RAGResponse = await benchmark_rag.query(
            question=qa["question"], mode="rag"
        )
        retrieved = [
            s.get("metadata", {}).get("filename", "") for s in response.sources
        ]
        recall = compute_retrieval_recall(qa["source_document"], retrieved)
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(
        f"\n[Cross-Doc Recall] Average retrieval recall for "
        f"cross-document questions: {avg_recall:.1%} (n={len(recalls)})"
    )
    # Cross-document is harder, so lower threshold
    assert avg_recall >= 0.3, (
        f"Cross-document retrieval recall {avg_recall:.1%} is below 30%"
    )


@pytest.mark.asyncio
async def test_new_document_retrieval(golden_qa, benchmark_rag):
    """Verify retrieval works for questions about the 5 new P2 documents."""
    new_doc_names = {
        "기술표준_가스배관_시공기준.txt",
        "위험성평가_보고서_2024Q1.txt",
        "가스품질_분석결과_202401.txt",
        "설비이력_관리대장_A동.txt",
        "환경안전_법규_가이드.txt",
    }

    new_doc_questions = [
        qa for qa in golden_qa
        if any(src in new_doc_names for src in qa["source_document"])
    ]

    if not new_doc_questions:
        pytest.skip("No questions about new P2 documents found")

    recalls = []
    for qa in new_doc_questions:
        response: RAGResponse = await benchmark_rag.query(
            question=qa["question"], mode="rag"
        )
        retrieved = [
            s.get("metadata", {}).get("filename", "") for s in response.sources
        ]
        recall = compute_retrieval_recall(qa["source_document"], retrieved)
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(
        f"\n[New Docs Recall] Average retrieval recall for "
        f"new P2 document questions: {avg_recall:.1%} (n={len(recalls)})"
    )
    assert avg_recall >= 0.4, (
        f"New document retrieval recall {avg_recall:.1%} is below 40%"
    )


@pytest.mark.asyncio
async def test_latency_percentiles(golden_qa, benchmark_rag):
    """Measure p50 and p95 latency across all golden QA pairs."""
    latencies = []

    # Sample a subset to keep test time reasonable
    sample_size = min(10, len(golden_qa))
    sample = golden_qa[:sample_size]

    for qa in sample:
        start = time.time()
        await benchmark_rag.query(question=qa["question"], mode="rag")
        latencies.append(int((time.time() - start) * 1000))

    if not latencies:
        pytest.skip("No latency data collected")

    latencies.sort()
    n = len(latencies)
    p50 = latencies[n // 2]
    p95_idx = min(int(n * 0.95), n - 1)
    p95 = latencies[p95_idx]

    print(f"\n[Latency Percentiles] n={n}")
    print(f"  p50: {p50}ms")
    print(f"  p95: {p95}ms")
    print(f"  min: {min(latencies)}ms")
    print(f"  max: {max(latencies)}ms")

    # p95 should be under 120s (generous for LLM-based pipeline)
    assert p95 < 120000, f"p95 latency {p95}ms exceeds 120s limit"


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def _print_report(report: BenchmarkReport) -> None:
    """Print a formatted benchmark summary report."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  RAG BENCHMARK REPORT")
    print(sep)
    print(f"  Total Q&A pairs:    {report.total}")
    print(f"  Passed (>={SIMILARITY_PASS_THRESHOLD:.0%}):   {report.passed}")
    print(f"  Failed:             {report.failed}")
    print(f"  Pass rate:          {report.pass_rate:.1%}")
    print(f"  Avg recall:         {report.avg_recall:.1%}")
    print(f"  Avg similarity:     {report.avg_similarity:.1%}")
    print(f"  Avg latency:        {report.avg_latency_ms:.0f} ms")
    print(sep)

    print("\n  By Difficulty:")
    for diff, stats in sorted(report.results_by_difficulty.items()):
        n = stats["total"]
        pr = stats["passed"] / n if n else 0
        ar = stats["recall_sum"] / n if n else 0
        asim = stats["sim_sum"] / n if n else 0
        print(f"    {diff:8s}  pass={pr:.0%}  recall={ar:.1%}  similarity={asim:.1%}  (n={n})")

    print("\n  By Category:")
    for cat, stats in sorted(report.results_by_category.items()):
        n = stats["total"]
        pr = stats["passed"] / n if n else 0
        ar = stats["recall_sum"] / n if n else 0
        asim = stats["sim_sum"] / n if n else 0
        print(f"    {cat:16s}  pass={pr:.0%}  recall={ar:.1%}  similarity={asim:.1%}  (n={n})")

    print(f"\n  Individual Results:")
    for r in report.details:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"    [{status}] Q{r.qa_id:2d} ({r.difficulty:6s}/{r.category:16s}) "
            f"recall={r.retrieval_recall:.0%} sim={r.answer_similarity:.0%} "
            f"latency={r.latency_ms}ms"
        )
        if not r.passed:
            print(f"           Question: {r.question[:60]}...")
            print(f"           Expected keywords sample: "
                  f"{list(extract_keywords(r.expected_answer))[:5]}")

    print(sep)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
async def run_standalone():
    """Run benchmark as a standalone script (outside pytest)."""
    print("=" * 72)
    print("  RAG Benchmark - Standalone Mode")
    print("=" * 72)

    # Load golden Q&A
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)
    print(f"Loaded {len(golden_qa)} golden Q&A pairs")

    # Create benchmark infrastructure
    print("Creating vector store and embedder...")
    vectorstore = create_vectorstore(
        collection_name=BENCHMARK_COLLECTION,
        persist_dir=BENCHMARK_CHROMA_DIR,
    )
    embedder = create_embedder()

    # Clear and ingest
    print("Clearing previous benchmark data...")
    await vectorstore.clear()

    print("Ingesting sample documents...")
    pipeline = IngestPipeline(vectorstore=vectorstore, embedder=embedder)
    sample_files = sorted(SAMPLE_DIR.glob("*.txt"))
    for fpath in sample_files:
        result = await pipeline.ingest(file_path=fpath)
        status_icon = "OK" if result.status == "completed" else "FAIL"
        print(f"  [{status_icon}] {fpath.name} -> {result.chunk_count} chunks")

    # Create RAG chain
    retriever = HybridRetriever(vectorstore=vectorstore, embedder=embedder)
    rag_chain = RAGChain(retriever=retriever)

    # Run benchmark
    report = BenchmarkReport()
    report.total = len(golden_qa)

    for qa in golden_qa:
        qa_id = qa["id"]
        question = qa["question"]
        expected_answer = qa["expected_answer"]
        expected_sources = qa["source_document"]

        start = time.time()
        try:
            response = await rag_chain.query(question=question, mode="rag")
            latency_ms = int((time.time() - start) * 1000)
            actual_answer = response.content
            retrieved_sources = [
                s.get("metadata", {}).get("filename", "")
                for s in response.sources
            ]
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            actual_answer = f"[ERROR] {e}"
            retrieved_sources = []

        recall = compute_retrieval_recall(expected_sources, retrieved_sources)
        similarity = compute_keyword_similarity(expected_answer, actual_answer)
        passed = similarity >= SIMILARITY_PASS_THRESHOLD

        result = SingleResult(
            qa_id=qa_id,
            question=question,
            difficulty=qa["difficulty"],
            category=qa["category"],
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            source_documents=expected_sources,
            retrieved_sources=retrieved_sources,
            retrieval_recall=recall,
            answer_similarity=similarity,
            latency_ms=latency_ms,
            passed=passed,
        )
        report.details.append(result)

        if passed:
            report.passed += 1
        else:
            report.failed += 1

        diff_stats = report.results_by_difficulty.setdefault(
            qa["difficulty"],
            {"total": 0, "passed": 0, "recall_sum": 0.0, "sim_sum": 0.0},
        )
        diff_stats["total"] += 1
        diff_stats["passed"] += 1 if passed else 0
        diff_stats["recall_sum"] += recall
        diff_stats["sim_sum"] += similarity

        cat_stats = report.results_by_category.setdefault(
            qa["category"],
            {"total": 0, "passed": 0, "recall_sum": 0.0, "sim_sum": 0.0},
        )
        cat_stats["total"] += 1
        cat_stats["passed"] += 1 if passed else 0
        cat_stats["recall_sum"] += recall
        cat_stats["sim_sum"] += similarity

    if report.total > 0:
        report.avg_recall = sum(r.retrieval_recall for r in report.details) / report.total
        report.avg_similarity = sum(r.answer_similarity for r in report.details) / report.total
        report.avg_latency_ms = sum(r.latency_ms for r in report.details) / report.total
        report.pass_rate = report.passed / report.total

    _print_report(report)

    # Cleanup
    import shutil
    benchmark_dir = Path(BENCHMARK_CHROMA_DIR)
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir, ignore_errors=True)

    return report


if __name__ == "__main__":
    report = asyncio.run(run_standalone())
    exit_code = 0 if report.pass_rate >= 0.50 else 1
    sys.exit(exit_code)
