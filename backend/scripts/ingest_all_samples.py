"""Batch ingestion script for all sample documents.

Loads all sample .txt documents from data/sample/, processes them through
the chunking pipeline, stores them in the vector DB, and prints statistics.

Usage:
    python scripts/ingest_all_samples.py
    python scripts/ingest_all_samples.py --verify  # also run sample queries
"""

import asyncio
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- allow imports from the backend package root
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config.settings import settings
from core.embeddings import create_embedder
from core.vectorstore import create_vectorstore
from pipeline.ingest import IngestPipeline


SAMPLE_DIR = BACKEND_DIR / "data" / "sample"

# Sample verification queries (Korean)
VERIFY_QUERIES = [
    "안전관리 총괄책임자는 누구인가요?",
    "탄소강 배관의 매설 깊이 기준은?",
    "2024년 1분기 위험성평가에서 식별된 위험요인 수는?",
    "천연가스의 메탄 함량은?",
    "A동 메인 압축기의 제조사는?",
]


async def ingest_all(verify: bool = False) -> None:
    """Ingest all sample documents and print statistics."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  flux-rag Sample Document Ingestion")
    print(sep)

    # Discover sample files
    sample_files = sorted(SAMPLE_DIR.glob("*.txt"))
    if not sample_files:
        print(f"  [ERROR] No .txt files found in {SAMPLE_DIR}")
        sys.exit(1)

    print(f"  Found {len(sample_files)} sample documents:")
    for f in sample_files:
        print(f"    - {f.name}")
    print()

    # Create pipeline components
    print("  Initializing embedder and vector store...")
    embedder = create_embedder()
    vectorstore = create_vectorstore()

    pipeline = IngestPipeline(
        vectorstore=vectorstore,
        embedder=embedder,
    )

    # Clear existing data
    print("  Clearing existing vector store data...")
    await vectorstore.clear()

    # Ingest each document
    print("\n  Ingesting documents:")
    print(f"  {'Document':<45} {'Chunks':>7} {'Status':>10} {'Time':>8}")
    print(f"  {'-'*45} {'-'*7} {'-'*10} {'-'*8}")

    results = []
    total_start = time.time()

    for fpath in sample_files:
        start = time.time()
        result = await pipeline.ingest(file_path=fpath)
        elapsed = time.time() - start
        results.append(result)

        status_icon = "OK" if result.status == "completed" else "FAIL"
        print(
            f"  {fpath.name:<45} {result.chunk_count:>7} "
            f"{status_icon:>10} {elapsed:>7.1f}s"
        )
        if result.error:
            print(f"    Error: {result.error}")

    total_elapsed = time.time() - total_start

    # Print statistics
    print(f"\n{sep}")
    print("  INGESTION STATISTICS")
    print(sep)

    total_docs = len(results)
    successful = sum(1 for r in results if r.status == "completed")
    failed = total_docs - successful
    total_chunks = sum(r.chunk_count for r in results)
    chunk_counts = [r.chunk_count for r in results if r.chunk_count > 0]

    print(f"  Total documents:     {total_docs}")
    print(f"  Successful:          {successful}")
    print(f"  Failed:              {failed}")
    print(f"  Total chunks:        {total_chunks}")

    if chunk_counts:
        avg_chunks = sum(chunk_counts) / len(chunk_counts)
        min_chunks = min(chunk_counts)
        max_chunks = max(chunk_counts)
        print(f"  Avg chunks/doc:      {avg_chunks:.1f}")
        print(f"  Min chunks:          {min_chunks}")
        print(f"  Max chunks:          {max_chunks}")

    print(f"  Total time:          {total_elapsed:.1f}s")
    print(f"  Avg time/doc:        {total_elapsed / total_docs:.1f}s")

    # Per-document breakdown
    print(f"\n  Per-Document Breakdown:")
    for r in results:
        print(f"    {r.filename:<45} {r.chunk_count:>4} chunks")

    print(sep)

    # Verification with sample queries
    if verify:
        print(f"\n{sep}")
        print("  VERIFICATION - Sample Queries")
        print(sep)

        from rag.retriever import HybridRetriever

        retriever = HybridRetriever(
            vectorstore=vectorstore,
            embedder=embedder,
        )

        for query in VERIFY_QUERIES:
            print(f"\n  Q: {query}")
            try:
                results_list = await retriever.retrieve(query=query)
                if results_list:
                    top = results_list[0]
                    source = top.metadata.get("filename", "unknown")
                    print(f"  Top result: [{source}] score={top.score:.3f}")
                    preview = top.content[:100].replace("\n", " ")
                    print(f"  Preview: {preview}...")
                else:
                    print("  No results found")
            except Exception as e:
                print(f"  Error: {e}")

        print(f"\n{sep}")

    if failed > 0:
        print(f"\n  WARNING: {failed} document(s) failed to ingest.")
        sys.exit(1)
    else:
        print(f"\n  All {successful} documents ingested successfully.")


def main():
    verify = "--verify" in sys.argv
    asyncio.run(ingest_all(verify=verify))


if __name__ == "__main__":
    main()
