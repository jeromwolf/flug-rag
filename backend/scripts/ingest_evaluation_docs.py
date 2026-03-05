"""Batch ingestion script for RAG evaluation documents.

Clears ALL existing vector store data, then ingests the 89 evaluation
documents (HWP + PDF) provided by KOGAS for the demo evaluation.

Usage:
    cd backend
    python scripts/ingest_evaluation_docs.py
    python scripts/ingest_evaluation_docs.py --verify  # also run sample queries
"""

import asyncio
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config.settings import settings
from core.embeddings import create_embedder
from core.vectorstore import create_vectorstore
from pipeline.ingest import IngestPipeline

# Evaluation documents directory
EVAL_DOCS_DIR = Path("/Users/blockmeta/Downloads/RAG평가용 문서 목록")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".hwp", ".pdf", ".txt", ".docx", ".xlsx", ".pptx"}

# Sample verification queries based on the evaluation documents
VERIFY_QUERIES = [
    "부정청탁 신고사무 처리절차는 어떻게 되나요?",
    "직장 내 괴롭힘 구제절차는?",
    "공익신고자 보호 방법은?",
    "채용절차에 관한 지침의 주요 내용은?",
    "이해충돌 방지제도 운영지침에서 사적이해관계 신고 대상은?",
    "건설업 안전보건관리지침의 적용 범위는?",
    "회계감사인선임위원회 구성은?",
    "갑질 예방지침에서 금지되는 행위 유형은?",
]


async def ingest_evaluation(verify: bool = False) -> None:
    """Clear existing data and ingest all evaluation documents."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  RAG 평가용 문서 인제스트 (클린 스타트)")
    print(sep)

    # Discover evaluation files
    if not EVAL_DOCS_DIR.exists():
        print(f"  [ERROR] 평가 문서 폴더를 찾을 수 없습니다: {EVAL_DOCS_DIR}")
        sys.exit(1)

    eval_files = sorted([
        f for f in EVAL_DOCS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not eval_files:
        print(f"  [ERROR] 지원되는 파일이 없습니다: {EVAL_DOCS_DIR}")
        sys.exit(1)

    # Count by extension
    ext_counts = {}
    for f in eval_files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print(f"  평가 문서 폴더: {EVAL_DOCS_DIR}")
    print(f"  총 파일 수: {len(eval_files)}")
    for ext, count in sorted(ext_counts.items()):
        print(f"    {ext}: {count}개")
    print()

    # Initialize pipeline
    print("  임베더 및 벡터스토어 초기화 중...")
    embedder = create_embedder()
    vectorstore = create_vectorstore()

    pipeline = IngestPipeline(
        vectorstore=vectorstore,
        embedder=embedder,
    )

    # ★ Clear ALL existing data
    print("  ★ 기존 벡터스토어 데이터 전체 삭제 중...")
    await vectorstore.clear()
    count_after_clear = await vectorstore.count()
    print(f"  삭제 완료. 현재 청크 수: {count_after_clear}")
    print()

    # Ingest each document
    print(f"  {'문서명':<55} {'청크':>5} {'상태':>6} {'시간':>7}")
    print(f"  {'-'*55} {'-'*5} {'-'*6} {'-'*7}")

    results = []
    total_start = time.time()
    failed_files = []

    for i, fpath in enumerate(eval_files, 1):
        start = time.time()
        try:
            result = await pipeline.ingest(
                file_path=fpath,
                dp_mode="force_dp",
                extra_metadata={
                    "source_type": "내부규정",
                    "dataset": "평가용",
                },
            )
            elapsed = time.time() - start
            results.append(result)

            status = "OK" if result.status == "completed" else "FAIL"
            print(
                f"  [{i:2d}/{len(eval_files)}] {fpath.name:<50} "
                f"{result.chunk_count:>5} {status:>6} {elapsed:>6.1f}s"
            )
            if result.error:
                print(f"         Error: {result.error}")
                failed_files.append(fpath.name)
        except Exception as e:
            elapsed = time.time() - start
            print(
                f"  [{i:2d}/{len(eval_files)}] {fpath.name:<50} "
                f"{'0':>5} {'ERROR':>6} {elapsed:>6.1f}s"
            )
            print(f"         Exception: {e}")
            failed_files.append(fpath.name)

    total_elapsed = time.time() - total_start

    # Statistics
    print(f"\n{sep}")
    print("  인제스트 통계")
    print(sep)

    total_docs = len(eval_files)
    successful = sum(1 for r in results if r.status == "completed")
    failed = total_docs - successful
    total_chunks = sum(r.chunk_count for r in results)

    chunk_counts = [r.chunk_count for r in results if r.chunk_count > 0]

    print(f"  전체 문서:       {total_docs}")
    print(f"  성공:            {successful}")
    print(f"  실패:            {failed}")
    print(f"  총 청크 수:      {total_chunks}")

    if chunk_counts:
        avg_chunks = sum(chunk_counts) / len(chunk_counts)
        print(f"  평균 청크/문서:  {avg_chunks:.1f}")
        print(f"  최소 청크:       {min(chunk_counts)}")
        print(f"  최대 청크:       {max(chunk_counts)}")

    print(f"  총 소요시간:     {total_elapsed:.1f}s")
    print(f"  평균 시간/문서:  {total_elapsed / max(total_docs, 1):.1f}s")

    # Final vector store count
    final_count = await vectorstore.count()
    print(f"\n  벡터스토어 최종 청크 수: {final_count}")

    if failed_files:
        print(f"\n  ⚠ 실패한 파일:")
        for fname in failed_files:
            print(f"    - {fname}")

    print(sep)

    # Verification
    if verify:
        print(f"\n{sep}")
        print("  검증 - 샘플 질의 테스트")
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
                    print(f"  → [{source}] score={top.score:.3f}")
                    preview = top.content[:120].replace("\n", " ")
                    print(f"    {preview}...")
                else:
                    print("  → 검색 결과 없음")
            except Exception as e:
                print(f"  → Error: {e}")

        print(f"\n{sep}")

    if failed > 0:
        print(f"\n  ⚠ {failed}개 문서 인제스트 실패. 확인 필요.")
        sys.exit(1)
    else:
        print(f"\n  ✅ {successful}개 문서 모두 인제스트 성공!")
        print(f"  총 {total_chunks}개 청크가 벡터스토어에 저장되었습니다.")


def main():
    verify = "--verify" in sys.argv
    asyncio.run(ingest_evaluation(verify=verify))


if __name__ == "__main__":
    main()
