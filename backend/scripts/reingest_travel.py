"""Re-ingest travel reports with Upstage Document Parse OCR.

Deletes existing low-quality travel chunks (ingested with local_only) from Milvus
and re-ingests with force_dp (Upstage OCR) for higher quality extraction.

Usage:
    cd backend
    python scripts/reingest_travel.py [options]

Options:
    --dp-mode       Document Parse mode: force_dp (default), auto, local_only
    --limit         Limit number of files to ingest (0 = all)
    --skip-delete   Skip deletion of existing chunks (for testing)
    --dry-run       Show what would be done without executing anything
    --target        Comma-separated filename patterns to filter (e.g. "Manzanillo,요르단,호주")

Examples:
    # Full re-ingest with Upstage OCR
    python scripts/reingest_travel.py

    # Dry run to preview
    python scripts/reingest_travel.py --dry-run

    # Re-ingest only specific files
    python scripts/reingest_travel.py --target "Manzanillo,요르단"

    # Test with 5 files, keep existing chunks
    python scripts/reingest_travel.py --limit 5 --skip-delete
"""

import asyncio
import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add backend to path so imports work from the backend/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("reingest_travel")
logger.setLevel(logging.INFO)

TRAVEL_DIR = "data/uploads/국외출장_결과보고서"
REPORT_PATH = "data/ingest_report_travel_reingest.json"
SOURCE_TYPE = "출장보고서"  # Value assigned by MetadataExtractor._detect_source_type


def validate_upstage_api_key(settings) -> bool:
    """Check that an Upstage API key is available when force_dp is requested."""
    key = settings.upstage_api_key or os.environ.get("UPSTAGE_API_KEY", "")
    if not key:
        logger.error(
            "Upstage API key is required for dp_mode=force_dp.\n"
            "  Set UPSTAGE_API_KEY environment variable or add it to backend/.env"
        )
        return False
    return True


async def count_travel_chunks_milvus() -> int:
    """Count chunks in Milvus where source_type matches the travel source type."""
    try:
        from core.vectorstore import create_vectorstore
        vs = create_vectorstore()
        results = await asyncio.to_thread(
            vs.client.query,
            collection_name=vs.collection_name,
            filter=f'source_type == "{SOURCE_TYPE}"',
            output_fields=["count(*)"],
        )
        if results and len(results) > 0:
            return results[0].get("count(*)", 0)
        return 0
    except Exception as e:
        logger.warning(f"Could not count travel chunks: {e}")
        return -1


async def delete_travel_chunks_milvus(dry_run: bool = False) -> int:
    """Delete all travel report chunks from Milvus.

    Returns:
        Number of chunks deleted (or that would be deleted in dry_run mode).
    """
    from core.vectorstore import create_vectorstore
    vs = create_vectorstore()

    count = await count_travel_chunks_milvus()

    if count == 0:
        logger.info(f"No existing chunks with source_type='{SOURCE_TYPE}' found in Milvus.")
        return 0

    logger.info(f"Found {count} chunks to delete (source_type='{SOURCE_TYPE}').")

    if dry_run:
        logger.info("[DRY RUN] Skipping deletion.")
        return count

    await asyncio.to_thread(
        vs.client.delete,
        collection_name=vs.collection_name,
        filter=f'source_type == "{SOURCE_TYPE}"',
    )
    logger.info(f"Deleted {count} chunks from Milvus.")
    return count


def collect_files(travel_dir: str, target_patterns: list[str] | None = None, limit: int = 0) -> list[str]:
    """Collect PDF and HWP files from the travel report directory recursively.

    Args:
        travel_dir: Root directory to search.
        target_patterns: If provided, only return files containing one of these substrings.
        limit: Maximum number of files to return (0 = all).

    Returns:
        Sorted list of file paths.
    """
    pdfs = sorted(glob.glob(f"{travel_dir}/**/*.pdf", recursive=True))
    hwps = sorted(glob.glob(f"{travel_dir}/**/*.hwp", recursive=True))
    all_files = pdfs + hwps

    if target_patterns:
        filtered = []
        for f in all_files:
            fname = Path(f).name
            if any(pat in fname for pat in target_patterns):
                filtered.append(f)
        all_files = filtered

    if limit > 0:
        all_files = all_files[:limit]

    return all_files


async def run_reingest(
    dp_mode: str = "force_dp",
    limit: int = 0,
    skip_delete: bool = False,
    dry_run: bool = False,
    target_patterns: list[str] | None = None,
) -> dict:
    """Execute the full re-ingest workflow.

    Returns:
        Report dictionary.
    """
    from config.settings import settings
    from pipeline.ingest import IngestPipeline

    logger.info("=" * 65)
    logger.info("=== 국외출장 결과보고서 재인제스트 시작 ===")
    logger.info(f"dp_mode     : {dp_mode}")
    logger.info(f"vectorstore : Milvus")
    logger.info(f"source_type : {SOURCE_TYPE}")
    if target_patterns:
        logger.info(f"target      : {target_patterns}")
    if dry_run:
        logger.info("[DRY RUN MODE] No changes will be made.")
    logger.info("=" * 65)

    # ------------------------------------------------------------------ #
    # Step 1: Validate Upstage API key (only required for force_dp / auto)
    # ------------------------------------------------------------------ #
    if dp_mode in ("force_dp", "auto"):
        if not validate_upstage_api_key(settings):
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 2: Count & delete existing travel chunks
    # ------------------------------------------------------------------ #
    chunks_before = await count_travel_chunks_milvus()
    logger.info(f"\n[Step 1] 기존 청크 현황: {chunks_before}개 (source_type='{SOURCE_TYPE}')")

    deleted_count = 0
    if skip_delete:
        logger.info("[Step 1] --skip-delete: 삭제 단계 생략.")
    else:
        deleted_count = await delete_travel_chunks_milvus(dry_run=dry_run)

    # ------------------------------------------------------------------ #
    # Step 3: Collect files
    # ------------------------------------------------------------------ #
    all_files = collect_files(TRAVEL_DIR, target_patterns=target_patterns, limit=limit)
    total = len(all_files)

    pdf_count = sum(1 for f in all_files if f.lower().endswith(".pdf"))
    hwp_count = sum(1 for f in all_files if f.lower().endswith(".hwp"))

    logger.info(f"\n[Step 2] 파일 수집 완료: {total}개 (PDF: {pdf_count}, HWP: {hwp_count})")

    if total == 0:
        logger.warning("처리할 파일이 없습니다. 경로 또는 --target 패턴을 확인하세요.")
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dp_mode": dp_mode,
            "dry_run": dry_run,
            "chunks_before": chunks_before,
            "deleted_chunks": deleted_count,
            "total_files": 0,
            "success": 0,
            "failed": 0,
            "total_chunks": 0,
            "elapsed_minutes": 0,
            "failed_files": [],
            "results": [],
        }
        _save_report(report)
        return report

    if dry_run:
        logger.info(f"\n[DRY RUN] 인제스트 대상 파일 ({total}개):")
        for i, f in enumerate(all_files[:20]):
            logger.info(f"  {i+1:3d}. {Path(f).name}")
        if total > 20:
            logger.info(f"  ... 외 {total - 20}개")
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dp_mode": dp_mode,
            "dry_run": True,
            "chunks_before": chunks_before,
            "deleted_chunks": deleted_count,
            "total_files": total,
            "success": 0,
            "failed": 0,
            "total_chunks": 0,
            "elapsed_minutes": 0,
            "failed_files": [],
            "results": [{"filename": Path(f).name, "status": "dry_run"} for f in all_files],
        }
        _save_report(report)
        return report

    # ------------------------------------------------------------------ #
    # Step 4: Re-ingest
    # ------------------------------------------------------------------ #
    logger.info(f"\n[Step 3] 재인제스트 시작 (dp_mode={dp_mode})...")
    pipeline = IngestPipeline()

    results = []
    success = 0
    failed = 0
    total_chunks = 0
    failed_files = []
    start_time = time.time()

    for i, path in enumerate(all_files):
        fname = Path(path).name
        t0 = time.time()

        try:
            r = await pipeline.ingest(
                path,
                dp_mode=dp_mode,
                extra_metadata={
                    "source": "국외출장_결과보고서",
                    "category": "travel_report",
                },
            )
            elapsed = time.time() - t0

            if r.status == "completed":
                success += 1
                total_chunks += r.chunk_count
                status_str = "OK"
            else:
                failed += 1
                failed_files.append({"file": fname, "error": r.error})
                status_str = "FAIL"

            results.append({
                "filename": fname,
                "status": r.status,
                "chunks": r.chunk_count,
                "elapsed_s": round(elapsed, 1),
                "error": r.error,
            })

            # Progress log every 10 files or on failure
            if (i + 1) % 10 == 0 or r.status == "failed":
                elapsed_total = time.time() - start_time
                rate = (i + 1) / elapsed_total * 60  # files per minute
                eta_min = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"[{i+1:3d}/{total}] {status_str} {r.chunk_count:3d}ch {elapsed:5.1f}s | "
                    f"누적: {success}ok {failed}fail {total_chunks}ch | "
                    f"속도: {rate:.0f}files/min ETA: {eta_min:.0f}min | {fname[:45]}"
                )

        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            failed_files.append({"file": fname, "error": str(e)})
            results.append({
                "filename": fname,
                "status": "error",
                "chunks": 0,
                "elapsed_s": round(elapsed, 1),
                "error": str(e),
            })
            logger.error(f"[{i+1:3d}/{total}] ERROR {fname[:45]}: {str(e)[:80]}")

    total_elapsed = time.time() - start_time

    # ------------------------------------------------------------------ #
    # Step 5: Verification - count chunks after ingestion
    # ------------------------------------------------------------------ #
    chunks_after = await count_travel_chunks_milvus()
    logger.info(f"\n[Step 4] 검증: Milvus 청크 수")
    logger.info(f"  이전: {chunks_before}개")
    logger.info(f"  삭제: {deleted_count}개")
    logger.info(f"  신규: {total_chunks}개 (파이프라인 집계)")
    logger.info(f"  이후: {chunks_after}개 (Milvus 실제 조회)")

    # ------------------------------------------------------------------ #
    # Step 6: Generate report
    # ------------------------------------------------------------------ #
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dp_mode": dp_mode,
        "dry_run": dry_run,
        "chunks_before": chunks_before,
        "deleted_chunks": deleted_count,
        "chunks_after": chunks_after,
        "total_files": total,
        "success": success,
        "failed": failed,
        "total_chunks": total_chunks,
        "elapsed_minutes": round(total_elapsed / 60, 1),
        "avg_seconds_per_file": round(total_elapsed / max(total, 1), 1),
        "failed_files": failed_files,
        "results": results,
    }

    _save_report(report)

    logger.info("=" * 65)
    logger.info("=== 재인제스트 완료 ===")
    logger.info(f"성공: {success}/{total}, 실패: {failed}/{total}")
    logger.info(f"총 청크: {total_chunks} (Milvus 실제: {chunks_after})")
    logger.info(f"소요 시간: {total_elapsed/60:.1f}분 ({total_elapsed/max(total,1):.1f}s/파일)")
    logger.info(f"리포트: {REPORT_PATH}")

    if failed_files:
        logger.info(f"\n=== 실패 파일 ({len(failed_files)}개) ===")
        for ff in failed_files[:20]:
            logger.info(f"  {ff['file']}: {ff.get('error', '')[:80]}")

    return report


def _save_report(report: dict) -> None:
    """Save the ingest report JSON file."""
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="국외출장 결과보고서 재인제스트 (Upstage Document Parse OCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full re-ingest with Upstage OCR (default)
  python scripts/reingest_travel.py

  # Preview what would be done
  python scripts/reingest_travel.py --dry-run

  # Re-ingest only files matching specific patterns
  python scripts/reingest_travel.py --target "Manzanillo,요르단,호주"

  # Test with 5 files, keep existing chunks
  python scripts/reingest_travel.py --limit 5 --skip-delete

  # Re-ingest without OCR (fallback to local extraction)
  python scripts/reingest_travel.py --dp-mode local_only --skip-delete
        """,
    )
    parser.add_argument(
        "--dp-mode",
        default="force_dp",
        choices=["auto", "local_only", "force_dp"],
        help="Document Parse mode: force_dp (Upstage OCR, default), auto, local_only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files to process (0 = all files)",
    )
    parser.add_argument(
        "--skip-delete",
        action="store_true",
        help="Skip deletion of existing travel chunks in Milvus",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing ingestion or deletion",
    )
    parser.add_argument(
        "--target",
        default="",
        help="Comma-separated filename substrings to filter (e.g. 'Manzanillo,요르단,호주')",
    )

    args = parser.parse_args()

    target_patterns = None
    if args.target:
        target_patterns = [p.strip() for p in args.target.split(",") if p.strip()]

    asyncio.run(
        run_reingest(
            dp_mode=args.dp_mode,
            limit=args.limit,
            skip_delete=args.skip_delete,
            dry_run=args.dry_run,
            target_patterns=target_patterns,
        )
    )
