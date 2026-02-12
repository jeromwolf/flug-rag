"""Batch ingest script for 한국가스기술공사 내부규정 documents.

Usage:
    cd backend
    python scripts/ingest_internal_rules.py [--dp-mode auto|local_only|force_dp] [--limit N]
"""

import asyncio
import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ingest_rules")
logger.setLevel(logging.INFO)

RULES_DIR = "data/uploads/한국가스기술공사_내부규정/한국가스기술공사_내부규정"
REPORT_PATH = "data/ingest_report_internal_rules.json"


async def main(dp_mode: str = "local_only", limit: int = 0):
    from pipeline.ingest import IngestPipeline

    pipeline = IngestPipeline()

    # Collect all PDF and HWP files
    pdfs = sorted(glob.glob(f"{RULES_DIR}/*.pdf"))
    hwps = sorted(glob.glob(f"{RULES_DIR}/*.hwp"))
    all_files = pdfs + hwps

    if limit > 0:
        all_files = all_files[:limit]

    total = len(all_files)
    logger.info("=== 내부규정 배치 인제스트 시작 ===")
    logger.info(f"총 파일: {total} (PDF: {len([f for f in all_files if f.endswith('.pdf')])}, HWP: {len([f for f in all_files if f.endswith('.hwp')])})")
    logger.info(f"dp_mode: {dp_mode}")

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
                extra_metadata={"source": "내부규정", "category": "internal_rules"},
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
                rate = (i + 1) / elapsed_total * 60
                eta_min = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"[{i+1:3d}/{total}] {status_str} {r.chunk_count:3d}ch {elapsed:5.1f}s | "
                    f"누적: {success}ok {failed}fail {total_chunks}ch | "
                    f"속도: {rate:.0f}files/min ETA: {eta_min:.0f}min | {fname[:40]}"
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
            logger.error(f"[{i+1:3d}/{total}] ERROR {fname[:40]}: {str(e)[:80]}")

    total_elapsed = time.time() - start_time

    # Generate report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dp_mode": dp_mode,
        "total_files": total,
        "success": success,
        "failed": failed,
        "total_chunks": total_chunks,
        "elapsed_minutes": round(total_elapsed / 60, 1),
        "avg_seconds_per_file": round(total_elapsed / max(total, 1), 1),
        "failed_files": failed_files,
        "results": results,
    }

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"=== 인제스트 완료 ===")
    logger.info(f"성공: {success}/{total}, 실패: {failed}/{total}")
    logger.info(f"총 청크: {total_chunks}")
    logger.info(f"소요 시간: {total_elapsed/60:.1f}분 ({total_elapsed/max(total,1):.1f}s/파일)")
    logger.info(f"리포트: {REPORT_PATH}")

    if failed_files:
        logger.info(f"\n=== 실패 파일 ({len(failed_files)}개) ===")
        for ff in failed_files[:20]:
            logger.info(f"  {ff['file']}: {ff['error'][:80]}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="내부규정 배치 인제스트")
    parser.add_argument("--dp-mode", default="local_only", choices=["auto", "local_only", "force_dp"])
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0=all)")
    args = parser.parse_args()
    asyncio.run(main(dp_mode=args.dp_mode, limit=args.limit))
