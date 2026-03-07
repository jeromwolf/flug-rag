"""Batch ingest evaluation documents into fresh Milvus DB.

Uses Upstage Document Parse OCR for HWP files (no local text extraction).
PDF files use standard loader.
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path("/Users/blockmeta/Downloads/RAG평가용 문서 목록")
SUPPORTED_EXTS = {".hwp", ".pdf", ".docx", ".xlsx", ".pptx", ".txt"}
# HWP files need OCR (Upstage Document Parse) for text extraction
OCR_REQUIRED_EXTS = {".hwp"}


async def main():
    from pipeline.ingest import IngestPipeline

    pipeline = IngestPipeline()

    # Collect files (skip .zip)
    files = sorted(
        f for f in EVAL_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    logger.info("Found %d files to ingest", len(files))

    total_chunks = 0
    success = 0
    zero_chunks = []
    failed = []
    start = time.time()

    for i, f in enumerate(files, 1):
        try:
            needs_ocr = f.suffix.lower() in OCR_REQUIRED_EXTS
            dp_mode = "force_dp" if needs_ocr else "auto"
            logger.info("[%d/%d] Ingesting: %s (dp_mode=%s)", i, len(files), f.name, dp_mode)
            result = await pipeline.ingest(
                f,
                extra_metadata={"source_type": "내부규정", "folder": "RAG평가용"},
                dp_mode=dp_mode,
            )
            if result.chunk_count == 0:
                zero_chunks.append(f.name)
                logger.warning("  -> 0 chunks (empty document?)")
            else:
                total_chunks += result.chunk_count
                success += 1
                logger.info("  -> %d chunks (total: %d)", result.chunk_count, total_chunks)
        except Exception as e:
            failed.append((f.name, str(e)[:200]))
            logger.error("  -> FAILED: %s", e)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("DONE in %.1f seconds", elapsed)
    logger.info("Success: %d/%d files, Total chunks: %d", success, len(files), total_chunks)
    if zero_chunks:
        logger.info("Zero-chunk files (%d):", len(zero_chunks))
        for name in zero_chunks:
            logger.info("  %s", name)
    if failed:
        logger.info("Failed files (%d):", len(failed))
        for name, err in failed:
            logger.info("  %s: %s", name, err)


if __name__ == "__main__":
    asyncio.run(main())
