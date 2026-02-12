"""
OCR 학습 데이터 수집기

자체 Document Parse 모델 학습을 위한 데이터 수집 파이프라인.
- OCR 처리 시 원본 이미지 + OCR 결과를 쌍으로 저장
- PyMuPDF로 PDF 페이지별 PNG 이미지 추출
- SQLite 메타데이터 인덱스
"""

import asyncio
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from core.db import AsyncSQLiteManager, create_async_singleton

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    """OCR 학습 데이터 레코드."""
    id: str
    filename: str
    document_type: str  # 규정집/검사보고서/설계도면/안전점검표/기타
    page_count: int
    ocr_provider: str
    enhanced: bool
    confidence: float
    file_size_bytes: int
    image_dir: str      # relative path from training_dir
    result_path: str    # relative path from training_dir
    annotation: str = ""
    created_at: str = ""


class TrainingDataCollector(AsyncSQLiteManager):
    """OCR 학습 데이터 수집 및 관리."""

    def __init__(self, training_dir: Optional[Path] = None, db_path: Optional[Path] = None):
        from config.settings import settings
        self.training_dir = Path(training_dir or settings.ocr_training_dir)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        (self.training_dir / "images").mkdir(exist_ok=True)
        (self.training_dir / "results").mkdir(exist_ok=True)
        (self.training_dir / "exports").mkdir(exist_ok=True)
        super().__init__(db_path or (self.training_dir / "ocr_training.db"))

    def _validate_path(self, relative_path: str) -> Path:
        """경로 순회 공격 방지."""
        resolved = (self.training_dir / relative_path).resolve()
        if not str(resolved).startswith(str(self.training_dir.resolve())):
            raise ValueError(f"Path traversal detected: {relative_path}")
        return resolved

    async def _create_tables(self, db: aiosqlite.Connection):
        await db.execute("""
            CREATE TABLE IF NOT EXISTS training_records (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                document_type TEXT DEFAULT '기타',
                page_count INTEGER DEFAULT 0,
                ocr_provider TEXT DEFAULT '',
                enhanced INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                file_size_bytes INTEGER DEFAULT 0,
                image_dir TEXT DEFAULT '',
                result_path TEXT DEFAULT '',
                annotation TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_doctype ON training_records(document_type)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_confidence ON training_records(confidence)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_created ON training_records(created_at)"
        )
        await db.commit()

    async def collect(
        self,
        file_path: Path,
        ocr_result,  # OCRResult from pipeline.ocr.base
        filename: str = "",
        enhanced: bool = False,
    ) -> Optional[str]:
        """OCR 처리 결과 수집. Returns record_id or None on failure."""
        await self._ensure_initialized()

        record_id = str(uuid.uuid4())
        filename = filename or file_path.name
        now = datetime.now(timezone.utc).isoformat()

        try:
            # 1) Extract page images from PDF using PyMuPDF
            image_dir = Path("images") / record_id
            abs_image_dir = self.training_dir / image_dir
            abs_image_dir.mkdir(parents=True, exist_ok=True)

            # Copy source file to training dir first (to avoid temp file deletion race)
            source_copy = abs_image_dir / f"source{file_path.suffix}"
            await asyncio.to_thread(shutil.copy2, file_path, source_copy)

            # Extract from the local copy
            page_count = await self._extract_images(source_copy, abs_image_dir)

            # Remove the source copy after extraction
            try:
                source_copy.unlink()
            except Exception:
                pass

            # 이미지 추출 실패 시 수집 건너뛰기
            if page_count == 0:
                logger.debug("No images extracted for %s, skipping collection", filename)
                # Cleanup
                if abs_image_dir.exists():
                    shutil.rmtree(abs_image_dir)
                return None

            # 2) Save OCR result as JSON
            result_path = Path("results") / f"{record_id}.json"
            abs_result_path = self.training_dir / result_path
            result_data = {
                "text": ocr_result.text,
                "confidence": ocr_result.confidence,
                "pages": ocr_result.pages,
                "tables": ocr_result.tables,
                "layout": ocr_result.layout,
                "metadata": ocr_result.metadata,
            }
            await asyncio.to_thread(
                self._write_json, abs_result_path, result_data
            )

            # 3) Get file size
            file_size = file_path.stat().st_size if file_path.exists() else 0

            # 4) Insert metadata record
            record = TrainingRecord(
                id=record_id,
                filename=filename,
                document_type="기타",
                page_count=page_count,
                ocr_provider=ocr_result.metadata.get("provider", ""),
                enhanced=enhanced,
                confidence=ocr_result.confidence,
                file_size_bytes=file_size,
                image_dir=str(image_dir),
                result_path=str(result_path),
                annotation="",
                created_at=now,
            )

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO training_records
                    (id, filename, document_type, page_count, ocr_provider, enhanced,
                     confidence, file_size_bytes, image_dir, result_path, annotation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id, record.filename, record.document_type, record.page_count,
                    record.ocr_provider, int(record.enhanced), record.confidence,
                    record.file_size_bytes, record.image_dir, record.result_path,
                    record.annotation, record.created_at,
                ))
                await db.commit()

            logger.info("OCR training data collected: %s (%d pages, conf=%.2f)",
                        filename, page_count, ocr_result.confidence)
            return record_id

        except Exception as e:
            logger.warning("OCR training data collection failed for %s: %s", filename, e)
            # Cleanup partial files
            try:
                abs_image_dir = self.training_dir / "images" / record_id
                if abs_image_dir.exists():
                    shutil.rmtree(abs_image_dir)
                abs_result_path = self.training_dir / "results" / f"{record_id}.json"
                if abs_result_path.exists():
                    abs_result_path.unlink()
            except Exception:
                pass
            return None

    async def _extract_images(self, file_path: Path, output_dir: Path) -> int:
        """PDF에서 페이지별 PNG 이미지 추출. 이미지 파일은 원본 복사."""
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return await asyncio.to_thread(self._extract_pdf_pages, file_path, output_dir)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            # 이미지 파일은 원본 복사
            dest = output_dir / f"page_001{ext}"
            await asyncio.to_thread(shutil.copy2, file_path, dest)
            return 1
        else:
            return 0

    def _extract_pdf_pages(self, file_path: Path, output_dir: Path) -> int:
        """PyMuPDF로 PDF 페이지를 PNG로 변환 (sync, called via to_thread)."""
        import fitz

        from config.settings import settings
        dpi = settings.ocr_training_image_dpi

        doc = fitz.open(str(file_path))
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            # DPI 기반 zoom factor 계산 (72 DPI 기준)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            output_path = output_dir / f"page_{page_num + 1:03d}.png"
            pix.save(str(output_path))

        doc.close()
        return page_count

    @staticmethod
    def _write_json(path: Path, data: dict):
        """JSON 파일 저장 (sync)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def list_records(
        self,
        document_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TrainingRecord], int]:
        """필터 기반 목록 조회. Returns (records, total_count)."""
        await self._ensure_initialized()

        conditions = []
        params = []

        if document_type:
            conditions.append("document_type = ?")
            params.append(document_type)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(max_confidence)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Total count
            async with db.execute(
                f"SELECT COUNT(*) FROM training_records{where_clause}", params
            ) as cursor:
                row = await cursor.fetchone()
                total = row[0]

            # Paginated results
            offset = (page - 1) * page_size
            query_params = params + [page_size, offset]
            async with db.execute(
                f"SELECT * FROM training_records{where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                query_params,
            ) as cursor:
                rows = await cursor.fetchall()
                records = [
                    TrainingRecord(**{**dict(r), "enhanced": bool(r["enhanced"])})
                    for r in rows
                ]

        return records, total

    async def get_record(self, record_id: str) -> Optional[TrainingRecord]:
        """단건 조회."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM training_records WHERE id = ?", (record_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return TrainingRecord(**{**dict(row), "enhanced": bool(row["enhanced"])})
        return None

    async def update_record(
        self, record_id: str, document_type: Optional[str] = None, annotation: Optional[str] = None
    ) -> Optional[TrainingRecord]:
        """라벨/메모 수정."""
        await self._ensure_initialized()

        updates = {}
        if document_type is not None:
            updates["document_type"] = document_type
        if annotation is not None:
            updates["annotation"] = annotation

        if not updates:
            return await self.get_record(record_id)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [record_id]

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"UPDATE training_records SET {set_clause} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                return None
            await db.commit()

        return await self.get_record(record_id)

    async def delete_record(self, record_id: str) -> bool:
        """레코드 + 파일 삭제."""
        await self._ensure_initialized()

        record = await self.get_record(record_id)
        if not record:
            return False

        # Delete files
        try:
            image_dir = self._validate_path(record.image_dir)
            if image_dir.exists():
                shutil.rmtree(image_dir)
            result_path = self._validate_path(record.result_path)
            if result_path.exists():
                result_path.unlink()
        except Exception as e:
            logger.warning("Failed to delete training files for %s: %s", record_id, e)

        # Delete DB record
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM training_records WHERE id = ?", (record_id,))
            await db.commit()

        return True

    async def get_stats(self) -> dict:
        """수집 통계."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 총 건수
            async with db.execute("SELECT COUNT(*) as cnt FROM training_records") as cur:
                total = (await cur.fetchone())["cnt"]

            # 문서 유형별 건수
            async with db.execute(
                "SELECT document_type, COUNT(*) as cnt FROM training_records GROUP BY document_type ORDER BY cnt DESC"
            ) as cur:
                by_type = {row["document_type"]: row["cnt"] async for row in cur}

            # 신뢰도 분포
            async with db.execute("""
                SELECT
                    MIN(confidence) as min_conf,
                    MAX(confidence) as max_conf,
                    AVG(confidence) as avg_conf,
                    SUM(page_count) as total_pages,
                    SUM(file_size_bytes) as total_bytes
                FROM training_records
            """) as cur:
                row = await cur.fetchone()

            # 프로바이더별 건수
            async with db.execute(
                "SELECT ocr_provider, COUNT(*) as cnt FROM training_records GROUP BY ocr_provider"
            ) as cur:
                by_provider = {row["ocr_provider"]: row["cnt"] async for row in cur}

        return {
            "total_records": total,
            "by_document_type": by_type,
            "by_provider": by_provider,
            "total_pages": row["total_pages"] or 0,
            "total_size_mb": round((row["total_bytes"] or 0) / (1024 * 1024), 1),
            "confidence": {
                "min": round(row["min_conf"] or 0, 3),
                "max": round(row["max_conf"] or 0, 3),
                "avg": round(row["avg_conf"] or 0, 3),
            },
        }

    async def export_jsonl(
        self, document_type: Optional[str] = None
    ) -> Path:
        """JSONL 학습 데이터 내보내기. Returns path to the export file."""
        await self._ensure_initialized()

        # Query records (async)
        conditions = []
        params = []
        if document_type:
            conditions.append("document_type = ?")
            params.append(document_type)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM training_records{where_clause} ORDER BY created_at",
                params,
            ) as cursor:
                records = [dict(row) async for row in cursor]

        # Generate JSONL in thread (sync I/O)
        export_dir = self.training_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"_{document_type}" if document_type else "_all"
        export_path = export_dir / f"training_data{suffix}_{timestamp}.jsonl"

        await asyncio.to_thread(self._write_export_jsonl, export_path, records)

        logger.info("Training data exported: %s", export_path.name)
        return export_path

    def _write_export_jsonl(self, export_path: Path, records: list[dict]):
        """JSONL 파일 생성 (sync, called via to_thread)."""
        with open(export_path, "w", encoding="utf-8") as f:
            for record in records:
                # Validate and load result path
                try:
                    result_path = self._validate_path(record["result_path"])
                except ValueError:
                    logger.warning("Skipping record with invalid result_path: %s", record["id"])
                    continue

                ocr_data = {}
                if result_path.exists():
                    with open(result_path, "r", encoding="utf-8") as rf:
                        ocr_data = json.load(rf)

                # Validate and list images
                try:
                    image_dir = self._validate_path(record["image_dir"])
                except ValueError:
                    logger.warning("Skipping record with invalid image_dir: %s", record["id"])
                    continue

                images = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
                # Also include jpg/jpeg/tiff for non-PDF sources
                if not images:
                    images = sorted(image_dir.glob("*.*"))
                    images = [i for i in images if i.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff")]

                pages = ocr_data.get("pages", [])
                for i, img_path in enumerate(images):
                    page_data = pages[i] if i < len(pages) else {}
                    entry = {
                        "image_path": str(img_path.relative_to(self.training_dir)),
                        "text": page_data.get("content", ocr_data.get("text", "") if i == 0 else ""),
                        "tables": ocr_data.get("tables", []) if i == 0 else [],
                        "layout": ocr_data.get("layout", {}),
                        "document_type": record["document_type"],
                        "confidence": record["confidence"],
                        "filename": record["filename"],
                        "page_number": i + 1,
                        "total_pages": record["page_count"],
                        "metadata": ocr_data.get("metadata", {}),
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Singleton
get_training_collector = create_async_singleton(TrainingDataCollector)
reset_training_collector = get_training_collector.reset
