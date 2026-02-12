"""
OCR 학습 데이터 관리 API

자체 Document Parse 모델 학습을 위한 데이터 관리 엔드포인트.
- 수집된 학습 데이터 목록/상세 조회
- 문서 유형 라벨링
- JSONL 내보내기
"""

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from auth.dependencies import require_role
from auth.models import Role, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr-training"])


# ==================== Request/Response Models ====================

DOCUMENT_TYPES = Literal["규정집", "검사보고서", "설계도면", "안전점검표", "기타"]

class TrainingDataUpdate(BaseModel):
    """학습 데이터 라벨 수정 요청."""
    document_type: DOCUMENT_TYPES | None = Field(None, description="문서 유형")
    annotation: str | None = Field(None, max_length=500, description="수동 라벨/메모")


class TrainingDataExportRequest(BaseModel):
    """학습 데이터 내보내기 요청."""
    document_type: str | None = Field(None, description="문서 유형 필터 (None이면 전체)")


# ==================== Endpoints ====================

@router.get("/training-data")
async def list_training_data(
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
    document_type: str | None = Query(None, description="문서 유형 필터"),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    max_confidence: float | None = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """수집된 OCR 학습 데이터 목록 조회."""
    from pipeline.ocr.training_collector import get_training_collector

    collector = await get_training_collector()
    records, total = await collector.list_records(
        document_type=document_type,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        page=page,
        page_size=page_size,
    )

    return {
        "records": [
            {
                "id": r.id,
                "filename": r.filename,
                "document_type": r.document_type,
                "page_count": r.page_count,
                "ocr_provider": r.ocr_provider,
                "enhanced": r.enhanced,
                "confidence": round(r.confidence, 3),
                "file_size_bytes": r.file_size_bytes,
                "annotation": r.annotation,
                "created_at": r.created_at,
            }
            for r in records
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/training-data/stats")
async def get_training_stats(
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """학습 데이터 수집 통계."""
    from pipeline.ocr.training_collector import get_training_collector

    collector = await get_training_collector()
    stats = await collector.get_stats()
    return stats


@router.get("/training-data/{record_id}")
async def get_training_record(
    record_id: str,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """학습 데이터 상세 조회 (이미지 목록 포함)."""
    from pipeline.ocr.training_collector import get_training_collector

    collector = await get_training_collector()
    record = await collector.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Training record not found")

    # List images in the record's image directory
    image_dir = collector.training_dir / record.image_dir
    images = []
    if image_dir.exists():
        images = sorted([
            {"filename": f.name, "size_bytes": f.stat().st_size}
            for f in image_dir.iterdir() if f.is_file()
        ], key=lambda x: x["filename"])

    return {
        "id": record.id,
        "filename": record.filename,
        "document_type": record.document_type,
        "page_count": record.page_count,
        "ocr_provider": record.ocr_provider,
        "enhanced": record.enhanced,
        "confidence": round(record.confidence, 3),
        "file_size_bytes": record.file_size_bytes,
        "image_dir": record.image_dir,
        "result_path": record.result_path,
        "annotation": record.annotation,
        "created_at": record.created_at,
        "images": images,
    }


@router.put("/training-data/{record_id}")
async def update_training_record(
    record_id: str,
    request: TrainingDataUpdate,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """학습 데이터 라벨/메모 수정."""
    from pipeline.ocr.training_collector import get_training_collector

    collector = await get_training_collector()
    record = await collector.update_record(
        record_id=record_id,
        document_type=request.document_type,
        annotation=request.annotation,
    )
    if not record:
        raise HTTPException(status_code=404, detail="Training record not found")

    return {
        "status": "updated",
        "id": record.id,
        "document_type": record.document_type,
        "annotation": record.annotation,
    }


@router.delete("/training-data/{record_id}")
async def delete_training_record(
    record_id: str,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """학습 데이터 삭제 (파일 + DB)."""
    from pipeline.ocr.training_collector import get_training_collector

    collector = await get_training_collector()
    deleted = await collector.delete_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Training record not found")

    return {"status": "deleted", "id": record_id}


@router.post("/training-data/export")
async def export_training_data(
    request: TrainingDataExportRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """학습 데이터 JSONL 내보내기."""
    from pipeline.ocr.training_collector import get_training_collector

    collector = await get_training_collector()
    try:
        export_path = await collector.export_jsonl(
            document_type=request.document_type,
        )
        return FileResponse(
            path=str(export_path),
            filename=export_path.name,
            media_type="application/jsonl",
        )
    except Exception as e:
        logger.error("Training data export failed: %s", e)
        raise HTTPException(status_code=500, detail=f"내보내기 실패: {str(e)}")
