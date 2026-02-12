"""
OCR 관리 API 엔드포인트

SFR-015: OCR 결과 조회, 재처리, 온프레미스 상태 확인
"""

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user, require_role
from auth.models import Role, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])


class OCRProcessRequest(BaseModel):
    """OCR 처리 요청"""
    provider: Literal["cloud", "onprem"] | None = Field(None, description="OCR 프로바이더 (None이면 설정값)")
    enhanced: bool = Field(False, description="Enhanced Mode (복잡한 표/차트/다단)")


class OCRResultResponse(BaseModel):
    """OCR 결과 응답"""
    text: str
    confidence: float
    page_count: int
    table_count: int
    provider: str
    enhanced: bool
    metadata: dict = {}


class OCRHealthResponse(BaseModel):
    """OCR 서버 상태 응답"""
    cloud_available: bool
    onprem_available: bool
    onprem_url: str
    current_provider: str


@router.get("/health", response_model=OCRHealthResponse)
async def ocr_health(
    current_user: User | None = Depends(get_current_user),
):
    """
    OCR 서버 상태 확인

    - 클라우드/온프레미스 프로바이더 가용성 확인
    """
    from config.settings import settings

    cloud_available = bool(settings.upstage_api_key)

    # 온프레미스 상태 확인
    onprem_available = False
    try:
        from pipeline.ocr import UpstageOnpremOCR
        onprem = UpstageOnpremOCR()
        onprem_available = await onprem.health_check()
        await onprem.close()
    except Exception:
        pass

    return OCRHealthResponse(
        cloud_available=cloud_available,
        onprem_available=onprem_available,
        onprem_url=settings.ocr_onprem_url,
        current_provider=settings.ocr_provider,
    )


@router.post("/process", response_model=OCRResultResponse)
async def process_document_ocr(
    file: UploadFile = File(...),
    provider: str | None = None,
    enhanced: bool = False,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER, Role.USER])),
):
    """
    문서 OCR 처리

    - 파일 업로드 후 OCR 처리 결과 반환
    - provider: cloud 또는 onprem (기본: 설정값)
    - enhanced: 복잡한 표/차트/다단 처리 모드
    """
    from config.settings import settings

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for OCR: {ext}"
        )

    # 임시 파일 저장
    import tempfile
    tmp_path = None
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from pipeline.ocr import create_ocr
        ocr = create_ocr(provider=provider)

        result = await ocr.process(tmp_path, enhanced=enhanced)

        # 학습 데이터 수집 (non-blocking)
        try:
            if settings.ocr_training_enabled:
                from pipeline.ocr.training_collector import get_training_collector
                collector = await get_training_collector()
                await collector.collect(
                    file_path=tmp_path,
                    ocr_result=result,
                    filename=file.filename or "unknown",
                    enhanced=enhanced,
                )
        except Exception as e:
            logger.debug("OCR training data collection skipped: %s", e)

        await ocr.close()

        return OCRResultResponse(
            text=result.text,
            confidence=result.confidence,
            page_count=len(result.pages),
            table_count=len(result.tables),
            provider=result.metadata.get("provider", settings.ocr_provider),
            enhanced=enhanced,
            metadata=result.metadata,
        )

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR 처리 실패: {str(e)}")

    finally:
        # 임시 파일 정리
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


@router.post("/switch-provider")
async def switch_ocr_provider(
    provider: Literal["cloud", "onprem"],
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """
    OCR 프로바이더 전환 (런타임)

    - Admin만 가능
    - 서버 재시작 없이 프로바이더 전환
    """
    from config.settings import settings

    old_provider = settings.ocr_provider

    # 온프레미스 전환 시 가용성 확인
    if provider == "onprem":
        try:
            from pipeline.ocr import UpstageOnpremOCR
            onprem = UpstageOnpremOCR()
            available = await onprem.health_check()
            await onprem.close()
            if not available:
                raise HTTPException(
                    status_code=503,
                    detail="온프레미스 OCR 서버에 연결할 수 없습니다"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"온프레미스 OCR 서버 확인 실패: {str(e)}"
            )

    # 런타임 설정 변경
    settings.ocr_provider = provider

    return {
        "status": "success",
        "previous_provider": old_provider,
        "current_provider": provider,
    }
