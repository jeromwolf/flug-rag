"""Quality management endpoints for RAG pipeline monitoring (SFR-008)."""

import logging
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.dependencies import require_role
from auth.models import Role, User
from config.settings import settings
from core.embeddings.tracker import get_tracker
from core.vectorstore.analyzer import VectorAnalyzer
from pipeline.document_monitor import get_document_monitor
from pipeline.reprocess_queue import get_reprocess_queue
from rag.chunk_quality import ChunkQualityAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy initialization for module-level singletons
_chunk_analyzer = None
_vector_analyzer = None


def _get_chunk_analyzer():
    """Get or create ChunkQualityAnalyzer instance."""
    global _chunk_analyzer
    if _chunk_analyzer is None:
        _chunk_analyzer = ChunkQualityAnalyzer()
    return _chunk_analyzer


def _get_vector_analyzer():
    """Get or create VectorAnalyzer instance."""
    global _vector_analyzer
    if _vector_analyzer is None:
        _vector_analyzer = VectorAnalyzer()
    return _vector_analyzer


# ============================================================================
# Response Models
# ============================================================================


class ChunkQualityReportResponse(BaseModel):
    """청크 품질 분석 결과 응답 모델."""

    total_chunks: int
    length_distribution: dict
    special_char_ratio: float
    duplicate_count: int
    near_duplicate_count: int
    empty_chunk_count: int
    table_chunk_count: int
    avg_semantic_completeness: float
    analyzed_at: str


class ChunkPreviewResponse(BaseModel):
    """청크 미리보기 응답 모델."""

    id: str
    content_preview: str
    index: int
    length: int
    has_table: bool
    page_number: Optional[int]


class DocumentChunkStatsResponse(BaseModel):
    """문서별 청크 통계 응답 모델."""

    document_id: str
    filename: str
    chunk_count: int
    avg_length: float
    min_length: int
    max_length: int


class EmbeddingStatusResponse(BaseModel):
    """임베딩 처리 상태 응답 모델."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    total_chunks_processed: int
    total_chunks_failed: int
    success_rate: float
    recent_failures: list[dict]


class EmbeddingJobResponse(BaseModel):
    """임베딩 작업 상세 응답 모델."""

    id: str
    document_id: str
    filename: str
    total_chunks: int
    success_count: int
    failure_count: int
    error_message: Optional[str]
    status: str
    started_at: str
    completed_at: Optional[str]


class VectorDistributionResponse(BaseModel):
    """벡터 분포 분석 응답 모델."""

    total_vectors: int
    dimension: int
    norm_stats: dict
    outlier_count: int
    outlier_ids: list[str]
    analyzed_at: str


class CollectionHealthResponse(BaseModel):
    """벡터 컬렉션 상태 응답 모델."""

    total_vectors: int
    collection_name: str
    index_type: str
    estimated_size_mb: float
    metadata: dict


class DocumentStatusResponse(BaseModel):
    """문서 처리 상태 응답 모델."""

    id: str
    filename: str
    file_type: str
    file_path: str
    file_hash: str
    status: str
    chunk_count: int
    error_message: Optional[str]
    processed_at: Optional[str]
    created_at: str
    updated_at: str


class DocumentStatusSummaryResponse(BaseModel):
    """문서 처리 상태 요약 응답 모델."""

    total: int
    by_status: dict[str, int]
    by_file_type: dict[str, int]


class DocumentChangeResponse(BaseModel):
    """파일 변경 감지 응답 모델."""

    filename: str
    file_path: str
    change_type: str
    old_hash: Optional[str]
    new_hash: Optional[str]


class QueueItemResponse(BaseModel):
    """재처리 큐 항목 응답 모델."""

    id: str
    document_id: str
    filename: str
    file_path: str
    error_message: str
    status: str
    retry_count: int
    max_retries: int
    created_at: str
    updated_at: str
    completed_at: Optional[str]


class QueueStatsResponse(BaseModel):
    """재처리 큐 통계 응답 모델."""

    total: int
    pending: int
    processing: int
    completed: int
    failed: int


# ============================================================================
# Document Processing Status Endpoints
# ============================================================================


@router.get("/quality/documents/status")
async def get_document_status(status_filter: Optional[str] = None):
    """
    문서 처리 상태 조회.

    Args:
        status_filter: 상태 필터 (pending/processing/completed/failed)

    Returns:
        문서 처리 상태 목록 및 요약 정보
    """
    try:
        monitor = await get_document_monitor()

        # Get summary stats
        summary = await monitor.get_status_summary()

        # Get document list
        documents = await monitor.get_all_status(status_filter=status_filter)

        return {
            "summary": asdict(summary),
            "documents": [asdict(doc) for doc in documents],
        }

    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(500, f"Failed to get document status: {str(e)}")


@router.get("/quality/documents/changes")
async def detect_document_changes():
    """
    업로드 디렉토리의 파일 변경 감지.

    Returns:
        변경된 파일 목록 (new/modified/deleted)
    """
    try:
        monitor = await get_document_monitor()
        changes = await monitor.detect_changes(settings.upload_dir)

        return {
            "total_changes": len(changes),
            "changes": [asdict(change) for change in changes],
        }

    except Exception as e:
        logger.error(f"Failed to detect document changes: {e}")
        raise HTTPException(500, f"Failed to detect document changes: {str(e)}")


# ============================================================================
# Chunk Quality Endpoints
# ============================================================================


@router.get("/quality/chunks/metrics", response_model=ChunkQualityReportResponse)
async def get_chunk_quality_metrics():
    """
    전체 청크 품질 지표 조회.

    Returns:
        청크 품질 분석 리포트 (길이 분포, 중복, 완결성 등)
    """
    try:
        analyzer = _get_chunk_analyzer()
        report = await analyzer.analyze_all_chunks()
        return asdict(report)

    except Exception as e:
        logger.error(f"Failed to analyze chunk quality: {e}")
        raise HTTPException(500, f"Failed to analyze chunk quality: {str(e)}")


@router.get("/quality/chunks/preview/{document_id}")
async def get_chunk_preview(document_id: str):
    """
    특정 문서의 청크 미리보기 조회.

    Args:
        document_id: 문서 ID

    Returns:
        청크 미리보기 목록 (첫 200자, 메타데이터)
    """
    try:
        analyzer = _get_chunk_analyzer()
        previews = await analyzer.get_chunk_preview(document_id)

        return {
            "document_id": document_id,
            "chunk_count": len(previews),
            "chunks": [asdict(preview) for preview in previews],
        }

    except Exception as e:
        logger.error(f"Failed to get chunk preview: {e}")
        raise HTTPException(500, f"Failed to get chunk preview: {str(e)}")


@router.get("/quality/chunks/by-document")
async def get_chunks_by_document():
    """
    문서별 청크 통계 집계.

    Returns:
        문서별 청크 개수, 평균 길이, 최소/최대 길이
    """
    try:
        analyzer = _get_chunk_analyzer()
        stats = await analyzer.get_chunks_by_document()

        return {
            "total_documents": len(stats),
            "documents": {
                doc_id: asdict(doc_stats) for doc_id, doc_stats in stats.items()
            },
        }

    except Exception as e:
        logger.error(f"Failed to get chunks by document: {e}")
        raise HTTPException(500, f"Failed to get chunks by document: {str(e)}")


# ============================================================================
# Embedding Status Endpoints
# ============================================================================


@router.get("/quality/embeddings/status", response_model=EmbeddingStatusResponse)
async def get_embedding_status():
    """
    임베딩 처리 현황 조회.

    Returns:
        임베딩 작업 통계 (완료/실패/대기, 성공률 등)
    """
    try:
        tracker = await get_tracker()
        status = await tracker.get_status()
        status_dict = asdict(status)

        # Convert recent_failures to list of dicts
        status_dict["recent_failures"] = [
            asdict(job) for job in status.recent_failures
        ]

        return status_dict

    except Exception as e:
        logger.error(f"Failed to get embedding status: {e}")
        raise HTTPException(500, f"Failed to get embedding status: {str(e)}")


@router.get("/quality/embeddings/history")
async def get_embedding_history(limit: int = 50):
    """
    임베딩 작업 이력 조회.

    Args:
        limit: 최대 조회 건수 (기본값: 50)

    Returns:
        최근 임베딩 작업 목록
    """
    try:
        tracker = await get_tracker()
        jobs = await tracker.get_job_history(limit=limit)

        return {
            "total": len(jobs),
            "jobs": [asdict(job) for job in jobs],
        }

    except Exception as e:
        logger.error(f"Failed to get embedding history: {e}")
        raise HTTPException(500, f"Failed to get embedding history: {str(e)}")


@router.get("/quality/embeddings/failed")
async def get_failed_embeddings():
    """
    실패한 임베딩 작업 조회.

    Returns:
        실패한 임베딩 작업 목록
    """
    try:
        tracker = await get_tracker()
        failed_jobs = await tracker.get_failed_jobs()

        return {
            "total": len(failed_jobs),
            "failed_jobs": [asdict(job) for job in failed_jobs],
        }

    except Exception as e:
        logger.error(f"Failed to get failed embeddings: {e}")
        raise HTTPException(500, f"Failed to get failed embeddings: {str(e)}")


# ============================================================================
# Vector Distribution Endpoints
# ============================================================================


@router.get("/quality/vectors/distribution", response_model=VectorDistributionResponse)
async def get_vector_distribution():
    """
    벡터 분포 분석 조회.

    Returns:
        벡터 노름 통계, 이상치 감지 결과
    """
    try:
        analyzer = _get_vector_analyzer()
        distribution = await analyzer.get_distribution()
        return asdict(distribution)

    except Exception as e:
        logger.error(f"Failed to get vector distribution: {e}")
        raise HTTPException(500, f"Failed to get vector distribution: {str(e)}")


@router.get("/quality/vectors/health", response_model=CollectionHealthResponse)
async def get_vector_health():
    """
    벡터 컬렉션 상태 조회.

    Returns:
        컬렉션 크기, 인덱스 타입, 예상 용량 등
    """
    try:
        analyzer = _get_vector_analyzer()
        health = await analyzer.get_collection_health()
        return asdict(health)

    except Exception as e:
        logger.error(f"Failed to get vector health: {e}")
        raise HTTPException(500, f"Failed to get vector health: {str(e)}")


# ============================================================================
# Reprocessing Queue Endpoints
# ============================================================================


@router.get("/quality/reprocess/queue")
async def get_reprocess_queue_items(
    status: Optional[str] = None,
    limit: int = 50,
):
    """
    재처리 큐 목록 조회.

    Args:
        status: 상태 필터 (pending/processing/completed/failed)
        limit: 최대 조회 건수 (기본값: 50)

    Returns:
        재처리 큐 항목 목록
    """
    try:
        queue = await get_reprocess_queue()
        items = await queue.get_queue(status=status, limit=limit)

        return {
            "total": len(items),
            "items": [asdict(item) for item in items],
        }

    except Exception as e:
        logger.error(f"Failed to get reprocess queue: {e}")
        raise HTTPException(500, f"Failed to get reprocess queue: {str(e)}")


@router.get("/quality/reprocess/stats", response_model=QueueStatsResponse)
async def get_reprocess_stats():
    """
    재처리 큐 통계 조회.

    Returns:
        큐 상태별 통계 (전체/대기/처리중/완료/실패)
    """
    try:
        queue = await get_reprocess_queue()
        stats = await queue.get_stats()
        return asdict(stats)

    except Exception as e:
        logger.error(f"Failed to get reprocess stats: {e}")
        raise HTTPException(500, f"Failed to get reprocess stats: {str(e)}")


@router.post("/quality/reprocess/retry/{queue_id}")
async def retry_reprocess_item(queue_id: str):
    """
    재처리 큐 항목 개별 재시도.

    Args:
        queue_id: 재처리 큐 항목 ID

    Returns:
        재시도 성공 여부
    """
    try:
        queue = await get_reprocess_queue()
        success = await queue.retry(queue_id)

        if not success:
            raise HTTPException(
                400,
                f"Cannot retry queue item {queue_id}: max retries exceeded or not found",
            )

        return {
            "status": "success",
            "queue_id": queue_id,
            "message": "Item reset to pending for retry",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry reprocess item: {e}")
        raise HTTPException(500, f"Failed to retry reprocess item: {str(e)}")


@router.post("/quality/reprocess/retry-all")
async def retry_all_failed():
    """
    실패한 모든 재처리 큐 항목 재시도.

    Returns:
        재시도된 항목 수
    """
    try:
        queue = await get_reprocess_queue()

        # Get count before retry
        stats_before = await queue.get_stats()
        failed_count = stats_before.failed

        # Retry all failed
        await queue.retry_all_failed()

        return {
            "status": "success",
            "retried_count": failed_count,
            "message": f"Reset {failed_count} failed items to pending",
        }

    except Exception as e:
        logger.error(f"Failed to retry all failed items: {e}")
        raise HTTPException(500, f"Failed to retry all failed items: {str(e)}")


@router.delete("/quality/reprocess/{queue_id}")
async def delete_reprocess_item(queue_id: str):
    """
    재처리 큐 항목 삭제.

    Args:
        queue_id: 재처리 큐 항목 ID

    Returns:
        삭제 성공 여부
    """
    try:
        queue = await get_reprocess_queue()
        await queue.delete(queue_id)

        return {
            "status": "success",
            "queue_id": queue_id,
            "message": "Queue item deleted",
        }

    except Exception as e:
        logger.error(f"Failed to delete reprocess item: {e}")
        raise HTTPException(500, f"Failed to delete reprocess item: {str(e)}")


# ============================================================================
# Golden Data Endpoints
# ============================================================================


class GoldenDataRequest(BaseModel):
    """골든 데이터 등록 요청 모델."""
    question: str = Field(..., description="질문")
    answer: str = Field(..., description="전문가 답변")
    source_message_id: str = Field("", description="원본 메시지 ID")
    source_session_id: str = Field("", description="원본 세션 ID")
    category: str = Field("", description="카테고리")
    evaluation_tag: str = Field("", description="평가 태그: accurate, partial, inaccurate, hallucination")


class GoldenDataUpdateRequest(BaseModel):
    """골든 데이터 수정 요청 모델."""
    question: Optional[str] = None
    answer: Optional[str] = None
    category: Optional[str] = None
    evaluation_tag: Optional[str] = None
    is_active: Optional[bool] = None


@router.get("/quality/golden-data")
async def list_golden_data(
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
    limit: int = 50,
    category: Optional[str] = None,
):
    """
    골든 데이터 목록 조회.

    Args:
        current_user: 현재 사용자 (ADMIN, MANAGER 권한 필요)
        limit: 최대 조회 건수 (기본값: 50)
        category: 카테고리 필터

    Returns:
        골든 데이터 항목 목록
    """
    try:
        from rag.golden_data import get_golden_data_manager
        manager = await get_golden_data_manager()
        entries = await manager.list_entries(limit=limit, category=category)
        return {"entries": [e.__dict__ for e in entries], "total": len(entries)}
    except Exception as e:
        logger.error(f"Failed to list golden data: {e}")
        raise HTTPException(500, f"Failed to list golden data: {str(e)}")


@router.post("/quality/golden-data")
async def create_golden_data(
    request: GoldenDataRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """
    골든 데이터 등록.

    Args:
        request: 골든 데이터 등록 요청
        current_user: 현재 사용자 (ADMIN, MANAGER 권한 필요)

    Returns:
        등록된 골든 데이터 항목
    """
    try:
        from rag.golden_data import get_golden_data_manager
        manager = await get_golden_data_manager()
        entry = await manager.add(
            question=request.question,
            answer=request.answer,
            source_message_id=request.source_message_id,
            source_session_id=request.source_session_id,
            category=request.category,
            evaluation_tag=request.evaluation_tag,
            created_by=current_user.username,
        )
        return {"status": "created", "entry": entry.__dict__}
    except Exception as e:
        logger.error(f"Failed to create golden data: {e}")
        raise HTTPException(500, f"Failed to create golden data: {str(e)}")


@router.put("/quality/golden-data/{entry_id}")
async def update_golden_data(
    entry_id: str,
    request: GoldenDataUpdateRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """
    골든 데이터 수정.

    Args:
        entry_id: 골든 데이터 항목 ID
        request: 골든 데이터 수정 요청
        current_user: 현재 사용자 (ADMIN, MANAGER 권한 필요)

    Returns:
        수정된 골든 데이터 항목
    """
    try:
        from rag.golden_data import get_golden_data_manager
        manager = await get_golden_data_manager()
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        entry = await manager.update(entry_id, **updates)
        return {"status": "updated", "entry": entry.__dict__}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update golden data: {e}")
        raise HTTPException(500, f"Failed to update golden data: {str(e)}")
