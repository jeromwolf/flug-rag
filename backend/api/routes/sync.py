"""
문서 동기화 API 라우트

SFR-005: 문서 동기화 시스템 엔드포인트
- 스케줄러 상태 조회
- 즉시 동기화 실행
- 스케줄 설정/중지
- 동기화 이력 조회
"""

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pipeline.scheduler import get_scheduler
from pipeline.sync import get_sync_engine

router = APIRouter(prefix="/sync")


# Request/Response Models
class ScheduleRequest(BaseModel):
    """스케줄 설정 요청"""
    cron_expression: str = Field(
        default="0 2 * * *",
        description="크론 표현식 (기본: 매일 새벽 2시)"
    )
    watch_dirs: list[str] | None = Field(
        default=None,
        description="감시할 디렉토리 목록 (None이면 기본 경로 사용)"
    )


class SyncStatusResponse(BaseModel):
    """스케줄러 상태 응답"""
    scheduler_running: bool = Field(description="스케줄러 실행 여부")
    next_run_time: str | None = Field(description="다음 실행 시각 (ISO 8601)")
    job_count: int = Field(description="등록된 작업 수")
    last_sync: str | None = Field(description="마지막 동기화 시각 (ISO 8601)")


class SyncResultResponse(BaseModel):
    """동기화 결과 응답"""
    started_at: str = Field(description="시작 시각 (ISO 8601)")
    completed_at: str = Field(description="완료 시각 (ISO 8601)")
    new_files: int = Field(description="신규 파일 수")
    modified_files: int = Field(description="수정된 파일 수")
    deleted_files: int = Field(description="삭제된 파일 수")
    failed_files: int = Field(description="실패한 파일 수")
    errors: list[str] = Field(description="오류 메시지 목록")
    duration_seconds: float = Field(description="소요 시간 (초)")


class SyncHistoryResponse(BaseModel):
    """동기화 이력 응답"""
    history: list[SyncResultResponse] = Field(description="동기화 이력 목록")
    total: int = Field(description="전체 이력 수")


@router.get("/status", response_model=SyncStatusResponse)
async def get_sync_status():
    """
    스케줄러 상태 조회

    Returns:
        SyncStatusResponse: 스케줄러 실행 상태 및 마지막 동기화 시각
    """
    try:
        scheduler = get_scheduler()
        engine = get_sync_engine()

        status = scheduler.get_status()
        last_sync = engine.get_last_sync()

        return SyncStatusResponse(
            scheduler_running=status["running"],
            next_run_time=status.get("next_run_time"),
            job_count=status.get("job_count", 0),
            last_sync=last_sync
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")


@router.post("/trigger", response_model=SyncResultResponse)
async def trigger_sync(source_dir: str | None = None):
    """
    즉시 동기화 실행

    Args:
        source_dir: 동기화할 소스 디렉토리 (None이면 기본 경로 사용)

    Returns:
        SyncResultResponse: 동기화 실행 결과
    """
    try:
        engine = get_sync_engine()

        # 동기화 실행
        result = engine.run_sync(source_dir)

        # dataclass를 dict로 변환 후 Response 모델로 변환
        result_dict = asdict(result)
        return SyncResultResponse(**result_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동기화 실행 실패: {str(e)}")


@router.post("/schedule")
async def schedule_sync(request: ScheduleRequest):
    """
    동기화 스케줄 설정 및 시작

    Args:
        request: 크론 표현식 및 감시 디렉토리 설정

    Returns:
        dict: 스케줄러 시작 상태
    """
    try:
        scheduler = get_scheduler()

        # 스케줄러 시작
        scheduler.start(
            watch_dirs=request.watch_dirs,
            cron_expression=request.cron_expression
        )

        # 상태 조회
        status = scheduler.get_status()

        return {
            "status": "started",
            "cron_expression": request.cron_expression,
            "watch_dirs": request.watch_dirs,
            "next_run_time": status.get("next_run_time"),
            "job_count": status.get("job_count", 0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스케줄 설정 실패: {str(e)}")


@router.post("/stop")
async def stop_sync():
    """
    스케줄러 중지

    Returns:
        dict: 중지 상태
    """
    try:
        scheduler = get_scheduler()
        scheduler.stop()

        return {"status": "stopped"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스케줄러 중지 실패: {str(e)}")


@router.get("/history", response_model=SyncHistoryResponse)
async def get_sync_history(
    limit: Annotated[int, Query(ge=1, le=100)] = 10
):
    """
    동기화 이력 조회

    Args:
        limit: 조회할 이력 개수 (1-100, 기본: 10)

    Returns:
        SyncHistoryResponse: 동기화 이력 목록 및 전체 개수
    """
    try:
        engine = get_sync_engine()

        # 이력 조회
        history_list = engine.get_history(limit=limit)

        # dataclass 목록을 Response 모델로 변환
        history_responses = [
            SyncResultResponse(**asdict(result))
            for result in history_list
        ]

        return SyncHistoryResponse(
            history=history_responses,
            total=len(history_responses)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이력 조회 실패: {str(e)}")
