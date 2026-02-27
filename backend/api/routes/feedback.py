"""Feedback endpoints for answer quality management."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.schemas import FeedbackRequest, FeedbackResponse
from auth.dependencies import get_current_user, require_role
from auth.models import Role, User
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Simple file-based feedback storage (SQLite in production)
FEEDBACK_FILE = settings.data_dir / "feedback.jsonl"


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, current_user: User | None = Depends(get_current_user)):
    feedback_id = str(uuid.uuid4())

    rating_label_map = {1: "accurate", 0: "partial", -1: "inaccurate"}

    entry = {
        "id": feedback_id,
        "message_id": request.message_id,
        "session_id": request.session_id,
        "rating": request.rating,
        "rating_label": rating_label_map.get(request.rating, "unknown"),
        "comment": request.comment,
        "corrected_answer": request.corrected_answer,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return FeedbackResponse(id=feedback_id, status="saved")


@router.get("/feedback")
async def list_feedback(limit: int = 50, current_user: User | None = Depends(get_current_user)):
    if not FEEDBACK_FILE.exists():
        return {"feedbacks": [], "total": 0}

    feedbacks = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                feedbacks.append(json.loads(line))

    feedbacks.reverse()  # Most recent first
    return {"feedbacks": feedbacks[:limit], "total": len(feedbacks)}


@router.get("/feedback/stats")
async def feedback_stats(current_user: User | None = Depends(get_current_user)):
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "positive": 0, "negative": 0, "neutral": 0}

    stats = {"total": 0, "positive": 0, "negative": 0, "neutral": 0}
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                stats["total"] += 1
                if entry["rating"] == 1:
                    stats["positive"] += 1
                elif entry["rating"] == -1:
                    stats["negative"] += 1
                else:
                    stats["neutral"] += 1

    return stats


class ErrorReportRequest(BaseModel):
    message_id: str
    session_id: str
    error_type: str = "incorrect_answer"  # incorrect_answer, hallucination, offensive, outdated, other
    description: str = ""
    expected_answer: str = ""


@router.post("/feedback/error-report")
async def submit_error_report(
    request: ErrorReportRequest,
    current_user: User | None = Depends(get_current_user),
):
    """오류 신고 제출."""
    report = {
        "id": str(uuid.uuid4()),
        "message_id": request.message_id,
        "session_id": request.session_id,
        "error_type": request.error_type,
        "description": request.description,
        "expected_answer": request.expected_answer,
        "reporter": current_user.username if current_user else "anonymous",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }

    # Append to error reports JSONL file
    report_path = Path(settings.data_dir) / "error_reports.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")

    logger.info("Error report submitted: %s by %s", request.error_type, report["reporter"])
    return {"status": "submitted", "report_id": report["id"]}


@router.get("/feedback/error-reports")
async def list_error_reports(
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
    limit: int = 50,
):
    """오류 신고 목록 조회 (관리자)."""
    report_path = Path(settings.data_dir) / "error_reports.jsonl"
    if not report_path.exists():
        return {"reports": [], "total": 0}

    reports = []
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    reports.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Most recent first
    reports.reverse()
    total = len(reports)
    return {"reports": reports[:limit], "total": total}
