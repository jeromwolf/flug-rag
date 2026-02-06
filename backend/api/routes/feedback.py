"""Feedback endpoints for answer quality management."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from api.schemas import FeedbackRequest, FeedbackResponse
from config.settings import settings

router = APIRouter()

# Simple file-based feedback storage (SQLite in production)
FEEDBACK_FILE = settings.data_dir / "feedback.jsonl"


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    feedback_id = str(uuid.uuid4())

    entry = {
        "id": feedback_id,
        "message_id": request.message_id,
        "session_id": request.session_id,
        "rating": request.rating,
        "comment": request.comment,
        "corrected_answer": request.corrected_answer,
        "created_at": datetime.utcnow().isoformat(),
    }

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return FeedbackResponse(id=feedback_id, status="saved")


@router.get("/feedback")
async def list_feedback(limit: int = 50):
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
async def feedback_stats():
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
