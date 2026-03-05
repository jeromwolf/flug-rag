"""Expert answer correction → review → golden dataset accumulation."""

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth.dependencies import require_role
from auth.models import Role, User
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

CORRECTIONS_FILE = settings.data_dir / "corrections.jsonl"


class CorrectionSubmit(BaseModel):
    message_id: str
    session_id: str
    query: str
    original_answer: str
    corrected_answer: str
    correction_reason: str = ""
    category: str = "factual"  # factual, inference, negative, multi_hop
    difficulty: str = "medium"  # easy, medium, hard


class CorrectionReview(BaseModel):
    approved: bool
    reviewer_comment: str = ""


@router.post("/corrections")
async def submit_correction(
    request: CorrectionSubmit,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER, Role.EXPERT])),
):
    """Submit an answer correction for review."""
    correction_id = str(uuid.uuid4())
    entry = {
        "id": correction_id,
        "message_id": request.message_id,
        "session_id": request.session_id,
        "query": request.query,
        "original_answer": request.original_answer,
        "corrected_answer": request.corrected_answer,
        "correction_reason": request.correction_reason,
        "category": request.category,
        "difficulty": request.difficulty,
        "status": "pending",  # pending → approved → golden / rejected
        "submitted_by": current_user.username,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": None,
        "reviewed_at": None,
        "reviewer_comment": "",
        "golden_data_id": None,
    }

    CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Correction submitted by %s for message %s", current_user.username, request.message_id)
    return {"status": "submitted", "correction_id": correction_id}


@router.get("/corrections")
async def list_corrections(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER, Role.EXPERT])),
):
    """List corrections with optional status filter."""
    if not CORRECTIONS_FILE.exists():
        return {"corrections": [], "total": 0}

    corrections = []
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if status and entry.get("status") != status:
                        continue
                    corrections.append(entry)
                except json.JSONDecodeError:
                    continue

    corrections.reverse()
    total = len(corrections)
    return {"corrections": corrections[:limit], "total": total}


@router.get("/corrections/stats")
async def correction_stats(
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER, Role.EXPERT])),
):
    """Get correction statistics."""
    if not CORRECTIONS_FILE.exists():
        return {"total": 0, "pending": 0, "approved": 0, "rejected": 0}

    stats: dict[str, int] = {"total": 0, "pending": 0, "approved": 0, "rejected": 0}
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    stats["total"] += 1
                    status_val = entry.get("status", "pending")
                    if status_val in stats:
                        stats[status_val] += 1
                except json.JSONDecodeError:
                    continue

    return stats


@router.post("/corrections/{correction_id}/review")
async def review_correction(
    correction_id: str,
    request: CorrectionReview,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Review a correction — approve or reject. Approved corrections go to golden dataset."""
    entries = _read_all()
    found = False
    target_entry = None

    for entry in entries:
        if entry["id"] == correction_id:
            if entry["status"] != "pending":
                raise HTTPException(400, f"Cannot review correction in status: {entry['status']}")

            entry["status"] = "approved" if request.approved else "rejected"
            entry["reviewed_by"] = current_user.username
            entry["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            entry["reviewer_comment"] = request.reviewer_comment
            found = True
            target_entry = entry
            break

    if not found:
        raise HTTPException(404, "Correction not found")

    # If approved, add to golden dataset
    if request.approved and target_entry:
        try:
            from rag.golden_data import get_golden_data_manager
            manager = await get_golden_data_manager()
            golden_entry = await manager.add(
                question=target_entry["query"],
                answer=target_entry["corrected_answer"],
                category=target_entry.get("category", "factual"),
                evaluation_tag="expert_correction",
                created_by=target_entry["submitted_by"],
                source_message_id=target_entry.get("message_id", ""),
                source_session_id=target_entry.get("session_id", ""),
            )
            target_entry["golden_data_id"] = golden_entry.id
            logger.info("Correction %s approved → golden data %s", correction_id, golden_entry.id)
        except Exception as e:
            logger.warning("Failed to add to golden dataset: %s", e)

    _write_all(entries)
    return {
        "status": target_entry["status"],
        "correction_id": correction_id,
        "golden_data_id": target_entry.get("golden_data_id"),
    }


def _read_all() -> list[dict]:
    if not CORRECTIONS_FILE.exists():
        return []
    entries = []
    with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _write_all(entries: list[dict]):
    CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
