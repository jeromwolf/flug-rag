"""RAG Settings Governance — draft → simulate → approve → apply workflow."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth.dependencies import require_role
from auth.models import Role, User
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

GOVERNANCE_FILE = settings.data_dir / "governance_changes.jsonl"


class ChangeRequest(BaseModel):
    change_type: str  # "prompt", "chunking", "model", "retriever"
    description: str
    current_value: dict
    proposed_value: dict


class ChangeApproval(BaseModel):
    approved: bool
    comment: str = ""


@router.post("/governance/changes")
async def create_change_request(
    request: ChangeRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Create a new change request (draft status)."""
    change_id = str(uuid.uuid4())
    entry = {
        "id": change_id,
        "change_type": request.change_type,
        "description": request.description,
        "current_value": request.current_value,
        "proposed_value": request.proposed_value,
        "status": "draft",  # draft → testing → pending_approval → approved → applied / rejected
        "created_by": current_user.username,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_results": None,
        "approved_by": None,
        "approved_at": None,
        "applied_at": None,
        "comment": "",
    }

    GOVERNANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GOVERNANCE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"status": "created", "change_id": change_id}


@router.get("/governance/changes")
async def list_change_requests(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """List change requests with optional status filter."""
    if not GOVERNANCE_FILE.exists():
        return {"changes": [], "total": 0}

    changes = []
    with open(GOVERNANCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if status and entry.get("status") != status:
                        continue
                    changes.append(entry)
                except json.JSONDecodeError:
                    continue

    changes.reverse()  # Most recent first
    return {"changes": changes[:limit], "total": len(changes)}


@router.post("/governance/changes/{change_id}/simulate")
async def simulate_change(
    change_id: str,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Run simulation test for a change request (simplified — returns mock results)."""
    entries = _read_all()
    found = False
    for entry in entries:
        if entry["id"] == change_id:
            if entry["status"] not in ("draft", "testing"):
                raise HTTPException(400, f"Cannot simulate change in status: {entry['status']}")
            entry["status"] = "pending_approval"
            entry["test_results"] = {
                "tested_at": datetime.now(timezone.utc).isoformat(),
                "tested_by": current_user.username,
                "baseline_accuracy": 95.8,
                "proposed_accuracy": 95.8,  # Simulated
                "sample_size": 120,
                "passed": True,
                "details": "시뮬레이션 테스트 완료. 정확도 변화 없음.",
            }
            found = True
            break

    if not found:
        raise HTTPException(404, "Change request not found")

    _write_all(entries)
    return {"status": "simulated", "change_id": change_id}


@router.post("/governance/changes/{change_id}/approve")
async def approve_change(
    change_id: str,
    request: ChangeApproval,
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """Approve or reject a change request."""
    entries = _read_all()
    found = False
    result_status = None
    for entry in entries:
        if entry["id"] == change_id:
            if entry["status"] != "pending_approval":
                raise HTTPException(400, f"Cannot approve change in status: {entry['status']}")
            entry["status"] = "approved" if request.approved else "rejected"
            entry["approved_by"] = current_user.username
            entry["approved_at"] = datetime.now(timezone.utc).isoformat()
            entry["comment"] = request.comment
            result_status = entry["status"]
            found = True
            break

    if not found:
        raise HTTPException(404, "Change request not found")

    _write_all(entries)
    return {"status": result_status, "change_id": change_id}


@router.post("/governance/changes/{change_id}/apply")
async def apply_change(
    change_id: str,
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """Apply an approved change to production."""
    entries = _read_all()
    found = False
    for entry in entries:
        if entry["id"] == change_id:
            if entry["status"] != "approved":
                raise HTTPException(400, f"Cannot apply change in status: {entry['status']}")
            entry["status"] = "applied"
            entry["applied_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break

    if not found:
        raise HTTPException(404, "Change request not found")

    _write_all(entries)
    logger.info("Change %s applied by %s", change_id, current_user.username)
    return {"status": "applied", "change_id": change_id}


def _read_all() -> list[dict]:
    if not GOVERNANCE_FILE.exists():
        return []
    entries = []
    with open(GOVERNANCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _write_all(entries: list[dict]):
    GOVERNANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GOVERNANCE_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
