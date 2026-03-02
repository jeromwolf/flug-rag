"""Bookmark routes for saving favorite messages."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.dependencies import get_current_user

router = APIRouter()

BOOKMARKS_FILE = Path(__file__).parent.parent.parent / "data" / "bookmarks.json"


def _load_bookmarks() -> list[dict]:
    if BOOKMARKS_FILE.exists():
        return json.loads(BOOKMARKS_FILE.read_text(encoding="utf-8"))
    return []


def _save_bookmarks(bookmarks: list[dict]) -> None:
    BOOKMARKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BOOKMARKS_FILE.write_text(
        json.dumps(bookmarks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class BookmarkCreate(BaseModel):
    message_id: str
    session_id: str
    content: str
    role: str = "assistant"
    note: str = ""


@router.post("/bookmarks")
async def add_bookmark(
    req: BookmarkCreate,
    current_user=Depends(get_current_user),
):
    user_id = current_user.id if current_user else "anonymous"
    bookmarks = _load_bookmarks()

    # 중복 방지
    if any(
        b["message_id"] == req.message_id and b["user_id"] == user_id
        for b in bookmarks
    ):
        return {"status": "already_exists"}

    bookmark = {
        "message_id": req.message_id,
        "session_id": req.session_id,
        "content": req.content[:500],
        "role": req.role,
        "note": req.note,
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
    }
    bookmarks.append(bookmark)
    _save_bookmarks(bookmarks)
    return {"status": "created", "bookmark": bookmark}


@router.get("/bookmarks")
async def list_bookmarks(current_user=Depends(get_current_user)):
    user_id = current_user.id if current_user else "anonymous"
    bookmarks = _load_bookmarks()
    user_bookmarks = [b for b in bookmarks if b["user_id"] == user_id]
    return {
        "bookmarks": sorted(
            user_bookmarks, key=lambda b: b["created_at"], reverse=True
        )
    }


@router.delete("/bookmarks/{message_id}")
async def remove_bookmark(
    message_id: str,
    current_user=Depends(get_current_user),
):
    user_id = current_user.id if current_user else "anonymous"
    bookmarks = _load_bookmarks()
    bookmarks = [
        b
        for b in bookmarks
        if not (b["message_id"] == message_id and b["user_id"] == user_id)
    ]
    _save_bookmarks(bookmarks)
    return {"status": "deleted"}
