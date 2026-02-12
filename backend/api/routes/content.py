"""
콘텐츠 관리 API
SFR-009: 공지사항, FAQ, 만족도 설문
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user, require_role
from auth.models import Role, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/content", tags=["content"])

_db_path: Path | None = None
_initialized = False
_init_lock = asyncio.Lock()


async def _get_db_path() -> Path:
    global _db_path
    if _db_path is None:
        from config.settings import settings
        _db_path = settings.data_dir / "content.db"
        _db_path.parent.mkdir(parents=True, exist_ok=True)
    return _db_path


async def _ensure_db():
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        db_path = await _get_db_path()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS announcements (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    is_pinned INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    start_date TEXT,
                    end_date TEXT,
                    created_by TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS faq (
                    id TEXT PRIMARY KEY,
                    category TEXT DEFAULT '일반',
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sort_order INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_by TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS surveys (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    questions TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_by TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS survey_responses (
                    id TEXT PRIMARY KEY,
                    survey_id TEXT NOT NULL,
                    user_id TEXT DEFAULT '',
                    answers TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    FOREIGN KEY (survey_id) REFERENCES surveys(id)
                )
            """)
            await db.commit()
        _initialized = True


# ==================== Announcements ====================


class AnnouncementCreate(BaseModel):
    title: str = Field(..., max_length=200)
    content: str
    is_pinned: bool = False
    start_date: str | None = None
    end_date: str | None = None


class AnnouncementUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    is_pinned: bool | None = None
    is_active: bool | None = None
    start_date: str | None = None
    end_date: str | None = None


@router.get("/announcements")
async def list_announcements(
    current_user: User | None = Depends(get_current_user),
    active_only: bool = True,
):
    """공지사항 목록."""
    await _ensure_db()
    db_path = await _get_db_path()
    if not active_only:
        if not current_user or current_user.role not in (Role.ADMIN, Role.MANAGER):
            raise HTTPException(status_code=403, detail="Only admins can view inactive content")
    where = "WHERE is_active = 1" if active_only else ""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM announcements {where} ORDER BY is_pinned DESC, created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
            return {"announcements": [
                {**dict(r), "is_pinned": bool(r["is_pinned"]), "is_active": bool(r["is_active"])}
                for r in rows
            ]}


@router.post("/announcements")
async def create_announcement(
    request: AnnouncementCreate,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN, Role.MANAGER]))],
):
    """공지사항 생성."""
    await _ensure_db()
    db_path = await _get_db_path()
    now = datetime.now(timezone.utc).isoformat()
    aid = str(uuid.uuid4())
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO announcements
               (id, title, content, is_pinned, is_active, start_date, end_date, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
            (aid, request.title, request.content, int(request.is_pinned),
             request.start_date, request.end_date, current_user.username, now, now),
        )
        await db.commit()
    return {"status": "created", "id": aid}


ANNOUNCEMENT_UPDATE_FIELDS = {"title", "content", "priority", "is_pinned", "is_active", "start_date", "end_date"}


@router.put("/announcements/{ann_id}")
async def update_announcement(
    ann_id: str,
    request: AnnouncementUpdate,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN, Role.MANAGER]))],
):
    """공지사항 수정."""
    await _ensure_db()
    db_path = await _get_db_path()
    updates = {k: v for k, v in request.model_dump().items() if v is not None and k in ANNOUNCEMENT_UPDATE_FIELDS}
    if "is_pinned" in updates:
        updates["is_pinned"] = int(updates["is_pinned"])
    if "is_active" in updates:
        updates["is_active"] = int(updates["is_active"])
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [ann_id]

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(f"UPDATE announcements SET {set_clause} WHERE id = ?", values)
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다")
        await db.commit()
    return {"status": "updated", "id": ann_id}


@router.delete("/announcements/{ann_id}")
async def delete_announcement(
    ann_id: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """공지사항 삭제."""
    await _ensure_db()
    db_path = await _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("DELETE FROM announcements WHERE id = ?", (ann_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="공지사항을 찾을 수 없습니다")
        await db.commit()
    return {"status": "deleted"}


# ==================== FAQ ====================


class FAQCreate(BaseModel):
    category: str = "일반"
    question: str
    answer: str
    sort_order: int = 0


class FAQUpdate(BaseModel):
    category: str | None = None
    question: str | None = None
    answer: str | None = None
    sort_order: int | None = None
    is_active: bool | None = None


@router.get("/faq")
async def list_faq(
    current_user: User | None = Depends(get_current_user),
    category: str | None = None,
    active_only: bool = True,
):
    """FAQ 목록."""
    await _ensure_db()
    db_path = await _get_db_path()
    if not active_only:
        if not current_user or current_user.role not in (Role.ADMIN, Role.MANAGER):
            raise HTTPException(status_code=403, detail="Only admins can view inactive content")
    conditions = ["is_active = 1"] if active_only else []
    params: list = []
    if category:
        conditions.append("category = ?")
        params.append(category)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM faq {where} ORDER BY sort_order, created_at DESC", params
        ) as cur:
            rows = await cur.fetchall()
            return {"faq": [{**dict(r), "is_active": bool(r["is_active"])} for r in rows]}


@router.post("/faq")
async def create_faq(
    request: FAQCreate,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN, Role.MANAGER]))],
):
    """FAQ 생성."""
    await _ensure_db()
    db_path = await _get_db_path()
    now = datetime.now(timezone.utc).isoformat()
    fid = str(uuid.uuid4())
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO faq
               (id, category, question, answer, sort_order, is_active, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (fid, request.category, request.question, request.answer,
             request.sort_order, current_user.username, now, now),
        )
        await db.commit()
    return {"status": "created", "id": fid}


FAQ_UPDATE_FIELDS = {"question", "answer", "category", "sort_order", "is_active"}


@router.put("/faq/{faq_id}")
async def update_faq(
    faq_id: str,
    request: FAQUpdate,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN, Role.MANAGER]))],
):
    """FAQ 수정."""
    await _ensure_db()
    db_path = await _get_db_path()
    updates = {k: v for k, v in request.model_dump().items() if v is not None and k in FAQ_UPDATE_FIELDS}
    if "is_active" in updates:
        updates["is_active"] = int(updates["is_active"])
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [faq_id]

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(f"UPDATE faq SET {set_clause} WHERE id = ?", values)
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="FAQ를 찾을 수 없습니다")
        await db.commit()
    return {"status": "updated", "id": faq_id}


@router.delete("/faq/{faq_id}")
async def delete_faq(
    faq_id: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """FAQ 삭제."""
    await _ensure_db()
    db_path = await _get_db_path()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("DELETE FROM faq WHERE id = ?", (faq_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="FAQ를 찾을 수 없습니다")
        await db.commit()
    return {"status": "deleted"}


# ==================== Surveys ====================


class SurveyQuestion(BaseModel):
    text: str
    type: Literal["rating", "single_choice", "multi_choice", "text"] = "single_choice"
    options: list[str] = []


class SurveyCreate(BaseModel):
    title: str
    description: str = ""
    questions: list[SurveyQuestion]


class SurveyAnswer(BaseModel):
    question_index: int = Field(..., ge=0)
    value: str = Field(..., max_length=1000)


class SurveyResponse(BaseModel):
    answers: list[SurveyAnswer] = Field(..., max_length=100)


@router.get("/surveys")
async def list_surveys(
    current_user: User | None = Depends(get_current_user),
    active_only: bool = True,
):
    """설문 목록."""
    await _ensure_db()
    db_path = await _get_db_path()
    where = "WHERE is_active = 1" if active_only else ""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM surveys {where} ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
            return {"surveys": [
                {**dict(r), "is_active": bool(r["is_active"]),
                 "questions": json.loads(r["questions"])}
                for r in rows
            ]}


@router.post("/surveys")
async def create_survey(
    request: SurveyCreate,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """설문 생성."""
    await _ensure_db()
    db_path = await _get_db_path()
    now = datetime.now(timezone.utc).isoformat()
    sid = str(uuid.uuid4())
    questions_json = json.dumps([q.model_dump() for q in request.questions], ensure_ascii=False)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT INTO surveys
               (id, title, description, questions, is_active, created_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?)""",
            (sid, request.title, request.description, questions_json,
             current_user.username, now, now),
        )
        await db.commit()
    return {"status": "created", "id": sid}


@router.post("/surveys/{survey_id}/respond")
async def submit_survey_response(
    survey_id: str,
    request: SurveyResponse,
    current_user: User | None = Depends(get_current_user),
):
    """설문 응답 제출."""
    await _ensure_db()
    db_path = await _get_db_path()

    # C-01: Verify survey exists and is active
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT id, is_active FROM surveys WHERE id = ?", (survey_id,)) as cur:
            survey = await cur.fetchone()
            if not survey:
                raise HTTPException(status_code=404, detail="Survey not found")
            if not survey["is_active"]:
                raise HTTPException(status_code=400, detail="Survey is not active")

    rid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    answers_json = json.dumps([a.model_dump() for a in request.answers], ensure_ascii=False)
    user_id = current_user.id if current_user else ""

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO survey_responses (id, survey_id, user_id, answers, submitted_at) VALUES (?, ?, ?, ?, ?)",
            (rid, survey_id, user_id, answers_json, now),
        )
        await db.commit()
    return {"status": "submitted", "id": rid}


@router.get("/surveys/{survey_id}/results")
async def get_survey_results(
    survey_id: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN, Role.MANAGER]))],
):
    """설문 결과 집계."""
    await _ensure_db()
    db_path = await _get_db_path()

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        # Get survey
        async with db.execute("SELECT * FROM surveys WHERE id = ?", (survey_id,)) as cur:
            survey = await cur.fetchone()
            if not survey:
                raise HTTPException(status_code=404, detail="설문을 찾을 수 없습니다")

        # Get responses
        async with db.execute(
            "SELECT * FROM survey_responses WHERE survey_id = ?", (survey_id,)
        ) as cur:
            responses = await cur.fetchall()

    questions = json.loads(survey["questions"])
    total_responses = len(responses)

    # H-04: Optimize aggregation - parse JSON once and build index
    response_by_question: dict[int, list[str]] = {}
    for resp in responses:
        answers = json.loads(resp["answers"])
        for a in answers:
            qi = a.get("question_index")
            if qi is not None:
                if qi not in response_by_question:
                    response_by_question[qi] = []
                response_by_question[qi].append(str(a.get("value", "")))

    # Aggregate per question
    aggregated = []
    for qi, q in enumerate(questions):
        q_agg = {"question": q["text"], "type": q.get("type", "single_choice"), "responses": {}}
        if qi in response_by_question:
            for val in response_by_question[qi]:
                q_agg["responses"][val] = q_agg["responses"].get(val, 0) + 1
        aggregated.append(q_agg)

    return {
        "survey_id": survey_id,
        "title": survey["title"],
        "total_responses": total_responses,
        "questions": aggregated,
    }
