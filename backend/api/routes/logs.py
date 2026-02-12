"""
로그 관리 API
SFR-010: 접속 로그, 질의 이력, 작업 로그 검색
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Annotated, Literal

import aiosqlite
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from auth.dependencies import require_role
from auth.models import Role, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/logs", tags=["logs"])


class LogEntry(BaseModel):
    timestamp: str
    user_id: str = ""
    username: str = ""
    action: str = ""
    resource: str = ""
    details: str = ""
    ip_address: str = ""


class LogSearchResponse(BaseModel):
    logs: list[LogEntry]
    total: int
    page: int
    page_size: int


class QueryLogEntry(BaseModel):
    timestamp: str
    session_id: str
    content: str
    role: str


class QueryLogResponse(BaseModel):
    queries: list[QueryLogEntry]
    total: int
    page: int
    page_size: int


async def _get_audit_db():
    from config.settings import settings
    return settings.data_dir / "audit.db"


async def _get_memory_db():
    from config.settings import settings
    return settings.data_dir / "memory.db"


@router.get("/access", response_model=LogSearchResponse)
async def search_access_logs(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
    username: str | None = None,
    action: str | None = None,
    ip_address: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """접속 로그 검색 (날짜/사용자/IP 필터)."""
    db_path = await _get_audit_db()

    conditions = []
    params = []

    if username:
        conditions.append("username LIKE ?")
        params.append(f"%{username}%")
    if action:
        conditions.append("action = ?")
        params.append(action)
    if ip_address:
        conditions.append("ip_address LIKE ?")
        params.append(f"%{ip_address}%")
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    offset = (page - 1) * page_size

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row

            # Count total
            async with db.execute(f"SELECT COUNT(*) as cnt FROM audit_log {where}", params) as cur:
                row = await cur.fetchone()
                total = row["cnt"] if row else 0

            # Fetch page
            async with db.execute(
                f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params + [page_size, offset]
            ) as cur:
                rows = await cur.fetchall()
                logs = [LogEntry(
                    timestamp=row["timestamp"],
                    user_id=row["user_id"] or "",
                    username=row["username"] or "",
                    action=row["action"] or "",
                    resource=row["resource"] or "",
                    details=row["details"] or "",
                    ip_address=row["ip_address"] or "",
                ) for row in rows]

        return LogSearchResponse(logs=logs, total=total, page=page, page_size=page_size)
    except Exception as e:
        logger.warning("Audit log query failed: %s", e)
        return LogSearchResponse(logs=[], total=0, page=page, page_size=page_size)


@router.get("/queries", response_model=QueryLogResponse)
async def search_query_logs(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN, Role.MANAGER]))],
    keyword: str | None = None,
    session_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """질의 이력 검색."""
    db_path = await _get_memory_db()

    conditions = ["role = 'user'"]
    params = []

    if keyword:
        conditions.append("content LIKE ?")
        params.append(f"%{keyword}%")
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    if start_date:
        conditions.append("created_at >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("created_at <= ?")
        params.append(end_date)

    where = "WHERE " + " AND ".join(conditions)
    offset = (page - 1) * page_size

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(f"SELECT COUNT(*) as cnt FROM messages {where}", params) as cur:
                row = await cur.fetchone()
                total = row["cnt"] if row else 0

            async with db.execute(
                f"SELECT * FROM messages {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [page_size, offset]
            ) as cur:
                rows = await cur.fetchall()
                queries = [QueryLogEntry(
                    timestamp=row["created_at"] or "",
                    session_id=row["session_id"] or "",
                    content=row["content"] or "",
                    role=row["role"] or "",
                ) for row in rows]

        return QueryLogResponse(queries=queries, total=total, page=page, page_size=page_size)
    except Exception as e:
        logger.warning("Query log search failed: %s", e)
        return QueryLogResponse(queries=[], total=0, page=page, page_size=page_size)


@router.get("/operations")
async def get_operation_logs(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
    action_filter: Literal[
        "all", "DOCUMENT_UPLOAD", "DOCUMENT_DELETE", "DOCUMENT_ACCESS",
        "SETTINGS_CHANGE", "ROLE_CHANGE", "USER_CREATED", "USER_DEACTIVATED"
    ] = "all",
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """작업 로그 (문서 업로드/삭제, 설정 변경 등)."""
    db_path = await _get_audit_db()

    op_actions = [
        "DOCUMENT_UPLOAD", "DOCUMENT_DELETE", "DOCUMENT_ACCESS",
        "SETTINGS_CHANGE", "ROLE_CHANGE", "USER_CREATED", "USER_DEACTIVATED"
    ]

    if action_filter == "all":
        placeholders = ",".join("?" * len(op_actions))
        where = f"WHERE action IN ({placeholders})"
        params = op_actions
    else:
        where = "WHERE action = ?"
        params = [action_filter]

    offset = (page - 1) * page_size

    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row

            async with db.execute(f"SELECT COUNT(*) as cnt FROM audit_log {where}", params) as cur:
                row = await cur.fetchone()
                total = row["cnt"] if row else 0

            async with db.execute(
                f"SELECT * FROM audit_log {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params + [page_size, offset]
            ) as cur:
                rows = await cur.fetchall()
                logs = [LogEntry(
                    timestamp=row["timestamp"],
                    user_id=row["user_id"] or "",
                    username=row["username"] or "",
                    action=row["action"] or "",
                    resource=row["resource"] or "",
                    details=row["details"] or "",
                    ip_address=row["ip_address"] or "",
                ) for row in rows]

        return {"logs": [l.model_dump() for l in logs], "total": total, "page": page, "page_size": page_size}
    except Exception as e:
        logger.warning("Operation log query failed: %s", e)
        return {"logs": [], "total": 0, "page": page, "page_size": page_size}
