"""
권한 신청/승인 워크플로우

SFR-002: 사용자 역할 변경 요청 및 관리자 승인/거절 프로세스
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class AccessRequest:
    """권한 신청"""
    id: str
    user_id: str
    username: str
    current_role: str
    requested_role: str
    reason: str
    status: str  # pending, approved, rejected
    reviewer_id: Optional[str] = None
    reviewer_comment: Optional[str] = None
    created_at: str = ""
    reviewed_at: Optional[str] = None


class AccessRequestManager:
    """권한 신청/승인 관리"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/access_requests.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _ensure_initialized(self):
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self._init_db()
                    self._initialized = True

    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS access_requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    current_role TEXT NOT NULL,
                    requested_role TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    reviewer_id TEXT,
                    reviewer_comment TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_user ON access_requests(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_status ON access_requests(status)"
            )
            await db.commit()

    async def create_request(
        self,
        user_id: str,
        username: str,
        current_role: str,
        requested_role: str,
        reason: str,
    ) -> AccessRequest:
        """권한 신청 생성"""
        await self._ensure_initialized()

        # 중복 신청 방지 (동일 사용자의 pending 신청이 있는지)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM access_requests WHERE user_id = ? AND status = 'pending'",
                (user_id,)
            ) as cursor:
                if await cursor.fetchone():
                    raise ValueError("이미 대기 중인 권한 신청이 있습니다")

        request = AccessRequest(
            id=str(uuid.uuid4()),
            user_id=user_id,
            username=username,
            current_role=current_role,
            requested_role=requested_role,
            reason=reason,
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO access_requests
                (id, user_id, username, current_role, requested_role, reason, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (request.id, request.user_id, request.username,
                 request.current_role, request.requested_role,
                 request.reason, request.status, request.created_at)
            )
            await db.commit()

        logger.info(f"Access request created: {request.username} requests {request.requested_role}")
        return request

    async def get_pending_requests(self) -> list[AccessRequest]:
        """대기 중인 신청 목록"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM access_requests WHERE status = 'pending' ORDER BY created_at ASC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [AccessRequest(**dict(row)) for row in rows]

    async def get_all_requests(self, limit: int = 50) -> list[AccessRequest]:
        """전체 신청 목록"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM access_requests ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [AccessRequest(**dict(row)) for row in rows]

    async def get_user_requests(self, user_id: str) -> list[AccessRequest]:
        """사용자 신청 이력"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM access_requests WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [AccessRequest(**dict(row)) for row in rows]

    async def review_request(
        self,
        request_id: str,
        reviewer_id: str,
        decision: Literal["approved", "rejected"],
        comment: str = "",
    ) -> AccessRequest:
        """신청 승인/거절"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 신청 조회
            async with db.execute(
                "SELECT * FROM access_requests WHERE id = ?",
                (request_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ValueError(f"Request not found: {request_id}")

                request = AccessRequest(**dict(row))
                if request.status != "pending":
                    raise ValueError(f"Request already {request.status}")

            # 상태 업데이트
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                """
                UPDATE access_requests
                SET status = ?, reviewer_id = ?, reviewer_comment = ?, reviewed_at = ?
                WHERE id = ?
                """,
                (decision, reviewer_id, comment, now, request_id)
            )
            await db.commit()

        # 승인된 경우 실제 역할 변경 (user_store에서)
        if decision == "approved":
            try:
                from auth.user_store import get_user_store
                from auth.models import Role
                role_map = {r.value: r for r in Role}
                new_role = role_map.get(request.requested_role)
                if new_role:
                    user_store = await get_user_store()
                    await user_store.update_role(request.user_id, new_role)
                    logger.info(f"Role updated: {request.username} -> {request.requested_role}")
            except Exception as e:
                logger.error(f"Failed to update role: {e}")

        request.status = decision
        request.reviewer_id = reviewer_id
        request.reviewer_comment = comment
        request.reviewed_at = now

        logger.info(f"Access request {decision}: {request_id} by {reviewer_id}")
        return request


# Singleton
_manager: Optional[AccessRequestManager] = None
_manager_lock = asyncio.Lock()


async def get_request_manager() -> AccessRequestManager:
    global _manager
    if _manager is not None:
        return _manager
    async with _manager_lock:
        if _manager is None:
            _manager = AccessRequestManager()
            await _manager._ensure_initialized()
    return _manager
