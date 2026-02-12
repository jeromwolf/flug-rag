"""
AI 윤리 서약 관리 모듈

SFR-002: 사용자가 AI 서비스 이용 전 윤리 서약에 동의해야 함
- 서약 버전 관리
- 사용자별 동의 이력
- 미동의 시 API 차단
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# 현재 윤리 서약 내용 (버전 관리)
CURRENT_PLEDGE_VERSION = "1.0"
CURRENT_PLEDGE_CONTENT = """
# 한국가스기술공사 생성형 AI 윤리 서약

본인은 한국가스기술공사 생성형 AI 플랫폼을 사용함에 있어 다음 사항을 준수할 것을 서약합니다.

## 1. 정보 보안
- AI 시스템에 영업비밀, 개인정보, 보안등급 문서를 무단 입력하지 않겠습니다.
- AI 생성 결과물에 포함된 민감정보를 외부에 유출하지 않겠습니다.

## 2. 책임 있는 사용
- AI 생성 결과물을 그대로 공식 문서나 의사결정에 사용하지 않고, 반드시 전문가 검토를 거치겠습니다.
- AI 시스템의 한계를 인지하고, 결과물의 정확성을 독립적으로 검증하겠습니다.

## 3. 공정한 활용
- AI를 차별적이거나 비윤리적인 목적으로 사용하지 않겠습니다.
- AI 활용 결과가 특정 개인이나 집단에 부당한 영향을 미치지 않도록 주의하겠습니다.

## 4. 투명성
- AI를 활용하여 작성한 문서나 결과물에 대해 AI 활용 사실을 명시하겠습니다.
- AI 시스템의 오류나 문제를 발견한 경우 즉시 관리자에게 보고하겠습니다.

## 5. 규정 준수
- 회사의 정보보안 정책 및 AI 활용 가이드라인을 준수하겠습니다.
- 관련 법률 및 규정을 위반하는 용도로 AI를 사용하지 않겠습니다.

위 사항을 위반할 경우 사내 규정에 따른 조치를 받을 수 있음을 인지합니다.
""".strip()


@dataclass
class PledgeRecord:
    """서약 동의 기록"""
    id: str
    user_id: str
    version: str
    agreed_at: str
    ip_address: str = ""


class EthicsPledgeManager:
    """AI 윤리 서약 관리"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/ethics.db")
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
                CREATE TABLE IF NOT EXISTS ethics_pledges (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    agreed_at TEXT NOT NULL,
                    ip_address TEXT DEFAULT '',
                    UNIQUE(user_id, version)
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pledges_user ON ethics_pledges(user_id)"
            )
            await db.commit()

    async def has_agreed(self, user_id: str, version: str | None = None) -> bool:
        """사용자가 현재 버전 서약에 동의했는지 확인"""
        await self._ensure_initialized()
        version = version or CURRENT_PLEDGE_VERSION

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM ethics_pledges WHERE user_id = ? AND version = ?",
                (user_id, version)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def agree(self, user_id: str, ip_address: str = "") -> PledgeRecord:
        """서약 동의 기록"""
        await self._ensure_initialized()
        import uuid

        record = PledgeRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            version=CURRENT_PLEDGE_VERSION,
            agreed_at=datetime.now(timezone.utc).isoformat(),
            ip_address=ip_address,
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO ethics_pledges (id, user_id, version, agreed_at, ip_address)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record.id, record.user_id, record.version, record.agreed_at, record.ip_address)
            )
            await db.commit()

        logger.info(f"Ethics pledge agreed: user={user_id}, version={CURRENT_PLEDGE_VERSION}")
        return record

    async def get_user_pledges(self, user_id: str) -> list[PledgeRecord]:
        """사용자 서약 이력 조회"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ethics_pledges WHERE user_id = ? ORDER BY agreed_at DESC",
                (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [PledgeRecord(**dict(row)) for row in rows]


# Singleton
_manager: Optional[EthicsPledgeManager] = None
_manager_lock = asyncio.Lock()


async def get_pledge_manager() -> EthicsPledgeManager:
    global _manager
    if _manager is not None:
        return _manager
    async with _manager_lock:
        if _manager is None:
            _manager = EthicsPledgeManager()
            await _manager._ensure_initialized()
    return _manager
