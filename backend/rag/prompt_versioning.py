"""
프롬프트 버전 관리 모듈

SFR-014: 프롬프트 변경 이력 추적 및 롤백
- 저장 시 자동 버전 증가
- 활성 버전 관리
- 버전별 비교 기능
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from core.db import AsyncSQLiteManager, create_async_singleton

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """프롬프트 버전"""
    id: str
    name: str  # prompt name (e.g., "rag_system")
    content: str
    version: int
    created_at: str
    created_by: str = ""
    is_active: bool = False
    change_note: str = ""


class PromptVersionManager(AsyncSQLiteManager):
    """프롬프트 버전 관리자"""

    def __init__(self, db_path: Optional[Path] = None):
        super().__init__(db_path or Path("data/prompt_versions.db"))

    async def _create_tables(self, db: aiosqlite.Connection):
        await db.execute("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT DEFAULT '',
                is_active INTEGER DEFAULT 0,
                change_note TEXT DEFAULT '',
                UNIQUE(name, version)
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompt_versions(name)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompts_active ON prompt_versions(name, is_active)"
        )
        await db.commit()

    async def save_version(
        self,
        name: str,
        content: str,
        created_by: str = "",
        change_note: str = "",
    ) -> PromptVersion:
        """새 버전 저장 (자동 버전 증가)."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("BEGIN EXCLUSIVE")
            try:
                # 현재 최신 버전 번호 조회
                async with db.execute(
                    "SELECT MAX(version) FROM prompt_versions WHERE name = ?",
                    (name,)
                ) as cursor:
                    row = await cursor.fetchone()
                    next_version = (row[0] or 0) + 1

                # 기존 활성 버전 비활성화
                await db.execute(
                    "UPDATE prompt_versions SET is_active = 0 WHERE name = ? AND is_active = 1",
                    (name,)
                )

                # 새 버전 저장
                version = PromptVersion(
                    id=str(uuid.uuid4()),
                    name=name,
                    content=content,
                    version=next_version,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    created_by=created_by,
                    is_active=True,
                    change_note=change_note,
                )

                await db.execute(
                    """
                    INSERT INTO prompt_versions
                    (id, name, content, version, created_at, created_by, is_active, change_note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (version.id, version.name, version.content, version.version,
                     version.created_at, version.created_by, 1, version.change_note)
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise

        logger.info(f"Prompt version saved: {name} v{next_version}")
        return version

    async def get_history(self, name: str, limit: int = 20) -> list[PromptVersion]:
        """프롬프트 버전 이력 조회."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM prompt_versions WHERE name = ? ORDER BY version DESC LIMIT ?",
                (name, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [PromptVersion(**{**dict(row), "is_active": bool(row["is_active"])}) for row in rows]

    async def get_all_prompts(self) -> list[PromptVersion]:
        """모든 프롬프트의 활성 버전 조회."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM prompt_versions WHERE is_active = 1 ORDER BY name"
            ) as cursor:
                rows = await cursor.fetchall()
                return [PromptVersion(**{**dict(row), "is_active": bool(row["is_active"])}) for row in rows]

    async def rollback(self, name: str, target_version: int) -> PromptVersion:
        """특정 버전으로 롤백 (해당 버전을 활성화)."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 대상 버전 조회
            async with db.execute(
                "SELECT * FROM prompt_versions WHERE name = ? AND version = ?",
                (name, target_version)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ValueError(f"Version {target_version} not found for prompt: {name}")

            # 모든 버전 비활성화
            await db.execute(
                "UPDATE prompt_versions SET is_active = 0 WHERE name = ?",
                (name,)
            )

            # 대상 버전 활성화
            await db.execute(
                "UPDATE prompt_versions SET is_active = 1 WHERE name = ? AND version = ?",
                (name, target_version)
            )
            await db.commit()

        target = PromptVersion(**{**dict(row), "is_active": True})
        logger.info(f"Prompt rolled back: {name} -> v{target_version}")
        return target

    async def get_version(self, name: str, version: int) -> Optional[PromptVersion]:
        """특정 버전 조회."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM prompt_versions WHERE name = ? AND version = ?",
                (name, version)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return PromptVersion(**{**dict(row), "is_active": bool(row["is_active"])})
        return None


# Singleton
get_version_manager = create_async_singleton(PromptVersionManager)
