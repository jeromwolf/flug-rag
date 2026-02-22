"""
비동기 SQLite 매니저 베이스 클래스.

aiosqlite 기반 매니저의 공통 패턴 추출:
- 지연 초기화 (double-checked locking)
- 테이블 자동 생성
- 싱글턴 팩토리 헬퍼
"""

import asyncio
import logging
from abc import abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


class AsyncSQLiteManager:
    """aiosqlite 기반 매니저 베이스 클래스.

    사용법:
        class MyManager(AsyncSQLiteManager):
            def __init__(self):
                super().__init__(Path("data/my.db"))

            async def _create_tables(self, db: aiosqlite.Connection):
                await db.execute("CREATE TABLE IF NOT EXISTS ...")
                await db.commit()
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """지연 초기화 (double-checked locking)."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("PRAGMA journal_mode=WAL")
                        await db.execute("PRAGMA synchronous=NORMAL")
                        await db.execute("PRAGMA busy_timeout=5000")
                        await self._create_tables(db)
                    self._initialized = True

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection with standard pragmas applied.

        Usage:
            async with self.get_connection() as db:
                await db.execute("SELECT ...")

        Applies WAL mode and busy timeout automatically.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA busy_timeout=5000")
            yield db

    @abstractmethod
    async def _create_tables(self, db: aiosqlite.Connection):
        """테이블 스키마 정의. 서브클래스에서 구현.

        Args:
            db: aiosqlite connection (commit은 구현부에서 호출).
        """


def create_async_singleton(manager_class: type, *args, **kwargs):
    """비동기 싱글턴 팩토리 생성 헬퍼.

    사용법:
        get_my_manager = create_async_singleton(MyManager)
        manager = await get_my_manager()  # 항상 같은 인스턴스 반환

    Args:
        manager_class: AsyncSQLiteManager 서브클래스
        *args, **kwargs: 생성자 인자

    Returns:
        async callable that returns the singleton instance.
    """
    _instance: Optional[manager_class] = None
    _lock = asyncio.Lock()

    async def get_instance() -> manager_class:
        nonlocal _instance
        if _instance is not None:
            return _instance
        async with _lock:
            if _instance is None:
                _instance = manager_class(*args, **kwargs)
                await _instance._ensure_initialized()
        return _instance

    def reset():
        """싱글턴 리셋 (테스트용)."""
        nonlocal _instance
        _instance = None

    get_instance.reset = reset
    return get_instance
