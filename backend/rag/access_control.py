"""
폴더 기반 접근 제어 시스템

SFR-005: 지식 폴더별 접근 권한 관리
- 폴더 계층 구조
- 역할 기반 접근 제어 (ADMIN, MANAGER, USER, VIEWER)
- 부서별 권한 관리
- 문서 메타데이터 기반 필터링
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import aiosqlite
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeFolder:
    """지식 폴더"""
    id: str
    name: str
    parent_id: Optional[str]  # 상위 폴더 (계층 구조)
    owner_id: Optional[str]  # 소유자
    access_level: str  # "public", "department", "private", "admin"
    department: Optional[str]  # 부서 (access_level="department"일 때)
    created_at: str
    updated_at: str


@dataclass
class FolderPermission:
    """폴더 권한"""
    id: str
    folder_id: str
    user_id: Optional[str]  # 특정 사용자
    department: Optional[str]  # 또는 부서 전체
    permission_type: str  # "read", "write", "admin"


class AccessControlManager:
    """접근 제어 관리자"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/access_control.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """DB 초기화 보장"""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self.init_db()
                    self._initialized = True

    async def init_db(self):
        """데이터베이스 초기화"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            # 폴더 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_folders (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_id TEXT,
                    owner_id TEXT,
                    access_level TEXT NOT NULL,
                    department TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (parent_id) REFERENCES knowledge_folders(id) ON DELETE CASCADE
                )
            """)

            # 권한 테이블
            await db.execute("""
                CREATE TABLE IF NOT EXISTS folder_permissions (
                    id TEXT PRIMARY KEY,
                    folder_id TEXT NOT NULL,
                    user_id TEXT,
                    department TEXT,
                    permission_type TEXT NOT NULL,
                    FOREIGN KEY (folder_id) REFERENCES knowledge_folders(id) ON DELETE CASCADE
                )
            """)

            # 인덱스
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_folders_parent ON knowledge_folders(parent_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_folders_owner ON knowledge_folders(owner_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_folders_department ON knowledge_folders(department)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_permissions_folder ON folder_permissions(folder_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_permissions_user ON folder_permissions(user_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_permissions_department ON folder_permissions(department)"
            )

            await db.commit()

        logger.info(f"Access control database initialized at {self.db_path}")

    async def create_folder(
        self,
        name: str,
        parent_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        access_level: str = "private",
        department: Optional[str] = None
    ) -> KnowledgeFolder:
        """폴더 생성"""
        await self._ensure_initialized()

        valid_access_levels = {"public", "department", "private", "admin"}
        if access_level not in valid_access_levels:
            raise ValueError(f"Invalid access_level: {access_level}. Must be one of {valid_access_levels}")

        folder_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        folder = KnowledgeFolder(
            id=folder_id,
            name=name,
            parent_id=parent_id,
            owner_id=owner_id,
            access_level=access_level,
            department=department,
            created_at=now,
            updated_at=now
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """
                INSERT INTO knowledge_folders
                (id, name, parent_id, owner_id, access_level, department, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (folder.id, folder.name, folder.parent_id, folder.owner_id,
                 folder.access_level, folder.department, folder.created_at, folder.updated_at)
            )
            await db.commit()

        logger.info(f"Created folder: {folder.name} (id={folder.id}, access_level={folder.access_level})")
        return folder

    async def get_folder(self, folder_id: str) -> Optional[KnowledgeFolder]:
        """폴더 조회"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM knowledge_folders WHERE id = ?",
                (folder_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return KnowledgeFolder(**dict(row))
        return None

    async def list_folders(self, parent_id: Optional[str] = None) -> list[KnowledgeFolder]:
        """폴더 목록 조회"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            db.row_factory = aiosqlite.Row
            if parent_id is None:
                query = "SELECT * FROM knowledge_folders WHERE parent_id IS NULL ORDER BY name"
                params = ()
            else:
                query = "SELECT * FROM knowledge_folders WHERE parent_id = ? ORDER BY name"
                params = (parent_id,)

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [KnowledgeFolder(**dict(row)) for row in rows]

    async def update_folder(self, folder_id: str, **kwargs) -> KnowledgeFolder:
        """폴더 수정"""
        await self._ensure_initialized()

        folder = await self.get_folder(folder_id)
        if not folder:
            raise ValueError(f"Folder not found: {folder_id}")

        # 업데이트 가능한 필드
        allowed_fields = {"name", "access_level", "department", "owner_id", "parent_id"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return folder

        # 업데이트된 시간 추가
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        # SQL 생성
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [folder_id]

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                f"UPDATE knowledge_folders SET {set_clause} WHERE id = ?",
                values
            )
            await db.commit()

        # 업데이트된 폴더 반환
        updated_folder = await self.get_folder(folder_id)
        logger.info(f"Updated folder: {folder_id} - {updates}")
        return updated_folder

    async def delete_folder(self, folder_id: str):
        """폴더 삭제 (권한도 함께 삭제)"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            # 권한 먼저 삭제
            await db.execute(
                "DELETE FROM folder_permissions WHERE folder_id = ?",
                (folder_id,)
            )
            # 폴더 삭제 (CASCADE로 하위 폴더도 삭제됨)
            await db.execute(
                "DELETE FROM knowledge_folders WHERE id = ?",
                (folder_id,)
            )
            await db.commit()

        logger.info(f"Deleted folder: {folder_id}")

    async def set_permission(
        self,
        folder_id: str,
        user_id: Optional[str] = None,
        department: Optional[str] = None,
        permission_type: str = "read"
    ):
        """권한 설정"""
        await self._ensure_initialized()

        if not user_id and not department:
            raise ValueError("Either user_id or department must be provided")

        valid_permission_types = {"read", "write", "admin"}
        if permission_type not in valid_permission_types:
            raise ValueError(f"Invalid permission_type: {permission_type}. Must be one of {valid_permission_types}")

        permission_id = str(uuid.uuid4())

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")

            # Check for existing permission
            async with db.execute(
                """
                SELECT id FROM folder_permissions
                WHERE folder_id = ? AND
                      (user_id IS ? OR (user_id IS NULL AND ? IS NULL)) AND
                      (department IS ? OR (department IS NULL AND ? IS NULL))
                """,
                (folder_id, user_id, user_id, department, department)
            ) as cursor:
                existing = await cursor.fetchone()

            if existing:
                # Update existing permission
                await db.execute(
                    "UPDATE folder_permissions SET permission_type = ? WHERE id = ?",
                    (permission_type, existing[0])
                )
            else:
                # Insert new permission
                await db.execute(
                    """
                    INSERT INTO folder_permissions (id, folder_id, user_id, department, permission_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (permission_id, folder_id, user_id, department, permission_type)
                )
            await db.commit()

        logger.info(
            f"Set permission: folder={folder_id}, user={user_id}, "
            f"department={department}, type={permission_type}"
        )

    async def remove_permission(self, permission_id: str):
        """권한 삭제"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                "DELETE FROM folder_permissions WHERE id = ?",
                (permission_id,)
            )
            await db.commit()

        logger.info(f"Removed permission: {permission_id}")

    async def get_folder_permissions(self, folder_id: str) -> list[FolderPermission]:
        """폴더 권한 목록 조회"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM folder_permissions WHERE folder_id = ?",
                (folder_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [FolderPermission(**dict(row)) for row in rows]

    async def build_access_filter(
        self,
        user_id: str,
        user_role: str,
        user_department: Optional[str] = None
    ) -> Optional[dict]:
        """
        사용자 접근 권한에 따른 VectorStore 필터 생성

        Returns:
            ChromaDB-compatible metadata filter or None (no filter for admin)
        """
        await self._ensure_initialized()

        user_role = user_role.upper()

        # ADMIN: 모든 접근 가능
        if user_role == "ADMIN":
            return None

        accessible_folder_ids = set()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            db.row_factory = aiosqlite.Row

            # 1. Public 폴더는 모두 접근 가능
            async with db.execute(
                "SELECT id FROM knowledge_folders WHERE access_level = 'public'"
            ) as cursor:
                rows = await cursor.fetchall()
                accessible_folder_ids.update(row["id"] for row in rows)

            # 2. MANAGER: 자기 부서 폴더 접근 가능
            if user_role == "MANAGER" and user_department:
                async with db.execute(
                    "SELECT id FROM knowledge_folders WHERE department = ?",
                    (user_department,)
                ) as cursor:
                    rows = await cursor.fetchall()
                    accessible_folder_ids.update(row["id"] for row in rows)

            # 3. 소유한 폴더 접근 가능
            async with db.execute(
                "SELECT id FROM knowledge_folders WHERE owner_id = ?",
                (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                accessible_folder_ids.update(row["id"] for row in rows)

            # 4. 명시적으로 권한 부여된 폴더
            async with db.execute(
                """
                SELECT folder_id FROM folder_permissions
                WHERE user_id = ? OR department = ?
                """,
                (user_id, user_department)
            ) as cursor:
                rows = await cursor.fetchall()
                accessible_folder_ids.update(row["folder_id"] for row in rows)

        # 접근 가능한 폴더가 없으면 빈 결과 반환하도록 불가능한 필터
        if not accessible_folder_ids:
            return {"folder_id": "__no_access__"}

        # ChromaDB $or 필터 생성
        # 여러 folder_id 중 하나와 매칭
        if len(accessible_folder_ids) == 1:
            return {"folder_id": list(accessible_folder_ids)[0]}
        else:
            return {
                "$or": [
                    {"folder_id": folder_id}
                    for folder_id in accessible_folder_ids
                ]
            }


# 싱글톤 인스턴스
_manager: Optional[AccessControlManager] = None
_manager_lock = asyncio.Lock()


async def get_access_manager() -> AccessControlManager:
    """AccessControlManager 싱글톤 인스턴스 반환"""
    global _manager
    if _manager is not None:
        return _manager
    async with _manager_lock:
        if _manager is None:
            _manager = AccessControlManager()
            await _manager._ensure_initialized()
    return _manager
