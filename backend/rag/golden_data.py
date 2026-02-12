"""
골든 데이터 관리
SFR-017: 전문가 수정 답변 → 골든 데이터 → 벡터DB 업데이트
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class GoldenDataEntry:
    """골든 데이터 항목"""
    id: str
    question: str
    answer: str
    source_message_id: str = ""
    source_session_id: str = ""
    category: str = ""  # factual, procedure, regulation, etc.
    evaluation_tag: str = ""  # accurate, partial, inaccurate, hallucination
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    is_active: bool = True
    vectorstore_synced: bool = False


class GoldenDataManager:
    """골든 데이터 관리자"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/golden_data.db")
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
                CREATE TABLE IF NOT EXISTS golden_data (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source_message_id TEXT DEFAULT '',
                    source_session_id TEXT DEFAULT '',
                    category TEXT DEFAULT '',
                    evaluation_tag TEXT DEFAULT '',
                    created_by TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    vectorstore_synced INTEGER DEFAULT 0
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_golden_category ON golden_data(category)"
            )
            await db.commit()

    async def add(self, question: str, answer: str, **kwargs) -> GoldenDataEntry:
        """골든 데이터 추가 + 벡터DB 동기화."""
        await self._ensure_initialized()
        now = datetime.now(timezone.utc).isoformat()
        entry = GoldenDataEntry(
            id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            source_message_id=kwargs.get("source_message_id", ""),
            source_session_id=kwargs.get("source_session_id", ""),
            category=kwargs.get("category", ""),
            evaluation_tag=kwargs.get("evaluation_tag", ""),
            created_by=kwargs.get("created_by", ""),
            created_at=now,
            updated_at=now,
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO golden_data
                   (id, question, answer, source_message_id, source_session_id,
                    category, evaluation_tag, created_by, created_at, updated_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (entry.id, entry.question, entry.answer,
                 entry.source_message_id, entry.source_session_id,
                 entry.category, entry.evaluation_tag,
                 entry.created_by, entry.created_at, entry.updated_at),
            )
            await db.commit()

        # Sync to vectorstore
        await self._sync_to_vectorstore(entry)

        logger.info("Golden data added: %s by %s", entry.id[:8], entry.created_by)
        return entry

    async def update(self, entry_id: str, **kwargs) -> GoldenDataEntry:
        """골든 데이터 수정."""
        await self._ensure_initialized()
        allowed = {"question", "answer", "category", "evaluation_tag", "is_active"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        updates["vectorstore_synced"] = 0  # Mark for re-sync

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [entry_id]

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(f"UPDATE golden_data SET {set_clause} WHERE id = ?", values)
            if cursor.rowcount == 0:
                raise ValueError(f"Golden data not found: {entry_id}")
            await db.commit()

            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM golden_data WHERE id = ?", (entry_id,)) as cur:
                row = await cur.fetchone()
                entry = GoldenDataEntry(**{
                    **dict(row),
                    "is_active": bool(row["is_active"]),
                    "vectorstore_synced": bool(row["vectorstore_synced"]),
                })

        # Re-sync to vectorstore
        if entry.is_active:
            await self._sync_to_vectorstore(entry)

        return entry

    async def list_entries(self, limit: int = 50, category: str | None = None) -> list[GoldenDataEntry]:
        """골든 데이터 목록."""
        await self._ensure_initialized()
        conditions = ["is_active = 1"]
        params: list = []
        if category:
            conditions.append("category = ?")
            params.append(category)

        where = "WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM golden_data {where} ORDER BY updated_at DESC LIMIT ?",
                params + [limit],
            ) as cur:
                rows = await cur.fetchall()
                return [GoldenDataEntry(**{
                    **dict(r),
                    "is_active": bool(r["is_active"]),
                    "vectorstore_synced": bool(r["vectorstore_synced"]),
                }) for r in rows]

    async def _sync_to_vectorstore(self, entry: GoldenDataEntry):
        """골든 데이터를 벡터DB에 동기화 (우선순위 메타데이터 포함)."""
        try:
            from core.vectorstore import create_vectorstore
            from core.embeddings import create_embeddings

            vs = create_vectorstore()
            embeddings = create_embeddings()

            # Combine question + answer for embedding
            text = f"질문: {entry.question}\n답변: {entry.answer}"
            embedding = await embeddings.embed_query(text)

            metadata = {
                "source": "golden_data",
                "golden_id": entry.id,
                "category": entry.category,
                "evaluation_tag": entry.evaluation_tag,
                "created_by": entry.created_by,
                "is_golden": True,  # For boosting in retrieval
            }

            # Use ChromaDB's upsert method for add-or-update
            await asyncio.to_thread(
                vs._collection.upsert,
                ids=[f"golden_{entry.id}"],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
            )

            # Mark as synced
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE golden_data SET vectorstore_synced = 1 WHERE id = ?",
                    (entry.id,),
                )
                await db.commit()

            logger.info("Golden data synced to vectorstore: %s", entry.id[:8])

        except Exception as e:
            logger.warning("Golden data vectorstore sync failed: %s", e)


# Singleton
_manager: GoldenDataManager | None = None
_manager_lock = asyncio.Lock()


async def get_golden_data_manager() -> GoldenDataManager:
    global _manager
    if _manager is not None:
        return _manager
    async with _manager_lock:
        if _manager is None:
            _manager = GoldenDataManager()
            await _manager._ensure_initialized()
    return _manager
