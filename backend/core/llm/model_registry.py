"""
LLM 모델 레지스트리

SFR-014: 관리자가 모델을 등록/수정/삭제 가능
- SQLite 기반 모델 설정 저장
- 헬스체크 API
- 핫 리로드
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiosqlite
import httpx

from core.db import AsyncSQLiteManager, create_async_singleton

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """LLM 모델 설정"""
    id: str
    name: str  # Display name
    provider: str  # vllm, ollama, openai, anthropic
    model_id: str  # Actual model identifier
    base_url: str = ""
    api_key: str = ""
    is_active: bool = True
    is_default: bool = False
    parameters: str = "{}"  # JSON string of model parameters
    created_at: str = ""
    updated_at: str = ""


class ModelRegistry(AsyncSQLiteManager):
    """LLM 모델 레지스트리"""

    def __init__(self, db_path: Optional[Path] = None):
        super().__init__(db_path or Path("data/model_registry.db"))

    async def _create_tables(self, db: aiosqlite.Connection):
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                base_url TEXT DEFAULT '',
                api_key TEXT DEFAULT '',
                is_active INTEGER DEFAULT 1,
                is_default INTEGER DEFAULT 0,
                parameters TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider)"
        )
        await db.commit()

    async def register_model(
        self,
        name: str,
        provider: str,
        model_id: str,
        base_url: str = "",
        api_key: str = "",
        parameters: str = "{}",
    ) -> ModelConfig:
        """모델 등록"""
        await self._ensure_initialized()
        now = datetime.now(timezone.utc).isoformat()

        model = ModelConfig(
            id=str(uuid.uuid4()),
            name=name,
            provider=provider,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            parameters=parameters,
            created_at=now,
            updated_at=now,
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO models
                (id, name, provider, model_id, base_url, api_key, is_active, is_default, parameters, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (model.id, model.name, model.provider, model.model_id,
                 model.base_url, model.api_key, 1, 0, model.parameters,
                 model.created_at, model.updated_at)
            )
            await db.commit()

        logger.info(f"Model registered: {name} ({provider}/{model_id})")
        return model

    async def list_models(self, active_only: bool = True) -> list[ModelConfig]:
        """모델 목록 조회"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM models"
            if active_only:
                query += " WHERE is_active = 1"
            query += " ORDER BY name"

            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                return [ModelConfig(**{
                    **dict(row),
                    "is_active": bool(row["is_active"]),
                    "is_default": bool(row["is_default"]),
                }) for row in rows]

    async def update_model(self, model_id: str, **kwargs) -> ModelConfig:
        """모델 수정"""
        await self._ensure_initialized()
        allowed = {"name", "provider", "model_id", "base_url", "api_key", "is_active", "parameters"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [model_id]

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Check existence BEFORE update
            async with db.execute("SELECT * FROM models WHERE id = ?", (model_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ValueError(f"Model not found: {model_id}")

            await db.execute(f"UPDATE models SET {set_clause} WHERE id = ?", values)
            await db.commit()

            # Re-fetch updated model
            async with db.execute("SELECT * FROM models WHERE id = ?", (model_id,)) as cursor:
                row = await cursor.fetchone()
                return ModelConfig(**{
                    **dict(row),
                    "is_active": bool(row["is_active"]),
                    "is_default": bool(row["is_default"]),
                })

    async def delete_model(self, model_id: str):
        """모델 삭제"""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM models WHERE id = ?", (model_id,))
            if cursor.rowcount == 0:
                raise ValueError(f"Model not found: {model_id}")
            await db.commit()

    async def test_model(self, model_id: str) -> dict:
        """모델 헬스체크"""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM models WHERE id = ?", (model_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    raise ValueError(f"Model not found: {model_id}")

        model = ModelConfig(**{**dict(row), "is_active": bool(row["is_active"]), "is_default": bool(row["is_default"])})

        # Validate base_url to prevent SSRF
        if model.base_url:
            parsed = urlparse(model.base_url)
            if parsed.scheme not in ("http", "https"):
                return {"status": "error", "error": "Invalid URL scheme"}
            if not parsed.hostname:
                return {"status": "error", "error": "Invalid URL"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if model.provider in ("vllm", "openai"):
                    url = f"{model.base_url}/models"
                    headers = {"Authorization": f"Bearer {model.api_key}"} if model.api_key else {}
                    resp = await client.get(url, headers=headers)
                    return {"status": "ok" if resp.status_code == 200 else "error", "code": resp.status_code}
                elif model.provider == "ollama":
                    url = f"{model.base_url}/api/tags"
                    resp = await client.get(url)
                    return {"status": "ok" if resp.status_code == 200 else "error", "code": resp.status_code}
                elif model.provider == "anthropic":
                    return {"status": "ok", "note": "Anthropic health check not available via API"}
                else:
                    return {"status": "unknown", "note": f"Unknown provider: {model.provider}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton
get_model_registry = create_async_singleton(ModelRegistry)
