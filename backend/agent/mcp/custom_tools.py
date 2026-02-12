"""
커스텀 도구 관리
SFR-018: 노코드 도구 정의, SQLite 저장, 런타임 등록
"""
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import httpx

from .tools.base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class CustomToolConfig:
    """커스텀 도구 설정"""
    id: str
    name: str
    description: str
    parameters_schema: str  # JSON string of parameters
    execution_type: str  # "api" or "python"
    api_url: str = ""
    api_method: str = "POST"
    api_headers: str = "{}"  # JSON string
    api_body_template: str = ""
    python_code: str = ""
    is_active: bool = True
    created_at: str = ""
    created_by: str = ""


class CustomToolExecutor(BaseTool):
    """커스텀 도구 런타임 래퍼"""

    def __init__(self, config: CustomToolConfig):
        self._config = config
        self._params = json.loads(config.parameters_schema) if config.parameters_schema else []

    def get_definition(self) -> ToolDefinition:
        parameters = []
        for p in self._params:
            param_type = getattr(ToolParamType, p.get("type", "STRING").upper(), ToolParamType.STRING)
            parameters.append(ToolParameter(
                name=p["name"],
                type=param_type,
                description=p.get("description", ""),
                required=p.get("required", True),
            ))
        return ToolDefinition(
            name=self._config.name,
            description=self._config.description,
            parameters=parameters,
            category="custom",
        )

    async def execute(self, **kwargs) -> ToolResult:
        if self._config.execution_type == "api":
            return await self._execute_api(**kwargs)
        else:
            return ToolResult(
                success=False,
                error="Python 코드 실행은 보안상 지원되지 않습니다.",
            )

    async def _execute_api(self, **kwargs) -> ToolResult:
        try:
            headers = json.loads(self._config.api_headers) if self._config.api_headers else {}
            headers.setdefault("Content-Type", "application/json")

            # Template substitution
            body = self._config.api_body_template
            for key, value in kwargs.items():
                body = body.replace(f"{{{{{key}}}}}", str(value))

            async with httpx.AsyncClient(timeout=30.0) as client:
                if self._config.api_method.upper() == "GET":
                    resp = await client.get(self._config.api_url, headers=headers, params=kwargs)
                else:
                    resp = await client.request(
                        self._config.api_method.upper(),
                        self._config.api_url,
                        headers=headers,
                        content=body if body else json.dumps(kwargs),
                    )

                if resp.status_code >= 400:
                    return ToolResult(
                        success=False,
                        error=f"API 오류 ({resp.status_code}): {resp.text[:500]}",
                    )

                return ToolResult(success=True, data=resp.text[:5000])

        except Exception as e:
            return ToolResult(success=False, error=f"API 호출 실패: {str(e)}")


class CustomToolStore:
    """커스텀 도구 SQLite 저장소"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/custom_tools.db")
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
                CREATE TABLE IF NOT EXISTS custom_tools (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    parameters_schema TEXT DEFAULT '[]',
                    execution_type TEXT DEFAULT 'api',
                    api_url TEXT DEFAULT '',
                    api_method TEXT DEFAULT 'POST',
                    api_headers TEXT DEFAULT '{}',
                    api_body_template TEXT DEFAULT '',
                    python_code TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    created_by TEXT DEFAULT ''
                )
            """)
            await db.commit()

    async def create(self, **kwargs) -> CustomToolConfig:
        await self._ensure_initialized()
        now = datetime.now(timezone.utc).isoformat()
        tool = CustomToolConfig(
            id=str(uuid.uuid4()),
            name=kwargs["name"],
            description=kwargs["description"],
            parameters_schema=kwargs.get("parameters_schema", "[]"),
            execution_type=kwargs.get("execution_type", "api"),
            api_url=kwargs.get("api_url", ""),
            api_method=kwargs.get("api_method", "POST"),
            api_headers=kwargs.get("api_headers", "{}"),
            api_body_template=kwargs.get("api_body_template", ""),
            python_code=kwargs.get("python_code", ""),
            created_at=now,
            created_by=kwargs.get("created_by", ""),
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO custom_tools
                   (id, name, description, parameters_schema, execution_type,
                    api_url, api_method, api_headers, api_body_template, python_code,
                    is_active, created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (tool.id, tool.name, tool.description, tool.parameters_schema,
                 tool.execution_type, tool.api_url, tool.api_method, tool.api_headers,
                 tool.api_body_template, tool.python_code, tool.created_at, tool.created_by),
            )
            await db.commit()
        return tool

    async def update(self, tool_id: str, **kwargs) -> CustomToolConfig:
        await self._ensure_initialized()
        allowed = {"name", "description", "parameters_schema", "execution_type",
                   "api_url", "api_method", "api_headers", "api_body_template",
                   "python_code", "is_active"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            raise ValueError("No valid fields to update")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [tool_id]

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(f"UPDATE custom_tools SET {set_clause} WHERE id = ?", values)
            if cursor.rowcount == 0:
                raise ValueError(f"Tool not found: {tool_id}")
            await db.commit()

            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM custom_tools WHERE id = ?", (tool_id,)) as cur:
                row = await cur.fetchone()
                return CustomToolConfig(**{**dict(row), "is_active": bool(row["is_active"])})

    async def delete(self, tool_id: str):
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM custom_tools WHERE id = ?", (tool_id,))
            if cursor.rowcount == 0:
                raise ValueError(f"Tool not found: {tool_id}")
            await db.commit()

    async def list_all(self) -> list[CustomToolConfig]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM custom_tools ORDER BY name") as cur:
                rows = await cur.fetchall()
                return [CustomToolConfig(**{**dict(r), "is_active": bool(r["is_active"])}) for r in rows]

    async def get(self, tool_id: str) -> CustomToolConfig | None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM custom_tools WHERE id = ?", (tool_id,)) as cur:
                row = await cur.fetchone()
                if row:
                    return CustomToolConfig(**{**dict(row), "is_active": bool(row["is_active"])})
        return None


# Singleton
_store: CustomToolStore | None = None
_store_lock = asyncio.Lock()


async def get_custom_tool_store() -> CustomToolStore:
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is None:
            _store = CustomToolStore()
            await _store._ensure_initialized()
    return _store
