"""자연어 SQL 쿼리 도구 — 자연어를 SQL로 변환하여 실행."""

import logging
import re

import aiosqlite

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from config.settings import settings
from core.llm import BaseLLM, create_llm

logger = logging.getLogger(__name__)

SQL_SYSTEM_PROMPT = """You are an SQL expert. Given the database schema below, write a single SELECT query to answer the user's question.

Rules:
- Return ONLY the SQL query, nothing else.
- Do NOT include any explanation, markdown, or code fences.
- Only SELECT queries are allowed. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
- Use appropriate JOINs when the question involves multiple tables.
- Limit results to 50 rows maximum (add LIMIT 50 if needed).

Schema:
{schema}"""

# Dangerous SQL patterns to reject
DANGEROUS_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)

DB_PATHS = {
    "system": "system.db",
    "audit": "audit.db",
    "memory": "memory.db",
}

MAX_ROWS = 50


class NlToSqlTool(BaseTool):
    """자연어 SQL 쿼리 도구 — 자연어를 SQL로 변환하여 실행"""

    def __init__(self, llm: BaseLLM | None = None):
        self._llm = llm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="nl_to_sql",
            description="자연어 질문을 SQL로 변환하여 내부 데이터베이스를 조회합니다.",
            category="data",
            help_text=(
                "자연어 질문을 SQL로 변환하여 내부 데이터베이스를 조회합니다. "
                "SELECT 쿼리만 허용됩니다.\n"
                "파라미터:\n"
                "  - question: 자연어 질문 (필수)\n"
                "  - database: 조회할 DB (system/audit/memory, 기본값: system)\n"
                "  - execute: SQL 실행 여부 (기본값: true, false면 SQL만 반환)\n"
                "지원 DB:\n"
                "  - system: 시스템 데이터 (사용자, 설정 등)\n"
                "  - audit: 감사 로그 (접근 기록, 쿼리 이력)\n"
                "  - memory: 메모리 데이터 (대화 이력, 세션)"
            ),
            parameters=[
                ToolParameter(
                    name="question",
                    type=ToolParamType.STRING,
                    description="자연어 질문 (예: '오늘 접속한 사용자 수는?')",
                ),
                ToolParameter(
                    name="database",
                    type=ToolParamType.STRING,
                    description="조회할 데이터베이스",
                    required=False,
                    default="system",
                    enum=["system", "audit", "memory"],
                ),
                ToolParameter(
                    name="execute",
                    type=ToolParamType.BOOLEAN,
                    description="SQL 실행 여부 (false면 SQL만 반환)",
                    required=False,
                    default=True,
                ),
            ],
        )

    async def _get_db_path(self, database: str) -> str:
        """Resolve database file path."""
        filename = DB_PATHS.get(database, "system.db")
        return str(settings.data_dir / filename)

    async def _get_schema(self, db_path: str) -> str:
        """Extract schema from all tables in the database."""
        schema_parts = []

        async with aiosqlite.connect(db_path) as db:
            # Get all table names
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = await cursor.fetchall()

            for (table_name,) in tables:
                schema_parts.append(f"\nTABLE: {table_name}")

                # Get column info
                cursor = await db.execute(f"PRAGMA table_info({table_name})")
                columns = await cursor.fetchall()

                for col in columns:
                    # col: (cid, name, type, notnull, dflt_value, pk)
                    col_name = col[1]
                    col_type = col[2] or "TEXT"
                    pk = " PRIMARY KEY" if col[5] else ""
                    notnull = " NOT NULL" if col[3] else ""
                    schema_parts.append(f"  - {col_name} {col_type}{pk}{notnull}")

                # Get row count for context
                cursor = await db.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = (await cursor.fetchone())[0]
                schema_parts.append(f"  (rows: {count})")

        return "\n".join(schema_parts)

    def _validate_sql(self, sql: str) -> str | None:
        """Validate SQL query. Returns error message if invalid, None if OK."""
        sql_stripped = sql.strip().rstrip(";").strip()

        if not sql_stripped.upper().startswith("SELECT"):
            return "SELECT 쿼리만 허용됩니다."

        if DANGEROUS_PATTERNS.search(sql_stripped):
            return "보안 위반: 데이터 변경 쿼리는 허용되지 않습니다."

        return None

    def _clean_sql(self, raw: str) -> str:
        """Clean LLM output to extract pure SQL."""
        sql = raw.strip()

        # Remove markdown code fences
        if sql.startswith("```"):
            lines = sql.split("\n")
            # Remove first line (```sql or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            sql = "\n".join(lines).strip()

        # Remove trailing semicolons for consistency
        sql = sql.rstrip(";").strip()

        return sql

    async def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        if not question:
            return ToolResult(success=False, error="question 파라미터가 필요합니다.")

        database = kwargs.get("database", "system")
        should_execute = kwargs.get("execute", True)

        db_path = await self._get_db_path(database)

        # Check if DB file exists
        from pathlib import Path
        if not Path(db_path).exists():
            return ToolResult(
                success=False,
                error=f"데이터베이스 파일을 찾을 수 없습니다: {DB_PATHS.get(database, database)}",
            )

        try:
            # Step 1: Get schema
            schema = await self._get_schema(db_path)
            if not schema.strip():
                return ToolResult(
                    success=False,
                    error=f"데이터베이스 '{database}'에 테이블이 없습니다.",
                )

            # Step 2: Generate SQL via LLM
            system = SQL_SYSTEM_PROMPT.format(schema=schema)
            response = await self.llm.generate(
                prompt=question,
                system=system,
                temperature=0.1,
                max_tokens=512,
            )

            sql = self._clean_sql(response.content)
            if not sql:
                return ToolResult(
                    success=False,
                    error="SQL 쿼리를 생성하지 못했습니다.",
                )

            # Step 3: Validate SQL
            validation_error = self._validate_sql(sql)
            if validation_error:
                return ToolResult(
                    success=False,
                    error=validation_error,
                    metadata={"generated_sql": sql},
                )

            # If execute=False, return SQL only
            if not should_execute:
                return ToolResult(
                    success=True,
                    data={
                        "sql": sql,
                        "database": database,
                        "executed": False,
                    },
                    metadata={"question": question},
                )

            # Step 4: Execute SQL
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(sql)
                rows = await cursor.fetchmany(MAX_ROWS)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Convert to list of dicts
            results = [dict(zip(columns, row)) for row in rows]

            logger.info(
                "NL-to-SQL 실행 완료: DB=%s, rows=%d, SQL=%s",
                database, len(results), sql[:100],
            )

            return ToolResult(
                success=True,
                data={
                    "sql": sql,
                    "database": database,
                    "columns": columns,
                    "results": results,
                    "row_count": len(results),
                    "executed": True,
                },
                metadata={"question": question},
            )

        except aiosqlite.OperationalError as e:
            logger.error("SQL 실행 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"SQL 실행 오류: {e}",
                metadata={"generated_sql": locals().get("sql", "")},
            )
        except Exception as e:
            logger.error("NL-to-SQL 처리 중 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"NL-to-SQL 처리 중 오류가 발생했습니다: {e}",
            )
