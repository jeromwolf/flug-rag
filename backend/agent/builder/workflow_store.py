"""SQLite-backed workflow persistence store."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from core.db.base import AsyncSQLiteManager, create_async_singleton


class WorkflowStore(AsyncSQLiteManager):
    """Stores user-created workflows in SQLite."""

    def __init__(self):
        super().__init__(Path("data/workflows.db"))

    async def _create_tables(self, db):
        await db.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                nodes TEXT NOT NULL DEFAULT '[]',
                edges TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT DEFAULT NULL
            )
        """)
        await db.commit()

    async def create(self, name: str, description: str, nodes: list, edges: list, user_id: str | None = None) -> dict:
        workflow_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        async with self.get_connection() as db:
            await db.execute(
                """INSERT INTO workflows (id, name, description, nodes, edges, created_at, updated_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (workflow_id, name, description, json.dumps(nodes), json.dumps(edges), now, now, user_id),
            )
            await db.commit()
        return await self.get(workflow_id)

    async def get(self, workflow_id: str) -> dict | None:
        async with self.get_connection() as db:
            db.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            async with db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        row["nodes"] = json.loads(row["nodes"])
        row["edges"] = json.loads(row["edges"])
        return row

    async def list_all(self, user_id: str | None = None) -> list[dict]:
        async with self.get_connection() as db:
            db.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            if user_id:
                async with db.execute(
                    "SELECT id, name, description, created_at, updated_at, created_by, "
                    "(SELECT COUNT(*) FROM json_each(nodes)) as node_count "
                    "FROM workflows WHERE created_by = ? OR created_by IS NULL ORDER BY updated_at DESC",
                    (user_id,),
                ) as cur:
                    rows = await cur.fetchall()
            else:
                async with db.execute(
                    "SELECT id, name, description, created_at, updated_at, created_by, "
                    "(SELECT COUNT(*) FROM json_each(nodes)) as node_count "
                    "FROM workflows ORDER BY updated_at DESC"
                ) as cur:
                    rows = await cur.fetchall()
        return rows

    async def update(self, workflow_id: str, name: str, description: str, nodes: list, edges: list) -> dict | None:
        now = datetime.utcnow().isoformat()
        async with self.get_connection() as db:
            await db.execute(
                """UPDATE workflows SET name=?, description=?, nodes=?, edges=?, updated_at=?
                   WHERE id=?""",
                (name, description, json.dumps(nodes), json.dumps(edges), now, workflow_id),
            )
            await db.commit()
        return await self.get(workflow_id)

    async def delete(self, workflow_id: str) -> bool:
        async with self.get_connection() as db:
            async with db.execute("SELECT id FROM workflows WHERE id=?", (workflow_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return False
            await db.execute("DELETE FROM workflows WHERE id=?", (workflow_id,))
            await db.commit()
        return True


get_workflow_store = create_async_singleton(WorkflowStore)
