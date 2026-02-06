#!/usr/bin/env python3
"""Migration script to transfer data from SQLite to PostgreSQL."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.memory import ConversationMemory
from agent.memory_pg import PostgresConversationMemory


async def migrate(sqlite_path: str, postgres_dsn: str, batch_size: int = 100):
    """Migrate all sessions and messages from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to SQLite database file
        postgres_dsn: PostgreSQL connection string
        batch_size: Number of records to insert per batch
    """
    print(f"Starting migration from {sqlite_path} to PostgreSQL...")

    # Initialize both memory backends
    sqlite_memory = ConversationMemory(db_path=sqlite_path)
    postgres_memory = PostgresConversationMemory(dsn=postgres_dsn)

    # Ensure PostgreSQL tables are created
    await postgres_memory._ensure_initialized()

    # Read all sessions from SQLite
    print("Reading sessions from SQLite...")
    sessions = await sqlite_memory.get_sessions(limit=999999)
    print(f"Found {len(sessions)} sessions")

    # Migrate sessions
    print("Migrating sessions...")
    session_count = 0
    for session in sessions:
        # Get full session details
        session_data = await sqlite_memory.get_session(session["id"])
        if not session_data:
            continue

        # Insert into PostgreSQL
        async with postgres_memory._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO sessions (id, title, created_at, updated_at, metadata)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (id) DO NOTHING""",
                session_data["id"],
                session_data["title"],
                session_data["created_at"],
                session_data["updated_at"],
                session_data["metadata"],
            )
        session_count += 1

        if session_count % 10 == 0:
            print(f"  Migrated {session_count}/{len(sessions)} sessions")

    print(f"Completed session migration: {session_count} sessions")

    # Migrate messages for each session
    print("Migrating messages...")
    total_messages = 0

    for session in sessions:
        # Get all messages for this session
        messages = await sqlite_memory.get_history(session["id"], limit=999999)

        # Batch insert messages
        batch = []
        for msg in messages:
            batch.append((
                msg["id"],
                session["id"],
                msg["role"],
                msg["content"],
                msg["metadata"],
                msg["created_at"],
            ))

            if len(batch) >= batch_size:
                async with postgres_memory._pool.acquire() as conn:
                    await conn.executemany(
                        """INSERT INTO messages (id, session_id, role, content, metadata, created_at)
                           VALUES ($1, $2, $3, $4, $5, $6)
                           ON CONFLICT (id) DO NOTHING""",
                        batch,
                    )
                total_messages += len(batch)
                batch = []

        # Insert remaining messages
        if batch:
            async with postgres_memory._pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO messages (id, session_id, role, content, metadata, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6)
                       ON CONFLICT (id) DO NOTHING""",
                    batch,
                )
            total_messages += len(batch)

        if total_messages % 100 == 0:
            print(f"  Migrated {total_messages} messages...")

    print(f"Completed message migration: {total_messages} messages")

    # Verify counts
    print("\nVerification:")
    async with postgres_memory._pool.acquire() as conn:
        pg_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        pg_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")

    print(f"  SQLite:     {len(sessions)} sessions, {total_messages} messages")
    print(f"  PostgreSQL: {pg_sessions} sessions, {pg_messages} messages")

    if pg_sessions == len(sessions) and pg_messages == total_messages:
        print("\n✓ Migration completed successfully!")
    else:
        print("\n⚠ Warning: Count mismatch detected")

    # Clean up
    await postgres_memory.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Migrate SQLite conversation memory to PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        default="./data/memory.db",
        help="Path to SQLite database file (default: ./data/memory.db)",
    )
    parser.add_argument(
        "--postgres-dsn",
        default="postgresql://flux_rag:flux_rag@localhost:5432/flux_rag",
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for message insertion (default: 100)",
    )

    args = parser.parse_args()

    # Run migration
    asyncio.run(migrate(args.sqlite_path, args.postgres_dsn, args.batch_size))


if __name__ == "__main__":
    main()
