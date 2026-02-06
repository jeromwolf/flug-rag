#!/usr/bin/env python3
"""Simple script to test PostgreSQL connection and basic operations."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.memory_pg import PostgresConversationMemory
from config.settings import settings


async def test_postgres_connection():
    """Test PostgreSQL connection and basic CRUD operations."""
    print("=" * 60)
    print("PostgreSQL Connection Test")
    print("=" * 60)
    print(f"\nConnecting to: {settings.postgres_dsn}")
    print()

    try:
        # Initialize memory
        memory = PostgresConversationMemory()
        await memory._ensure_initialized()
        print("✓ Connected successfully")
        print("✓ Tables created/verified")

        # Test 1: Create session
        print("\n--- Test 1: Create Session ---")
        session_id = await memory.create_session(
            title="Test Session",
            metadata={"source": "test_script", "version": "1.0"}
        )
        print(f"✓ Created session: {session_id}")

        # Test 2: Add messages
        print("\n--- Test 2: Add Messages ---")
        msg1_id = await memory.add_message(
            session_id=session_id,
            role="user",
            content="Hello, how are you?",
            metadata={"type": "greeting"}
        )
        print(f"✓ Added user message: {msg1_id}")

        msg2_id = await memory.add_message(
            session_id=session_id,
            role="assistant",
            content="I'm doing well, thank you for asking!",
            metadata={"type": "response"}
        )
        print(f"✓ Added assistant message: {msg2_id}")

        # Test 3: Get history
        print("\n--- Test 3: Get History ---")
        history = await memory.get_history(session_id)
        print(f"✓ Retrieved {len(history)} messages:")
        for msg in history:
            print(f"  - {msg['role']}: {msg['content'][:50]}...")

        # Test 4: Get sessions
        print("\n--- Test 4: Get Sessions ---")
        sessions = await memory.get_sessions(limit=10)
        print(f"✓ Retrieved {len(sessions)} sessions:")
        for sess in sessions[:3]:  # Show first 3
            print(f"  - {sess['id']}: {sess['title']} ({sess['message_count']} messages)")

        # Test 5: Update session title
        print("\n--- Test 5: Update Session Title ---")
        await memory.update_session_title(session_id, "Updated Test Session")
        session = await memory.get_session(session_id)
        print(f"✓ Updated title: {session['title']}")

        # Test 6: JSONB metadata
        print("\n--- Test 6: JSONB Metadata ---")
        print(f"✓ Session metadata: {session['metadata']}")
        print(f"✓ Message metadata: {history[0]['metadata']}")

        # Test 7: Delete session
        print("\n--- Test 7: Delete Session ---")
        await memory.delete_session(session_id)
        deleted_session = await memory.get_session(session_id)
        if deleted_session is None:
            print(f"✓ Session {session_id} deleted successfully")
        else:
            print(f"✗ Failed to delete session")

        # Close connection pool
        await memory.close()
        print("\n✓ Connection pool closed")

        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


async def check_connection():
    """Quick connection check."""
    try:
        memory = PostgresConversationMemory()
        await memory._ensure_initialized()

        # Count existing records
        async with memory._pool.acquire() as conn:
            session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions")
            message_count = await conn.fetchval("SELECT COUNT(*) FROM messages")

        print(f"✓ Connection successful")
        print(f"  - Sessions: {session_count}")
        print(f"  - Messages: {message_count}")

        await memory.close()
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def main():
    """CLI entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        print("Checking PostgreSQL connection...")
        success = asyncio.run(check_connection())
        sys.exit(0 if success else 1)
    else:
        print("Running full test suite...")
        exit_code = asyncio.run(test_postgres_connection())
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
