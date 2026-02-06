"""Tests for PostgreSQL conversation memory."""

import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.memory_factory import create_memory
from agent.memory_pg import PostgresConversationMemory


class TestPostgresConversationMemory:
    """Test PostgresConversationMemory with mocked asyncpg."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dsn = "postgresql://test:test@localhost:5432/test"

    @patch("agent.memory_pg.asyncpg.create_pool")
    def test_initialization(self, mock_create_pool):
        """Test memory initialization."""
        mock_pool = MagicMock()
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        assert memory.dsn == self.dsn
        assert memory._pool is None
        assert memory._initialized is False

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_ensure_initialized_creates_pool(self, mock_create_pool):
        """Test that _ensure_initialized creates pool and tables."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        # Verify pool creation
        mock_create_pool.assert_called_once_with(dsn=self.dsn, min_size=2, max_size=10)
        assert memory._initialized is True

        # Verify table creation calls
        assert mock_conn.execute.call_count == 3  # sessions, messages, index

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_create_session(self, mock_create_pool):
        """Test session creation."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        session_id = await memory.create_session(title="Test Session", metadata={"key": "value"})

        # Verify UUID format
        assert len(session_id) == 36
        assert "-" in session_id

        # Verify execute was called with correct parameters
        call_args = mock_conn.execute.call_args_list[-1]
        assert "INSERT INTO sessions" in call_args[0][0]

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_add_message(self, mock_create_pool):
        """Test message addition."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        session_id = "test-session-id"
        msg_id = await memory.add_message(
            session_id=session_id,
            role="user",
            content="Hello",
            metadata={"timestamp": "2024-01-01"},
        )

        # Verify message ID
        assert len(msg_id) == 36

        # Verify two execute calls (insert message + update session)
        assert mock_conn.execute.call_count >= 2

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_get_history(self, mock_create_pool):
        """Test retrieving conversation history."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        # Mock database rows
        mock_rows = [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Hello",
                "metadata": {"key": "value"},
                "created_at": datetime(2024, 1, 1, 12, 0, 0),
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": "Hi there",
                "metadata": {},
                "created_at": datetime(2024, 1, 1, 12, 1, 0),
            },
        ]
        mock_conn.fetch.return_value = mock_rows

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        history = await memory.get_history("session-id", limit=10)

        # Verify results (reversed to chronological order)
        assert len(history) == 2
        assert history[0]["id"] == "msg-2"  # Reversed
        assert history[0]["role"] == "assistant"
        assert isinstance(history[0]["created_at"], str)

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_get_sessions(self, mock_create_pool):
        """Test retrieving session list."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        # Mock session rows
        mock_rows = [
            {
                "id": "session-1",
                "title": "Test Session",
                "created_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 2),
                "metadata": {"key": "value"},
                "message_count": 5,
            }
        ]
        mock_conn.fetch.return_value = mock_rows

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        sessions = await memory.get_sessions(limit=50, offset=0)

        # Verify results
        assert len(sessions) == 1
        assert sessions[0]["id"] == "session-1"
        assert sessions[0]["message_count"] == 5

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_get_session(self, mock_create_pool):
        """Test retrieving a single session."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        # Mock session row
        mock_row = {
            "id": "session-1",
            "title": "Test",
            "created_at": datetime(2024, 1, 1),
            "updated_at": datetime(2024, 1, 1),
            "metadata": {},
        }
        mock_conn.fetchrow.return_value = mock_row

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        session = await memory.get_session("session-1")

        # Verify result
        assert session is not None
        assert session["id"] == "session-1"
        assert session["title"] == "Test"

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_get_session_not_found(self, mock_create_pool):
        """Test retrieving non-existent session."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        mock_conn.fetchrow.return_value = None

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        session = await memory.get_session("nonexistent")

        assert session is None

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_update_session_title(self, mock_create_pool):
        """Test updating session title."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        await memory.update_session_title("session-1", "New Title")

        # Verify execute was called
        call_args = mock_conn.execute.call_args_list[-1]
        assert "UPDATE sessions" in call_args[0][0]

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_delete_session(self, mock_create_pool):
        """Test deleting a session."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        await memory.delete_session("session-1")

        # Verify two deletes (messages + sessions)
        assert mock_conn.execute.call_count >= 2

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_clear_all(self, mock_create_pool):
        """Test clearing all data."""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()

        await memory.clear_all()

        # Verify two deletes
        assert mock_conn.execute.call_count >= 2

    @pytest.mark.asyncio
    @patch("agent.memory_pg.asyncpg.create_pool", new_callable=AsyncMock)
    async def test_close(self, mock_create_pool):
        """Test pool closure."""
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mock_create_pool.return_value = mock_pool

        memory = PostgresConversationMemory(dsn=self.dsn)
        await memory._ensure_initialized()
        await memory.close()

        mock_pool.close.assert_called_once()


class TestMemoryFactory(unittest.TestCase):
    """Test memory factory function."""

    @patch("agent.memory_factory.settings")
    def test_factory_returns_sqlite(self, mock_settings):
        """Test factory returns SQLite backend."""
        mock_settings.database_backend = "sqlite"

        memory = create_memory()

        from agent.memory import ConversationMemory
        self.assertIsInstance(memory, ConversationMemory)

    @patch("agent.memory_factory.settings")
    def test_factory_returns_postgres(self, mock_settings):
        """Test factory returns PostgreSQL backend."""
        mock_settings.database_backend = "postgres"
        mock_settings.postgres_dsn = "postgresql://test:test@localhost/test"

        memory = create_memory()

        from agent.memory_pg import PostgresConversationMemory
        self.assertIsInstance(memory, PostgresConversationMemory)


if __name__ == "__main__":
    # Run async tests
    import asyncio

    def run_async_test(coro):
        """Helper to run async test methods."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
