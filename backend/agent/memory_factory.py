"""Factory function to create the appropriate memory backend."""

from config.settings import settings


def create_memory():
    """Create and return the appropriate memory backend based on settings."""
    if settings.database_backend == "postgres":
        from agent.memory_pg import PostgresConversationMemory
        return PostgresConversationMemory()
    else:
        from agent.memory import ConversationMemory
        return ConversationMemory()
