"""Abstract base class for cache implementations."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class BaseCache(ABC):
    """Abstract cache interface supporting get/set/delete and cache-aside pattern."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key. Returns None if not found or expired."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with optional TTL in seconds."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from the cache."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists and has not expired."""
        ...

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern. Returns count of deleted keys."""
        ...

    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl: Optional[int] = None,
    ) -> Any:
        """Cache-aside pattern: return cached value or compute and store it.

        Args:
            key: Cache key.
            factory: Async callable that produces the value if not cached.
            ttl: Time-to-live in seconds.

        Returns:
            Cached or freshly computed value.
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Factory may be sync or async
        import asyncio

        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl=ttl)
        return value
