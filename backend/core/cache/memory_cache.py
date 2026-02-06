"""In-memory cache with TTL and LRU eviction (fallback when Redis is unavailable)."""

import asyncio
import fnmatch
import time
from collections import OrderedDict
from typing import Any, Optional

from .base import BaseCache


class InMemoryCache(BaseCache):
    """Thread-safe in-memory cache with TTL support and LRU eviction.

    Intended as a development/testing fallback when Redis is not available.
    """

    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value, returning None if expired or missing."""
        async with self._lock:
            if key not in self._store:
                return None

            value, expires_at = self._store[key]
            if expires_at > 0 and time.time() > expires_at:
                del self._store[key]
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with optional TTL."""
        async with self._lock:
            ttl = ttl if ttl is not None else self._default_ttl
            expires_at = time.time() + ttl if ttl > 0 else 0
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key)
            self._evict_if_needed()

    async def delete(self, key: str) -> None:
        """Delete a key."""
        async with self._lock:
            self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Check if a key exists and has not expired."""
        async with self._lock:
            if key not in self._store:
                return False
            _, expires_at = self._store[key]
            if expires_at > 0 and time.time() > expires_at:
                del self._store[key]
                return False
            return True

    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern."""
        async with self._lock:
            to_delete = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
            for k in to_delete:
                del self._store[k]
            return len(to_delete)

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._store.clear()

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries when max_size is exceeded.

        Must be called while holding self._lock.
        """
        now = time.time()

        # First pass: remove expired entries
        expired_keys = [
            k for k, (_, exp) in self._store.items() if exp > 0 and now > exp
        ]
        for k in expired_keys:
            del self._store[k]

        # Second pass: LRU eviction if still over capacity
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)  # Remove oldest (least recently used)

    @property
    def size(self) -> int:
        """Current number of entries (including potentially expired)."""
        return len(self._store)
