"""Caching layer for flux-rag with Redis and in-memory fallback."""

import asyncio

from config.settings import settings

from .base import BaseCache
from .decorators import cache_invalidate, cached
from .memory_cache import InMemoryCache
from .redis_cache import RedisCache


async def create_cache() -> BaseCache:
    """Create a cache instance based on settings.

    Returns RedisCache if redis is available and configured,
    otherwise falls back to InMemoryCache.
    """
    if not settings.cache_enabled:
        return InMemoryCache()

    try:
        cache = RedisCache(url=settings.redis_url, max_connections=settings.redis_max_connections)
        if await cache.health_check():
            return cache
    except Exception:
        pass

    return InMemoryCache()


# Module-level singleton (initialized lazily)
_cache_instance: BaseCache | None = None
_cache_lock = asyncio.Lock()


async def get_cache() -> BaseCache:
    """Get or create the global cache singleton."""
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance
    async with _cache_lock:
        if _cache_instance is None:
            _cache_instance = await create_cache()
    return _cache_instance


def reset_cache() -> None:
    """Reset the cache singleton (for testing)."""
    global _cache_instance
    _cache_instance = None


__all__ = [
    "BaseCache",
    "RedisCache",
    "InMemoryCache",
    "cached",
    "cache_invalidate",
    "create_cache",
    "get_cache",
    "reset_cache",
]
