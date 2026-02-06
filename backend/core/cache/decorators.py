"""Caching decorators for transparent function-level caching."""

import functools
import hashlib
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _build_cache_key(func: Callable, args: tuple, kwargs: dict, key_builder: Optional[Callable] = None) -> str:
    """Build a deterministic cache key from function name and arguments.

    Args:
        func: The decorated function.
        args: Positional arguments.
        kwargs: Keyword arguments.
        key_builder: Optional custom key builder function.

    Returns:
        A string cache key.
    """
    if key_builder is not None:
        return key_builder(*args, **kwargs)

    # Auto-generate key from function module, name, and serialized args
    parts = [func.__module__, func.__qualname__]

    # Serialize arguments to a stable string
    try:
        arg_str = json.dumps({"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}})
    except (TypeError, ValueError):
        arg_str = str(args) + str(sorted(kwargs.items()))

    arg_hash = hashlib.md5(arg_str.encode()).hexdigest()[:16]
    return ":".join(parts) + ":" + arg_hash


def cached(
    ttl: int = 300,
    key_builder: Optional[Callable] = None,
    cache_name: str = "default",
):
    """Decorator for caching async function results.

    Usage::

        @cached(ttl=60)
        async def get_data(query: str) -> dict:
            ...

    Args:
        ttl: Time-to-live in seconds.
        key_builder: Optional callable(args, kwargs) -> str for custom keys.
        cache_name: Label for metrics tracking.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Import here to avoid circular imports
            from core.cache import get_cache

            try:
                cache = await get_cache()
            except Exception:
                # If cache is unavailable, just call the function
                return await func(*args, **kwargs)

            key = _build_cache_key(func, args, kwargs, key_builder)

            # Try to get from cache
            try:
                result = await cache.get(key)
                if result is not None:
                    _record_hit(cache_name)
                    return result
            except Exception:
                pass

            _record_miss(cache_name)

            # Call the actual function
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                await cache.set(key, result, ttl=ttl)
            except Exception:
                pass

            return result

        # Attach metadata for introspection
        wrapper._cache_ttl = ttl
        wrapper._cache_name = cache_name
        return wrapper

    return decorator


def cache_invalidate(pattern: str):
    """Decorator that invalidates cache keys matching a pattern after function execution.

    Usage::

        @cache_invalidate("documents:*")
        async def upload_document(...):
            ...

    Args:
        pattern: Glob pattern for keys to invalidate.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Invalidate cache after successful execution
            try:
                from core.cache import get_cache

                cache = await get_cache()
                deleted = await cache.clear_pattern(pattern)
                if deleted > 0:
                    logger.debug("Invalidated %d cache keys matching '%s'", deleted, pattern)
            except Exception as e:
                logger.debug("Cache invalidation failed for pattern '%s': %s", pattern, e)

            return result

        return wrapper

    return decorator


def _record_hit(cache_name: str) -> None:
    """Record a cache hit metric (no-op if monitoring disabled)."""
    try:
        from monitoring.metrics import cache_hits_total

        cache_hits_total.labels(cache_name=cache_name).inc()
    except Exception:
        pass


def _record_miss(cache_name: str) -> None:
    """Record a cache miss metric (no-op if monitoring disabled)."""
    try:
        from monitoring.metrics import cache_misses_total

        cache_misses_total.labels(cache_name=cache_name).inc()
    except Exception:
        pass
