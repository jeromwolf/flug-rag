"""Redis-based cache implementation using redis.asyncio."""

import json
import logging
from typing import Any, Optional

from .base import BaseCache

logger = logging.getLogger(__name__)

DEFAULT_TTL = 300  # 5 minutes
KEY_PREFIX = "flux-rag:"


class RedisCache(BaseCache):
    """Redis cache with connection pooling and JSON serialization."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        max_connections: int = 20,
        default_ttl: int = DEFAULT_TTL,
        key_prefix: str = KEY_PREFIX,
    ):
        self._url = url
        self._max_connections = max_connections
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix
        self._redis = None

    async def _get_client(self):
        """Lazy-initialize the Redis client with connection pool."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    self._url,
                    max_connections=self._max_connections,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
            except ImportError:
                raise ImportError(
                    "redis package required for RedisCache. Install with: pip install redis"
                )
        return self._redis

    def _make_key(self, key: str) -> str:
        """Prepend the key prefix."""
        return f"{self._key_prefix}{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string."""
        return json.dumps(value, ensure_ascii=False, default=str)

    def _deserialize(self, raw: Optional[str]) -> Optional[Any]:
        """Deserialize JSON string back to Python object."""
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        try:
            client = await self._get_client()
            raw = await client.get(self._make_key(key))
            return self._deserialize(raw)
        except Exception as e:
            logger.warning("Redis GET error for key=%s: %s", key, e)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with optional TTL."""
        try:
            client = await self._get_client()
            ttl = ttl if ttl is not None else self._default_ttl
            serialized = self._serialize(value)
            if ttl > 0:
                await client.setex(self._make_key(key), ttl, serialized)
            else:
                await client.set(self._make_key(key), serialized)
        except Exception as e:
            logger.warning("Redis SET error for key=%s: %s", key, e)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        try:
            client = await self._get_client()
            await client.delete(self._make_key(key))
        except Exception as e:
            logger.warning("Redis DELETE error for key=%s: %s", key, e)

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            client = await self._get_client()
            return bool(await client.exists(self._make_key(key)))
        except Exception as e:
            logger.warning("Redis EXISTS error for key=%s: %s", key, e)
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern."""
        try:
            client = await self._get_client()
            full_pattern = self._make_key(pattern)
            deleted = 0
            async for key in client.scan_iter(match=full_pattern, count=100):
                await client.delete(key)
                deleted += 1
            return deleted
        except Exception as e:
            logger.warning("Redis CLEAR_PATTERN error for pattern=%s: %s", pattern, e)
            return 0

    async def health_check(self) -> bool:
        """Check if Redis is reachable."""
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception as e:
            logger.warning("Redis health check failed: %s", e)
            return False

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
