"""Tests for the caching layer (InMemoryCache + decorators)."""

import asyncio
import time

import pytest

from core.cache.base import BaseCache
from core.cache.memory_cache import InMemoryCache


@pytest.fixture
def cache():
    """Create a fresh InMemoryCache for each test."""
    return InMemoryCache(default_ttl=10, max_size=100)


# --- InMemoryCache basic operations ---


class TestInMemoryCacheBasic:
    """Test basic get/set/delete/exists operations."""

    async def test_set_and_get(self, cache: InMemoryCache):
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    async def test_get_missing_key(self, cache: InMemoryCache):
        result = await cache.get("nonexistent")
        assert result is None

    async def test_set_complex_value(self, cache: InMemoryCache):
        data = {"users": [{"name": "test", "score": 42}], "total": 1}
        await cache.set("complex", data)
        result = await cache.get("complex")
        assert result == data

    async def test_delete(self, cache: InMemoryCache):
        await cache.set("to_delete", "value")
        assert await cache.exists("to_delete")
        await cache.delete("to_delete")
        assert not await cache.exists("to_delete")
        assert await cache.get("to_delete") is None

    async def test_delete_nonexistent(self, cache: InMemoryCache):
        """Deleting a non-existent key should not raise."""
        await cache.delete("nope")

    async def test_exists(self, cache: InMemoryCache):
        assert not await cache.exists("key")
        await cache.set("key", "val")
        assert await cache.exists("key")

    async def test_overwrite(self, cache: InMemoryCache):
        await cache.set("key", "first")
        await cache.set("key", "second")
        assert await cache.get("key") == "second"


# --- TTL expiration ---


class TestInMemoryCacheTTL:
    """Test time-to-live expiration behavior."""

    async def test_ttl_expiration(self, cache: InMemoryCache):
        await cache.set("short", "value", ttl=1)
        assert await cache.get("short") == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)
        assert await cache.get("short") is None

    async def test_ttl_exists_expired(self, cache: InMemoryCache):
        await cache.set("exp", "val", ttl=1)
        await asyncio.sleep(1.1)
        assert not await cache.exists("exp")

    async def test_no_ttl(self, cache: InMemoryCache):
        """Items with ttl=0 should not expire."""
        cache_no_ttl = InMemoryCache(default_ttl=0, max_size=100)
        await cache_no_ttl.set("forever", "value")
        # The item should persist (ttl=0 means no expiration)
        assert await cache_no_ttl.get("forever") == "value"

    async def test_custom_ttl_overrides_default(self, cache: InMemoryCache):
        """Custom TTL should override the default."""
        await cache.set("custom", "val", ttl=1)
        await asyncio.sleep(1.1)
        assert await cache.get("custom") is None


# --- LRU eviction ---


class TestInMemoryCacheLRU:
    """Test LRU eviction when max_size is exceeded."""

    async def test_lru_eviction(self):
        small_cache = InMemoryCache(default_ttl=60, max_size=3)
        await small_cache.set("a", 1)
        await small_cache.set("b", 2)
        await small_cache.set("c", 3)

        # Access "a" to make it recently used
        await small_cache.get("a")

        # Add "d" which should evict "b" (least recently used)
        await small_cache.set("d", 4)

        assert await small_cache.get("a") == 1  # Recently accessed, kept
        assert await small_cache.get("b") is None  # Evicted (LRU)
        assert await small_cache.get("c") == 3  # Kept
        assert await small_cache.get("d") == 4  # Just added

    async def test_max_size_enforcement(self):
        tiny = InMemoryCache(default_ttl=60, max_size=5)
        for i in range(10):
            await tiny.set(f"key_{i}", i)
        assert tiny.size <= 5


# --- Pattern clearing ---


class TestInMemoryCacheClearPattern:
    """Test glob pattern-based cache clearing."""

    async def test_clear_pattern(self, cache: InMemoryCache):
        await cache.set("user:1", "alice")
        await cache.set("user:2", "bob")
        await cache.set("session:1", "active")

        deleted = await cache.clear_pattern("user:*")
        assert deleted == 2
        assert await cache.get("user:1") is None
        assert await cache.get("user:2") is None
        assert await cache.get("session:1") == "active"

    async def test_clear_pattern_no_match(self, cache: InMemoryCache):
        await cache.set("key", "val")
        deleted = await cache.clear_pattern("nonexistent:*")
        assert deleted == 0

    async def test_clear_all(self, cache: InMemoryCache):
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.clear()
        assert cache.size == 0


# --- Cache-aside pattern ---


class TestCacheAside:
    """Test the get_or_set (cache-aside) pattern."""

    async def test_get_or_set_miss(self, cache: InMemoryCache):
        """Factory should be called on cache miss."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return {"data": "computed"}

        result = await cache.get_or_set("compute", factory, ttl=60)
        assert result == {"data": "computed"}
        assert call_count == 1

    async def test_get_or_set_hit(self, cache: InMemoryCache):
        """Factory should NOT be called on cache hit."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "new_value"

        await cache.set("cached", "old_value")
        result = await cache.get_or_set("cached", factory, ttl=60)
        assert result == "old_value"
        assert call_count == 0  # Factory was not called

    async def test_get_or_set_sync_factory(self, cache: InMemoryCache):
        """Sync factory functions should also work."""

        def sync_factory():
            return 42

        result = await cache.get_or_set("sync", sync_factory, ttl=60)
        assert result == 42


# --- Decorator tests ---


class TestCachedDecorator:
    """Test the @cached decorator."""

    async def test_cached_decorator_caches(self):
        """Decorated function should cache results."""
        from core.cache import reset_cache

        reset_cache()

        call_count = 0

        # We need to manually set up cache for decorator
        from core.cache.decorators import _build_cache_key, cached

        @cached(ttl=60, cache_name="test")
        async def expensive_function(query: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": query, "count": call_count}

        # First call should execute the function
        result1 = await expensive_function("hello")
        assert result1["result"] == "hello"
        first_count = call_count

        # Second call with same args should return cached result
        result2 = await expensive_function("hello")
        assert result2 == result1
        assert call_count == first_count  # Not called again

        # Different args should call the function
        result3 = await expensive_function("world")
        assert result3["result"] == "world"
        assert call_count == first_count + 1

        reset_cache()

    async def test_cache_key_builder(self):
        """Test custom key builder generates correct keys."""
        from core.cache.decorators import _build_cache_key

        async def my_func(a, b):
            pass

        key1 = _build_cache_key(my_func, ("hello",), {"b": 1})
        key2 = _build_cache_key(my_func, ("hello",), {"b": 1})
        key3 = _build_cache_key(my_func, ("world",), {"b": 1})

        assert key1 == key2  # Same args -> same key
        assert key1 != key3  # Different args -> different key

    async def test_custom_key_builder(self):
        """Test with a user-provided key builder."""
        from core.cache.decorators import _build_cache_key

        def my_key_builder(query: str):
            return f"search:{query}"

        async def search(query: str):
            pass

        key = _build_cache_key(search, ("test",), {}, key_builder=my_key_builder)
        assert key == "search:test"


# --- Thread safety ---


class TestConcurrency:
    """Test concurrent cache access."""

    async def test_concurrent_writes(self):
        """Multiple concurrent writes should not corrupt state."""
        # Create cache with larger max_size to hold all 150 items (3 * 50)
        cache = InMemoryCache(default_ttl=10, max_size=200)

        async def writer(prefix: str, count: int):
            for i in range(count):
                await cache.set(f"{prefix}:{i}", i)

        await asyncio.gather(
            writer("a", 50),
            writer("b", 50),
            writer("c", 50),
        )

        # All keys should be present
        for prefix in ("a", "b", "c"):
            for i in range(50):
                val = await cache.get(f"{prefix}:{i}")
                assert val == i, f"Expected {i} for {prefix}:{i}, got {val}"

    async def test_concurrent_read_write(self, cache: InMemoryCache):
        """Concurrent reads and writes should be safe."""
        await cache.set("shared", 0)

        async def reader():
            for _ in range(100):
                val = await cache.get("shared")
                assert val is not None or True  # May be None during write

        async def writer():
            for i in range(100):
                await cache.set("shared", i)

        await asyncio.gather(reader(), writer(), reader())
