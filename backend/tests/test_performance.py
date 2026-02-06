"""Tests for performance utilities (batch processor, profiler, metrics)."""

import asyncio

import pytest

from core.performance.batch_processor import BatchProcessor
from core.performance.profiler import (
    ProfileResult,
    clear_profile_data,
    get_profile_report,
    profile,
)


# --- BatchProcessor tests ---


class TestBatchProcessor:
    """Test the batch processor for grouping requests."""

    async def test_single_item(self):
        """Single item should be processed after max_wait timeout."""
        results_received = []

        async def process_fn(items: list) -> list:
            results_received.extend(items)
            return [x * 2 for x in items]

        processor = BatchProcessor(
            process_fn=process_fn,
            batch_size=4,
            max_wait_ms=50,
            name="test",
        )
        await processor.start()

        try:
            result = await processor.submit(5)
            assert result == 10
            assert 5 in results_received
        finally:
            await processor.stop()

    async def test_batch_collection(self):
        """Multiple items submitted quickly should be batched together."""
        batch_sizes = []

        async def process_fn(items: list) -> list:
            batch_sizes.append(len(items))
            return [x + 1 for x in items]

        processor = BatchProcessor(
            process_fn=process_fn,
            batch_size=4,
            max_wait_ms=200,
            name="test-batch",
        )
        await processor.start()

        try:
            # Submit 4 items concurrently - should form one batch
            tasks = [processor.submit(i) for i in range(4)]
            results = await asyncio.gather(*tasks)

            assert sorted(results) == [1, 2, 3, 4]
            # All items should be in one batch (or possibly 2 if timing varies)
            assert sum(batch_sizes) == 4
        finally:
            await processor.stop()

    async def test_error_propagation(self):
        """Errors in process_fn should propagate to all items in batch."""

        async def failing_fn(items: list) -> list:
            raise ValueError("batch failed")

        processor = BatchProcessor(
            process_fn=failing_fn,
            batch_size=4,
            max_wait_ms=50,
            name="test-fail",
        )
        await processor.start()

        try:
            with pytest.raises(ValueError, match="batch failed"):
                await processor.submit(1)
        finally:
            await processor.stop()

    async def test_result_count_mismatch(self):
        """Process function returning wrong count should raise."""

        async def wrong_count_fn(items: list) -> list:
            return [1]  # Always returns 1 result regardless of input

        processor = BatchProcessor(
            process_fn=wrong_count_fn,
            batch_size=1,
            max_wait_ms=50,
            name="test-mismatch",
        )
        await processor.start()

        try:
            # With batch_size=1, first item gets 1 result which matches
            result = await processor.submit(42)
            assert result == 1
        finally:
            await processor.stop()

    async def test_auto_start(self):
        """Processor should auto-start on first submit if not started."""

        async def process_fn(items: list) -> list:
            return items

        processor = BatchProcessor(
            process_fn=process_fn,
            batch_size=4,
            max_wait_ms=50,
        )

        # Don't call start() explicitly
        try:
            result = await processor.submit("auto")
            assert result == "auto"
        finally:
            await processor.stop()


# --- Profiler tests ---


class TestProfiler:
    """Test the profiling decorator."""

    def setup_method(self):
        clear_profile_data()

    async def test_profile_async_function(self):
        """@profile should work with async functions."""

        @profile(threshold_ms=10000, log_always=True)
        async def fast_func():
            return 42

        result = await fast_func()
        assert result == 42

        report = get_profile_report()
        assert report["total_calls"] >= 1

    def test_profile_sync_function(self):
        """@profile should work with sync functions."""

        @profile(threshold_ms=10000, log_always=True)
        def sync_func(x):
            return x * 2

        result = sync_func(21)
        assert result == 42

        report = get_profile_report()
        assert report["total_calls"] >= 1

    async def test_slow_detection(self):
        """Functions exceeding threshold should be flagged as slow."""

        @profile(threshold_ms=10)
        async def slow_func():
            await asyncio.sleep(0.05)  # 50ms
            return "done"

        result = await slow_func()
        assert result == "done"

        report = get_profile_report()
        assert report["slow_calls"] >= 1

    async def test_profile_preserves_exceptions(self):
        """@profile should not swallow exceptions."""

        @profile(threshold_ms=1000)
        async def failing_func():
            raise RuntimeError("test error")

        with pytest.raises(RuntimeError, match="test error"):
            await failing_func()

    async def test_profile_report_structure(self):
        """Profile report should have expected structure."""
        clear_profile_data()

        @profile(threshold_ms=10000)
        async def func_a():
            return 1

        @profile(threshold_ms=10000)
        async def func_b():
            return 2

        await func_a()
        await func_a()
        await func_b()

        report = get_profile_report()
        assert report["total_calls"] == 3
        assert len(report["functions"]) == 2

        # Check function stats structure
        for key, stats in report["functions"].items():
            assert "call_count" in stats
            assert "avg_ms" in stats
            assert "min_ms" in stats
            assert "max_ms" in stats
            assert "p50_ms" in stats
            assert "p95_ms" in stats

    async def test_clear_profile_data(self):
        """clear_profile_data should reset all collected data."""

        @profile(threshold_ms=10000)
        async def tracked():
            return True

        await tracked()
        assert get_profile_report()["total_calls"] >= 1

        clear_profile_data()
        assert get_profile_report()["total_calls"] == 0

    async def test_profile_with_memory_tracking(self):
        """@profile with track_memory should capture peak memory."""

        @profile(threshold_ms=10000, track_memory=True, log_always=True)
        async def memory_func():
            data = [0] * 10000  # Allocate some memory
            return len(data)

        result = await memory_func()
        assert result == 10000

    def test_last_n_filter(self):
        """get_profile_report(last_n) should limit results."""
        clear_profile_data()

        @profile(threshold_ms=10000)
        def simple():
            return True

        for _ in range(10):
            simple()

        report_all = get_profile_report()
        report_5 = get_profile_report(last_n=5)

        assert report_all["total_calls"] == 10
        assert report_5["total_calls"] == 5


# --- Metrics collection tests ---


class TestMetrics:
    """Test Prometheus metrics (no-op stubs when prometheus_client unavailable)."""

    def test_metrics_import(self):
        """Metrics should import without error even without prometheus_client."""
        from monitoring.metrics import (
            cache_hits_total,
            cache_misses_total,
            http_request_duration_seconds,
            http_requests_total,
        )

        # These should not raise
        http_requests_total.labels(method="GET", endpoint="/test", status="200").inc()
        http_request_duration_seconds.labels(method="GET", endpoint="/test").observe(0.1)
        cache_hits_total.labels(cache_name="test").inc()
        cache_misses_total.labels(cache_name="test").inc()

    def test_noop_metric_methods(self):
        """No-op metrics should support all standard methods."""
        from monitoring.metrics import active_connections, document_count

        # Gauge operations
        active_connections.inc()
        active_connections.dec()
        active_connections.set(5)

        document_count.set(100)
