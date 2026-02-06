"""Performance profiling decorator and utilities."""

import asyncio
import functools
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of a profiled function call."""

    function_name: str
    module: str
    elapsed_ms: float
    memory_peak_kb: float = 0.0
    is_slow: bool = False
    timestamp: float = field(default_factory=time.time)


# Collected profile results (for reporting)
_profile_results: list[ProfileResult] = []
_max_results = 1000


def profile(
    threshold_ms: float = 500.0,
    track_memory: bool = False,
    log_always: bool = False,
):
    """Decorator for profiling async and sync functions.

    Logs a warning when execution exceeds the threshold.

    Usage::

        @profile(threshold_ms=200)
        async def slow_query(q: str) -> dict:
            ...

    Args:
        threshold_ms: Log a warning if execution takes longer than this (milliseconds).
        track_memory: If True, use tracemalloc to track peak memory.
        log_always: If True, log even fast operations at DEBUG level.
    """

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _run_profiled(func, args, kwargs, threshold_ms, track_memory, log_always, is_async=True)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _run_profiled_sync(func, args, kwargs, threshold_ms, track_memory, log_always)

        return async_wrapper if is_async else sync_wrapper

    return decorator


async def _run_profiled(
    func: Callable,
    args: tuple,
    kwargs: dict,
    threshold_ms: float,
    track_memory: bool,
    log_always: bool,
    is_async: bool = True,
) -> object:
    """Execute and profile an async function."""
    mem_start = None
    if track_memory:
        tracemalloc.start()
        mem_start = tracemalloc.get_traced_memory()

    start = time.perf_counter()
    try:
        result = await func(*args, **kwargs)
        return result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000

        memory_peak_kb = 0.0
        if track_memory and mem_start is not None:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_peak_kb = peak / 1024

        is_slow = elapsed_ms > threshold_ms
        _record_result(func, elapsed_ms, memory_peak_kb, is_slow)

        if is_slow:
            logger.warning(
                "SLOW: %s.%s took %.1fms (threshold: %.1fms)%s",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
                threshold_ms,
                f" | memory_peak: {memory_peak_kb:.1f}KB" if track_memory else "",
            )
        elif log_always:
            logger.debug(
                "PROFILE: %s.%s took %.1fms%s",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
                f" | memory_peak: {memory_peak_kb:.1f}KB" if track_memory else "",
            )


def _run_profiled_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    threshold_ms: float,
    track_memory: bool,
    log_always: bool,
) -> object:
    """Execute and profile a sync function."""
    mem_start = None
    if track_memory:
        tracemalloc.start()
        mem_start = tracemalloc.get_traced_memory()

    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000

        memory_peak_kb = 0.0
        if track_memory and mem_start is not None:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_peak_kb = peak / 1024

        is_slow = elapsed_ms > threshold_ms
        _record_result(func, elapsed_ms, memory_peak_kb, is_slow)

        if is_slow:
            logger.warning(
                "SLOW: %s.%s took %.1fms (threshold: %.1fms)%s",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
                threshold_ms,
                f" | memory_peak: {memory_peak_kb:.1f}KB" if track_memory else "",
            )
        elif log_always:
            logger.debug(
                "PROFILE: %s.%s took %.1fms%s",
                func.__module__,
                func.__qualname__,
                elapsed_ms,
                f" | memory_peak: {memory_peak_kb:.1f}KB" if track_memory else "",
            )


def _record_result(func: Callable, elapsed_ms: float, memory_peak_kb: float, is_slow: bool) -> None:
    """Record a profile result for later reporting."""
    global _profile_results
    result = ProfileResult(
        function_name=func.__qualname__,
        module=func.__module__,
        elapsed_ms=elapsed_ms,
        memory_peak_kb=memory_peak_kb,
        is_slow=is_slow,
    )
    _profile_results.append(result)
    if len(_profile_results) > _max_results:
        _profile_results = _profile_results[-_max_results:]


def get_profile_report(last_n: Optional[int] = None) -> dict:
    """Generate a performance report from collected profile data.

    Args:
        last_n: Limit to the last N results. Defaults to all.

    Returns:
        Dictionary with summary statistics and slow operations.
    """
    results = _profile_results if last_n is None else _profile_results[-last_n:]

    if not results:
        return {"total_calls": 0, "slow_calls": 0, "functions": {}}

    # Group by function
    by_function: dict[str, list[ProfileResult]] = {}
    for r in results:
        key = f"{r.module}.{r.function_name}"
        by_function.setdefault(key, []).append(r)

    function_stats = {}
    for key, entries in by_function.items():
        times = [e.elapsed_ms for e in entries]
        times.sort()
        function_stats[key] = {
            "call_count": len(entries),
            "avg_ms": sum(times) / len(times),
            "min_ms": times[0],
            "max_ms": times[-1],
            "p50_ms": times[len(times) // 2],
            "p95_ms": times[int(len(times) * 0.95)] if len(times) > 1 else times[0],
            "slow_count": sum(1 for e in entries if e.is_slow),
        }

    return {
        "total_calls": len(results),
        "slow_calls": sum(1 for r in results if r.is_slow),
        "functions": function_stats,
    }


def clear_profile_data() -> None:
    """Clear all collected profile data."""
    global _profile_results
    _profile_results.clear()
