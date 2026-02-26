"""Simple in-memory rate limiter for login endpoint."""

from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    """Token-bucket style rate limiter keyed by client identifier (IP, username, etc.).

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts allowed within *window_seconds*.
    window_seconds : float
        Sliding window duration in seconds.
    max_keys : int
        Maximum number of tracked keys before forced cleanup.
    """

    def __init__(self, max_attempts: int = 5, window_seconds: float = 60.0, max_keys: int = 10000) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        # key -> list of timestamps
        self._attempts: dict[str, list[float]] = defaultdict(list)
        self._cleanup_counter = 0

    def is_allowed(self, key: str) -> bool:
        """Return ``True`` if *key* has not exceeded the rate limit."""
        now = time.monotonic()
        if key in self._attempts:
            self._prune(key, now)
        self._maybe_cleanup(now)
        return len(self._attempts.get(key, [])) < self.max_attempts

    def record(self, key: str) -> None:
        """Record an attempt for *key*."""
        now = time.monotonic()
        self._prune(key, now)
        self._attempts[key].append(now)

    def remaining(self, key: str) -> int:
        """Return remaining attempts for *key*."""
        now = time.monotonic()
        self._prune(key, now)
        return max(0, self.max_attempts - len(self._attempts[key]))

    def reset(self, key: str) -> None:
        """Reset the counter for *key*."""
        self._attempts.pop(key, None)

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._attempts[key] = [t for t in self._attempts[key] if t > cutoff]
        if not self._attempts[key]:
            del self._attempts[key]

    def _maybe_cleanup(self, now: float) -> None:
        """Periodically clean all expired entries to prevent unbounded growth."""
        self._cleanup_counter += 1
        if self._cleanup_counter < 100 and len(self._attempts) < self.max_keys:
            return
        self._cleanup_counter = 0
        cutoff = now - self.window_seconds
        expired_keys = [k for k, ts in self._attempts.items() if not ts or ts[-1] <= cutoff]
        for k in expired_keys:
            del self._attempts[k]


# Singleton for login rate-limiting: 5 attempts per 60 seconds
login_rate_limiter = RateLimiter(max_attempts=5, window_seconds=60.0)
