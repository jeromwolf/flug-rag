"""In-memory ring buffer handler for recent WARNING/ERROR logs.

Used by the admin dashboard to show real-time error logs
instead of hardcoded mock data.
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any


class MemoryLogHandler(logging.Handler):
    """Keeps the most recent WARNING+ log records in memory."""

    def __init__(self, maxlen: int = 200):
        super().__init__(level=logging.WARNING)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": record.module,
                "funcName": record.funcName,
                "lineno": record.lineno,
            }
            if record.exc_info and record.exc_info[1]:
                entry["exception"] = str(record.exc_info[1])
            self._buffer.append(entry)
        except Exception:
            self.handleError(record)

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return most recent entries (newest first)."""
        entries = list(self._buffer)
        entries.reverse()
        return entries[:limit]

    def clear(self) -> None:
        self._buffer.clear()


# Module-level singleton
_handler: MemoryLogHandler | None = None


def get_memory_log_handler() -> MemoryLogHandler:
    """Get or create the singleton MemoryLogHandler."""
    global _handler
    if _handler is None:
        _handler = MemoryLogHandler()
        _handler.setFormatter(logging.Formatter("%(message)s"))
    return _handler


def install_memory_log_handler() -> MemoryLogHandler:
    """Install the handler on the root logger and return it."""
    handler = get_memory_log_handler()
    root = logging.getLogger()
    # Avoid duplicate registration
    if handler not in root.handlers:
        root.addHandler(handler)
    return handler
