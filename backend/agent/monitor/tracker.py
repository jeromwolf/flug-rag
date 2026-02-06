"""Execution tracker for agent and chain metrics."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ExecutionType(str, Enum):
    AGENT = "agent"
    CHAIN = "chain"
    TOOL = "tool"


class ExecutionState(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionRecord:
    """Record of a single execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    execution_type: ExecutionType = ExecutionType.AGENT
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    state: ExecutionState = ExecutionState.RUNNING
    tokens_used: int = 0
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    # Chain-specific
    total_steps: int = 0
    completed_steps: int = 0
    current_step: str = ""

    @property
    def duration_ms(self) -> int | None:
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return None

    @property
    def is_active(self) -> bool:
        return self.state == ExecutionState.RUNNING

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "execution_type": self.execution_type.value,
            "name": self.name,
            "state": self.state.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "error": self.error,
            "metadata": self.metadata,
        }


class ExecutionTracker:
    """Tracks agent, chain, and tool executions.

    Provides real-time progress updates via callbacks and
    aggregated metrics.
    """

    def __init__(self):
        self._records: dict[str, ExecutionRecord] = {}
        self._on_update: list[Callable] = []

    # ---- Lifecycle ----

    def start(
        self,
        execution_type: ExecutionType,
        name: str,
        total_steps: int = 0,
        **metadata,
    ) -> str:
        """Start tracking an execution. Returns execution ID."""
        record = ExecutionRecord(
            execution_type=execution_type,
            name=name,
            total_steps=total_steps,
            metadata=metadata,
        )
        self._records[record.id] = record
        self._notify(record)
        return record.id

    def update_step(self, execution_id: str, step_name: str, completed_steps: int | None = None) -> None:
        """Update the current step of a chain execution."""
        record = self._records.get(execution_id)
        if not record:
            return
        record.current_step = step_name
        if completed_steps is not None:
            record.completed_steps = completed_steps
        self._notify(record)

    def add_tokens(self, execution_id: str, tokens: int) -> None:
        """Add token usage to an execution."""
        record = self._records.get(execution_id)
        if record:
            record.tokens_used += tokens

    def complete(self, execution_id: str, **metadata) -> None:
        """Mark an execution as completed."""
        record = self._records.get(execution_id)
        if not record:
            return
        record.state = ExecutionState.COMPLETED
        record.end_time = time.time()
        record.completed_steps = record.total_steps
        record.metadata.update(metadata)
        self._notify(record)

    def fail(self, execution_id: str, error: str) -> None:
        """Mark an execution as failed."""
        record = self._records.get(execution_id)
        if not record:
            return
        record.state = ExecutionState.FAILED
        record.end_time = time.time()
        record.error = error
        self._notify(record)

    # ---- Queries ----

    def get(self, execution_id: str) -> ExecutionRecord | None:
        """Get an execution record by ID."""
        return self._records.get(execution_id)

    def get_active(self) -> list[ExecutionRecord]:
        """Get all currently active executions."""
        return [r for r in self._records.values() if r.is_active]

    def get_recent(self, limit: int = 50) -> list[ExecutionRecord]:
        """Get most recent executions."""
        records = sorted(self._records.values(), key=lambda r: r.start_time, reverse=True)
        return records[:limit]

    def get_by_type(self, execution_type: ExecutionType) -> list[ExecutionRecord]:
        """Get executions filtered by type."""
        return [r for r in self._records.values() if r.execution_type == execution_type]

    # ---- Metrics ----

    def get_metrics(self, time_range_seconds: float | None = None) -> dict:
        """Compute aggregated metrics.

        Args:
            time_range_seconds: Only include records from the last N seconds.
        """
        records = list(self._records.values())
        if time_range_seconds:
            cutoff = time.time() - time_range_seconds
            records = [r for r in records if r.start_time >= cutoff]

        if not records:
            return {
                "total_executions": 0,
                "completed": 0,
                "failed": 0,
                "running": 0,
                "avg_duration_ms": 0,
                "total_tokens": 0,
                "success_rate": 0.0,
            }

        completed = [r for r in records if r.state == ExecutionState.COMPLETED]
        failed = [r for r in records if r.state == ExecutionState.FAILED]
        running = [r for r in records if r.state == ExecutionState.RUNNING]

        durations = [r.duration_ms for r in completed if r.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        total_tokens = sum(r.tokens_used for r in records)

        finished = len(completed) + len(failed)
        success_rate = len(completed) / finished if finished > 0 else 0.0

        return {
            "total_executions": len(records),
            "completed": len(completed),
            "failed": len(failed),
            "running": len(running),
            "avg_duration_ms": round(avg_duration),
            "total_tokens": total_tokens,
            "success_rate": round(success_rate, 3),
            "by_type": {
                etype.value: len([r for r in records if r.execution_type == etype])
                for etype in ExecutionType
            },
        }

    # ---- Callbacks ----

    def on_update(self, callback: Callable) -> None:
        """Register a callback for execution updates.

        The callback receives (ExecutionRecord) on any state change.
        """
        self._on_update.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a previously registered callback."""
        self._on_update = [cb for cb in self._on_update if cb is not callback]

    def _notify(self, record: ExecutionRecord) -> None:
        """Notify all registered callbacks."""
        for callback in self._on_update:
            try:
                callback(record)
            except Exception:
                pass

    # ---- Cleanup ----

    def clear(self, keep_active: bool = True) -> int:
        """Clear historical records.

        Args:
            keep_active: If True, keep currently running records.

        Returns:
            Number of records removed.
        """
        if keep_active:
            active_ids = {r.id for r in self.get_active()}
            to_remove = [rid for rid in self._records if rid not in active_ids]
        else:
            to_remove = list(self._records.keys())

        for rid in to_remove:
            del self._records[rid]
        return len(to_remove)
