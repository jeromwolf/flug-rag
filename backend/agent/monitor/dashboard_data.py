"""Dashboard data provider for frontend monitoring page."""

from typing import Any

from agent.monitor.tracker import ExecutionRecord, ExecutionTracker, ExecutionType


class DashboardDataProvider:
    """Provides formatted data for the frontend monitoring dashboard.

    Wraps ExecutionTracker to expose structured data suitable
    for charts and tables.
    """

    def __init__(self, tracker: ExecutionTracker):
        self._tracker = tracker

    def get_agent_metrics(self, time_range_seconds: float | None = None) -> dict:
        """Get agent-specific metrics.

        Args:
            time_range_seconds: Only include data from the last N seconds.

        Returns:
            Dict with agent execution statistics.
        """
        records = self._tracker.get_by_type(ExecutionType.AGENT)
        if time_range_seconds:
            import time
            cutoff = time.time() - time_range_seconds
            records = [r for r in records if r.start_time >= cutoff]

        return self._build_metrics(records, "agent")

    def get_chain_metrics(self, chain_id: str | None = None) -> dict:
        """Get chain execution metrics.

        Args:
            chain_id: Filter by a specific chain execution ID.

        Returns:
            Dict with chain execution statistics.
        """
        if chain_id:
            record = self._tracker.get(chain_id)
            if not record:
                return {"error": "Chain execution not found"}
            return {
                "chain_id": chain_id,
                "name": record.name,
                "state": record.state.value,
                "total_steps": record.total_steps,
                "completed_steps": record.completed_steps,
                "current_step": record.current_step,
                "duration_ms": record.duration_ms,
                "tokens_used": record.tokens_used,
                "error": record.error,
            }

        records = self._tracker.get_by_type(ExecutionType.CHAIN)
        return self._build_metrics(records, "chain")

    def get_tool_usage_stats(self) -> dict:
        """Get tool usage statistics."""
        records = self._tracker.get_by_type(ExecutionType.TOOL)

        tool_counts: dict[str, int] = {}
        tool_durations: dict[str, list[int]] = {}
        tool_errors: dict[str, int] = {}

        for r in records:
            name = r.name
            tool_counts[name] = tool_counts.get(name, 0) + 1
            if r.duration_ms is not None:
                tool_durations.setdefault(name, []).append(r.duration_ms)
            if r.error:
                tool_errors[name] = tool_errors.get(name, 0) + 1

        tools = []
        for name in tool_counts:
            durations = tool_durations.get(name, [])
            avg_duration = sum(durations) / len(durations) if durations else 0
            tools.append({
                "name": name,
                "call_count": tool_counts[name],
                "avg_duration_ms": round(avg_duration),
                "error_count": tool_errors.get(name, 0),
            })

        tools.sort(key=lambda x: x["call_count"], reverse=True)

        return {
            "total_tool_calls": sum(tool_counts.values()),
            "unique_tools": len(tool_counts),
            "tools": tools,
        }

    def get_active_executions(self) -> list[dict]:
        """Get currently active execution details."""
        active = self._tracker.get_active()
        return [self._format_record(r) for r in active]

    def get_overview(self) -> dict:
        """Get a high-level overview for the dashboard."""
        metrics = self._tracker.get_metrics()
        active = self._tracker.get_active()
        recent = self._tracker.get_recent(limit=10)

        return {
            "metrics": metrics,
            "active_count": len(active),
            "active_executions": [self._format_record(r) for r in active],
            "recent_executions": [self._format_record(r) for r in recent],
        }

    # ---- Internal ----

    def _build_metrics(self, records: list[ExecutionRecord], label: str) -> dict:
        """Build aggregated metrics from a list of records."""
        if not records:
            return {
                "label": label,
                "total": 0,
                "completed": 0,
                "failed": 0,
                "running": 0,
                "avg_duration_ms": 0,
                "total_tokens": 0,
                "success_rate": 0.0,
            }

        from agent.monitor.tracker import ExecutionState

        completed = [r for r in records if r.state == ExecutionState.COMPLETED]
        failed = [r for r in records if r.state == ExecutionState.FAILED]
        running = [r for r in records if r.state == ExecutionState.RUNNING]

        durations = [r.duration_ms for r in completed if r.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0
        total_tokens = sum(r.tokens_used for r in records)

        finished = len(completed) + len(failed)
        success_rate = len(completed) / finished if finished > 0 else 0.0

        return {
            "label": label,
            "total": len(records),
            "completed": len(completed),
            "failed": len(failed),
            "running": len(running),
            "avg_duration_ms": round(avg_duration),
            "total_tokens": total_tokens,
            "success_rate": round(success_rate, 3),
        }

    def _format_record(self, record: ExecutionRecord) -> dict:
        """Format an execution record for the frontend."""
        return record.to_dict()
