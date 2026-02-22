"""Agent API routes: tools, chains, and monitoring."""

import asyncio
import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from auth.dependencies import get_current_user, require_role
from auth.models import Role, User

from agent.chaining.templates import CHAIN_TEMPLATES, list_chain_templates
from agent.mcp.registry import ToolRegistry, create_default_registry
from agent.monitor.dashboard_data import DashboardDataProvider
from agent.monitor.tracker import ExecutionTracker, ExecutionType

router = APIRouter()

# ---- Singletons ----

_registry: ToolRegistry | None = None
_tracker: ExecutionTracker | None = None
_dashboard: DashboardDataProvider | None = None


def _get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = create_default_registry()
    return _registry


def _get_tracker() -> ExecutionTracker:
    global _tracker
    if _tracker is None:
        _tracker = ExecutionTracker()
    return _tracker


def _get_dashboard() -> DashboardDataProvider:
    global _dashboard
    if _dashboard is None:
        _dashboard = DashboardDataProvider(tracker=_get_tracker())
    return _dashboard


# ---- Tool Routes ----


@router.get("/agents/tools")
async def list_tools(current_user: User = Depends(get_current_user)):
    """List all available MCP tools."""
    registry = _get_registry()
    return {"tools": registry.list_schemas(), "count": registry.tool_count}


@router.post("/agents/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, arguments: dict = {}, current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER]))):
    """Execute a specific MCP tool."""
    registry = _get_registry()
    tracker = _get_tracker()

    exec_id = tracker.start(
        execution_type=ExecutionType.TOOL,
        name=tool_name,
    )

    try:
        result = await registry.execute(tool_name, **arguments)
        if result.success:
            tracker.complete(exec_id)
        else:
            tracker.fail(exec_id, result.error or "Unknown error")

        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "metadata": result.metadata,
            "execution_id": exec_id,
        }
    except Exception as e:
        tracker.fail(exec_id, str(e))
        return {"success": False, "error": str(e), "execution_id": exec_id}


# ---- Chain Routes ----


@router.get("/agents/chains")
async def list_chains(current_user: User = Depends(get_current_user)):
    """List available chain templates."""
    return {"chains": list_chain_templates()}


@router.post("/agents/chains/{template}/execute")
async def execute_chain(template: str, input_data: dict = {}, current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER]))):
    """Execute a chain template.

    Note: In production, chain steps would have real agent execute functions.
    This endpoint demonstrates the structure and tracking.
    """
    template_info = CHAIN_TEMPLATES.get(template)
    if not template_info:
        return {
            "success": False,
            "error": f"Unknown chain template: {template}. Available: {list(CHAIN_TEMPLATES.keys())}",
        }

    tracker = _get_tracker()
    steps = template_info["steps"]
    exec_id = tracker.start(
        execution_type=ExecutionType.CHAIN,
        name=template,
        total_steps=len(steps),
    )

    # Create a chain with pass-through functions for demonstration
    factory = template_info["factory"]

    # Build step functions that wrap tool calls where applicable
    registry = _get_registry()

    async def _tool_passthrough(data):
        """Pass data through, simulating a tool call."""
        return data

    # Build the chain with simple pass-through functions
    kwargs = {f"{step}_fn": _tool_passthrough for step in steps}
    chain = factory(**kwargs)

    try:
        result = await chain.execute(input_data)

        if result.success:
            tracker.complete(exec_id)
        else:
            tracker.fail(exec_id, result.error or "Chain execution failed")

        return {
            "success": result.success,
            "final_output": result.final_output,
            "step_results": result.step_results,
            "total_duration_ms": result.total_duration_ms,
            "error": result.error,
            "execution_id": exec_id,
        }
    except Exception as e:
        tracker.fail(exec_id, str(e))
        return {"success": False, "error": str(e), "execution_id": exec_id}


# ---- Monitor Routes ----


@router.get("/agents/monitor/metrics")
async def get_metrics(time_range: int | None = None, current_user: User = Depends(get_current_user)):
    """Get execution metrics.

    Args:
        time_range: Time range in seconds. None for all time.
    """
    tracker = _get_tracker()
    return tracker.get_metrics(time_range_seconds=time_range)


@router.get("/agents/monitor/active")
async def get_active_executions(current_user: User = Depends(get_current_user)):
    """Get currently active executions."""
    dashboard = _get_dashboard()
    return {"executions": dashboard.get_active_executions()}


@router.get("/agents/monitor/overview")
async def get_overview(current_user: User = Depends(get_current_user)):
    """Get dashboard overview with metrics and recent executions."""
    dashboard = _get_dashboard()
    return dashboard.get_overview()


@router.get("/agents/monitor/tools")
async def get_tool_stats(current_user: User = Depends(get_current_user)):
    """Get tool usage statistics."""
    dashboard = _get_dashboard()
    return dashboard.get_tool_usage_stats()


# ---- WebSocket for real-time updates ----


class ConnectionManager:
    """Manages WebSocket connections for real-time monitoring."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


_ws_manager = ConnectionManager()


@router.websocket("/agents/monitor/ws")
async def monitor_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time execution updates."""
    # Validate JWT from query parameter
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return
    try:
        from auth.jwt_handler import verify_token
        verify_token(token)
    except Exception:
        await websocket.close(code=1008)
        return

    await _ws_manager.connect(websocket)

    # Register tracker callback to broadcast updates
    tracker = _get_tracker()

    def on_update(record):
        """Sync callback that schedules async broadcast."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_ws_manager.broadcast(record.to_dict()))
        except RuntimeError:
            pass

    tracker.on_update(on_update)

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Client can send "ping" to keep alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        _ws_manager.disconnect(websocket)
        tracker.remove_callback(on_update)
