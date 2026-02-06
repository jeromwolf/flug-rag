"""MCP tool endpoints."""

from fastapi import APIRouter

from agent.mcp import MCPServer, ToolRegistry, create_default_registry
from api.schemas import MCPCallRequest

router = APIRouter()

_registry = None
_server = None


def _get_registry():
    global _registry
    if _registry is None:
        _registry = create_default_registry()
    return _registry


def _get_server():
    global _server
    if _server is None:
        _server = MCPServer(registry=_get_registry())
    return _server


@router.get("/mcp/tools")
async def list_tools():
    registry = _get_registry()
    return {"tools": registry.list_schemas()}


@router.post("/mcp/call")
async def call_tool(request: MCPCallRequest):
    registry = _get_registry()
    result = await registry.execute(request.tool_name, **request.arguments)
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "metadata": result.metadata,
    }
