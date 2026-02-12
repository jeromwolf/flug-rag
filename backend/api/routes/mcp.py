"""MCP tool endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from agent.mcp import MCPServer, get_registry
from api.schemas import MCPCallRequest
from auth.dependencies import get_current_user, require_role
from auth.models import Role, User

router = APIRouter()

_server = None


def _get_server():
    global _server
    if _server is None:
        _server = MCPServer(registry=get_registry())
    return _server


@router.get("/mcp/tools")
async def list_tools(current_user: User | None = Depends(get_current_user)):
    registry = get_registry()
    return {"tools": registry.list_schemas()}


@router.post("/mcp/call")
async def call_tool(request: MCPCallRequest, current_user: User | None = Depends(get_current_user)):
    registry = get_registry()
    result = await registry.execute(request.tool_name, **request.arguments)
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "metadata": result.metadata,
    }


# ============================================================================
# Custom Tool Management Endpoints
# ============================================================================


@router.post("/mcp/tools/custom")
async def create_custom_tool(
    request: dict,
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """커스텀 도구 생성."""
    from agent.mcp.custom_tools import get_custom_tool_store, CustomToolExecutor

    store = await get_custom_tool_store()
    registry = get_registry()

    # Validate name uniqueness against built-in tools
    if request.get("name") in [t.name for t in registry.list_tools()]:
        raise HTTPException(status_code=409, detail=f"도구 이름 '{request['name']}'이 이미 존재합니다")

    tool = await store.create(created_by=current_user.username, **request)

    # Register in runtime
    executor = CustomToolExecutor(tool)
    registry.register(executor)

    return {"status": "created", "tool": tool.__dict__}


@router.get("/mcp/tools/custom")
async def list_custom_tools(
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """커스텀 도구 목록."""
    from agent.mcp.custom_tools import get_custom_tool_store
    store = await get_custom_tool_store()
    tools = await store.list_all()
    return {"tools": [t.__dict__ for t in tools]}


@router.put("/mcp/tools/custom/{tool_id}")
async def update_custom_tool(
    tool_id: str,
    request: dict,
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """커스텀 도구 수정."""
    from agent.mcp.custom_tools import get_custom_tool_store, CustomToolExecutor
    store = await get_custom_tool_store()
    registry = get_registry()

    try:
        tool = await store.update(tool_id, **request)

        # Re-register executor with updated config
        executor = CustomToolExecutor(tool)
        registry.unregister(tool.name)
        registry.register(executor)

        return {"status": "updated", "tool": tool.__dict__}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/mcp/tools/custom/{tool_id}")
async def delete_custom_tool(
    tool_id: str,
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """커스텀 도구 삭제."""
    from agent.mcp.custom_tools import get_custom_tool_store
    store = await get_custom_tool_store()
    registry = get_registry()

    try:
        # Get tool name before deletion
        config = await store.get(tool_id)
        if not config:
            raise ValueError(f"Tool not found: {tool_id}")

        # Unregister from runtime
        registry.unregister(config.name)

        # Delete from storage
        await store.delete(tool_id)

        return {"status": "deleted", "tool_id": tool_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/mcp/tools/custom/{tool_id}/test")
async def test_custom_tool(
    tool_id: str,
    test_input: dict,
    current_user: User = Depends(require_role([Role.ADMIN])),
):
    """커스텀 도구 테스트 실행."""
    from agent.mcp.custom_tools import get_custom_tool_store, CustomToolExecutor
    store = await get_custom_tool_store()
    config = await store.get(tool_id)
    if not config:
        raise HTTPException(status_code=404, detail="Tool not found")

    executor = CustomToolExecutor(config)
    result = await executor.execute(**test_input)
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "metadata": result.metadata,
    }
