"""MCP (Model Context Protocol) server implementation."""

import json
from dataclasses import dataclass, field
from typing import Any

from agent.mcp.registry import ToolRegistry, create_default_registry
from agent.mcp.tools.base import ToolResult


@dataclass
class MCPRequest:
    """Incoming MCP request."""
    method: str
    params: dict = field(default_factory=dict)
    id: str | int | None = None


@dataclass
class MCPResponse:
    """Outgoing MCP response."""
    result: Any = None
    error: dict | None = None
    id: str | int | None = None

    def to_dict(self) -> dict:
        resp = {}
        if self.id is not None:
            resp["id"] = self.id
        if self.error:
            resp["error"] = self.error
        else:
            resp["result"] = self.result
        return resp


class MCPServer:
    """MCP-compatible server for tool execution.

    Implements a subset of the MCP protocol:
    - tools/list: List available tools
    - tools/call: Execute a tool
    - ping: Health check
    """

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or create_default_registry()
        self._handlers = {
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
        }

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request and return response."""
        handler = self._handlers.get(request.method)
        if not handler:
            return MCPResponse(
                error={"code": -32601, "message": f"Method not found: {request.method}"},
                id=request.id,
            )

        try:
            result = await handler(request.params)
            return MCPResponse(result=result, id=request.id)
        except Exception as e:
            return MCPResponse(
                error={"code": -32000, "message": str(e)},
                id=request.id,
            )

    async def handle_json(self, json_str: str) -> str:
        """Handle a raw JSON-RPC string. Returns JSON response string."""
        try:
            data = json.loads(json_str)
            request = MCPRequest(
                method=data.get("method", ""),
                params=data.get("params", {}),
                id=data.get("id"),
            )
            response = await self.handle_request(request)
            return json.dumps(response.to_dict(), ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps({
                "error": {"code": -32700, "message": "Parse error"},
            })

    async def _handle_tools_list(self, params: dict) -> dict:
        """Handle tools/list request."""
        schemas = self.registry.list_schemas()
        return {"tools": schemas}

    async def _handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        result: ToolResult = await self.registry.execute(tool_name, **arguments)

        if result.success:
            return {
                "content": [
                    {"type": "text", "text": json.dumps(result.data, ensure_ascii=False, default=str)}
                ],
            }
        else:
            return {
                "content": [
                    {"type": "text", "text": f"Error: {result.error}"}
                ],
                "isError": True,
            }

    async def _handle_ping(self, params: dict) -> dict:
        return {"status": "ok"}
