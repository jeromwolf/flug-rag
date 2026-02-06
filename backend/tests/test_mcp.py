"""Tests for MCP server, registry, and tools."""

import json

import pytest

from agent.mcp.registry import ToolRegistry
from agent.mcp.server import MCPRequest, MCPServer
from agent.mcp.tools.base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult
from agent.mcp.tools.calculator_tool import CalculatorTool, safe_eval


# ---------------------------------------------------------------------------
# Calculator Tests
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    def setup_method(self):
        self.calc = CalculatorTool()

    async def test_basic_addition(self):
        result = await self.calc.execute(expression="2 + 3")
        assert result.success
        assert result.data["result"] == 5

    async def test_complex_expression(self):
        result = await self.calc.execute(expression="(100 + 200) * 1.1")
        assert result.success
        assert abs(result.data["result"] - 330.0) < 0.01

    async def test_division_by_zero(self):
        result = await self.calc.execute(expression="1 / 0")
        assert not result.success
        assert "error" in result.error.lower() or "division" in result.error.lower()

    async def test_empty_expression(self):
        result = await self.calc.execute(expression="")
        assert not result.success

    async def test_invalid_expression(self):
        result = await self.calc.execute(expression="import os")
        assert not result.success

    def test_safe_eval_basic(self):
        assert safe_eval("2 + 3") == 5
        assert safe_eval("10 - 4") == 6
        assert safe_eval("3 * 4") == 12
        assert safe_eval("10 / 4") == 2.5

    def test_safe_eval_power(self):
        assert safe_eval("2 ** 10") == 1024

    def test_safe_eval_negative(self):
        assert safe_eval("-5 + 3") == -2


# ---------------------------------------------------------------------------
# Tool Definition Tests
# ---------------------------------------------------------------------------


class TestToolDefinition:
    def test_to_schema(self):
        defn = ToolDefinition(
            name="test",
            description="A test tool",
            parameters=[
                ToolParameter(name="query", type=ToolParamType.STRING, description="Search query"),
                ToolParameter(name="limit", type=ToolParamType.INTEGER, description="Max results", required=False, default=10),
            ],
        )
        schema = defn.to_schema()
        assert schema["name"] == "test"
        assert "query" in schema["inputSchema"]["properties"]
        assert "limit" in schema["inputSchema"]["properties"]
        assert "query" in schema["inputSchema"]["required"]
        assert "limit" not in schema["inputSchema"]["required"]

    def test_to_schema_with_enum(self):
        defn = ToolDefinition(
            name="action_tool",
            description="Tool with enum",
            parameters=[
                ToolParameter(name="action", type=ToolParamType.STRING, description="Action", enum=["a", "b", "c"]),
            ],
        )
        schema = defn.to_schema()
        assert schema["inputSchema"]["properties"]["action"]["enum"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_and_list(self):
        self.registry.register(CalculatorTool())
        tools = self.registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_get_tool(self):
        calc = CalculatorTool()
        self.registry.register(calc)
        assert self.registry.get_tool("calculator") is calc
        assert self.registry.get_tool("nonexistent") is None

    def test_unregister(self):
        self.registry.register(CalculatorTool())
        self.registry.unregister("calculator")
        assert self.registry.tool_count == 0

    async def test_execute(self):
        self.registry.register(CalculatorTool())
        result = await self.registry.execute("calculator", expression="1+1")
        assert result.success
        assert result.data["result"] == 2

    async def test_execute_unknown_tool(self):
        result = await self.registry.execute("unknown_tool")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_list_schemas(self):
        self.registry.register(CalculatorTool())
        schemas = self.registry.list_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "calculator"
        assert "inputSchema" in schemas[0]

    def test_get_tools_by_category(self):
        self.registry.register(CalculatorTool())
        utility_tools = self.registry.get_tools_by_category("utility")
        assert len(utility_tools) == 1
        other_tools = self.registry.get_tools_by_category("nonexistent")
        assert len(other_tools) == 0


# ---------------------------------------------------------------------------
# MCP Server Tests
# ---------------------------------------------------------------------------


class TestMCPServer:
    def setup_method(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        self.server = MCPServer(registry=registry)

    async def test_ping(self):
        req = MCPRequest(method="ping", id=1)
        resp = await self.server.handle_request(req)
        assert resp.result == {"status": "ok"}
        assert resp.id == 1

    async def test_tools_list(self):
        req = MCPRequest(method="tools/list", id=2)
        resp = await self.server.handle_request(req)
        assert "tools" in resp.result
        assert len(resp.result["tools"]) == 1

    async def test_tools_call(self):
        req = MCPRequest(
            method="tools/call",
            params={"name": "calculator", "arguments": {"expression": "5 * 5"}},
            id=3,
        )
        resp = await self.server.handle_request(req)
        assert resp.error is None
        content_text = resp.result["content"][0]["text"]
        data = json.loads(content_text)
        assert data["result"] == 25

    async def test_tools_call_error(self):
        req = MCPRequest(
            method="tools/call",
            params={"name": "calculator", "arguments": {"expression": "1/0"}},
            id=4,
        )
        resp = await self.server.handle_request(req)
        assert resp.result.get("isError") is True

    async def test_unknown_method(self):
        req = MCPRequest(method="unknown/method", id=5)
        resp = await self.server.handle_request(req)
        assert resp.error is not None
        assert resp.error["code"] == -32601

    async def test_handle_json(self):
        json_str = json.dumps({
            "method": "ping",
            "id": 10,
        })
        resp_str = await self.server.handle_json(json_str)
        resp = json.loads(resp_str)
        assert resp["result"]["status"] == "ok"
        assert resp["id"] == 10

    async def test_handle_invalid_json(self):
        resp_str = await self.server.handle_json("not json {{{")
        resp = json.loads(resp_str)
        assert "error" in resp
        assert resp["error"]["code"] == -32700
