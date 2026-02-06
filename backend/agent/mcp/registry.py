"""Tool registry: manages available tools and their execution."""

from agent.mcp.tools.base import BaseTool, ToolDefinition, ToolResult


class ToolRegistry:
    """Registry for MCP tools. Manages tool registration and execution."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tool definitions."""
        return [tool.get_definition() for tool in self._tools.values()]

    def list_schemas(self) -> list[dict]:
        """List all tool schemas (MCP-compatible format)."""
        return [tool.get_definition().to_schema() for tool in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with given parameters."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}. Available: {list(self._tools.keys())}",
            )
        return await tool.execute(**kwargs)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_tools_by_category(self, category: str) -> list[ToolDefinition]:
        """Get tools filtered by category."""
        return [
            tool.get_definition()
            for tool in self._tools.values()
            if tool.get_definition().category == category
        ]


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools."""
    from agent.mcp.tools.calculator_tool import CalculatorTool
    from agent.mcp.tools.data_analyzer_tool import DataAnalyzerTool
    from agent.mcp.tools.database_tool import KnowledgeBaseTool
    from agent.mcp.tools.email_composer_tool import EmailComposerTool
    from agent.mcp.tools.report_generator_tool import ReportGeneratorTool
    from agent.mcp.tools.search_tool import DocumentSearchTool
    from agent.mcp.tools.summarizer_tool import SummarizerTool
    from agent.mcp.tools.translator_tool import TranslatorTool

    registry = ToolRegistry()
    registry.register(DocumentSearchTool())
    registry.register(KnowledgeBaseTool())
    registry.register(CalculatorTool())
    registry.register(SummarizerTool())
    registry.register(TranslatorTool())
    registry.register(ReportGeneratorTool())
    registry.register(EmailComposerTool())
    registry.register(DataAnalyzerTool())
    return registry
