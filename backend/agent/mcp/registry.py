"""Tool registry: manages available tools and their execution."""

from agent.mcp.tools.base import BaseTool, ToolDefinition, ToolResult
from config.settings import settings


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
        """Execute a tool by name with validated parameters."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}. Available: {list(self._tools.keys())}",
            )

        # Validate arguments against tool's declared parameters
        definition = tool.get_definition()
        declared_params = {p.name for p in definition.parameters}
        required_params = {p.name for p in definition.parameters if p.required}

        # Check for unknown parameters (only if tool has declared params)
        if declared_params:
            unknown = set(kwargs.keys()) - declared_params
            if unknown:
                return ToolResult(
                    success=False,
                    error=f"Unknown parameters: {unknown}. Allowed: {declared_params}",
                )

            # Check for missing required parameters
            missing = required_params - set(kwargs.keys())
            if missing:
                return ToolResult(
                    success=False,
                    error=f"Missing required parameters: {missing}",
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
    from agent.mcp.tools.asset_management_tool import AssetManagementTool
    from agent.mcp.tools.calculator_tool import CalculatorTool
    from agent.mcp.tools.chart_generator_tool import ChartGeneratorTool
    from agent.mcp.tools.code_executor_tool import CodeExecutorTool
    from agent.mcp.tools.data_analyzer_tool import DataAnalyzerTool
    from agent.mcp.tools.database_tool import KnowledgeBaseTool
    from agent.mcp.tools.db_query_tool import SystemDbQueryTool
    from agent.mcp.tools.ehsq_tool import EhsqTool
    from agent.mcp.tools.email_composer_tool import EmailComposerTool
    from agent.mcp.tools.erp_lookup_tool import ErpLookupTool
    from agent.mcp.tools.file_parser_tool import FileParserTool
    from agent.mcp.tools.groupware_tool import GroupwareTool
    from agent.mcp.tools.http_request_tool import HttpRequestTool
    from agent.mcp.tools.regulation_review_tool import RegulationReviewTool
    from agent.mcp.tools.report_draft_tool import ReportDraftTool
    from agent.mcp.tools.report_generator_tool import ReportGeneratorTool
    from agent.mcp.tools.safety_checklist_tool import SafetyChecklistTool
    from agent.mcp.tools.search_tool import DocumentSearchTool
    from agent.mcp.tools.summarizer_tool import SummarizerTool
    from agent.mcp.tools.training_material_tool import TrainingMaterialTool
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
    registry.register(RegulationReviewTool())
    registry.register(SafetyChecklistTool())
    registry.register(ReportDraftTool())
    registry.register(TrainingMaterialTool())
    # Enterprise system integrations
    registry.register(ErpLookupTool())
    registry.register(EhsqTool())
    registry.register(GroupwareTool())
    # External API integration (real HTTP call)
    registry.register(AssetManagementTool())
    # System DB real-time query
    registry.register(SystemDbQueryTool())
    # Utility & data tools
    registry.register(HttpRequestTool())
    registry.register(FileParserTool())
    registry.register(ChartGeneratorTool())
    registry.register(CodeExecutorTool())
    # Batch 2 tools
    from agent.mcp.tools.calendar_tool import CalendarTool
    from agent.mcp.tools.law_search_tool import LawSearchTool
    from agent.mcp.tools.nl_to_sql_tool import NlToSqlTool
    from agent.mcp.tools.teams_notify_tool import TeamsNotifyTool
    registry.register(CalendarTool())
    registry.register(LawSearchTool())
    registry.register(NlToSqlTool())
    registry.register(TeamsNotifyTool())
    # Upstage-dependent tools (only when API key is configured)
    if settings.upstage_api_key:
        from agent.mcp.tools.upstage_extract_tool import UpstageExtractTool
        from agent.mcp.tools.vision_analyzer_tool import VisionAnalyzerTool
        registry.register(UpstageExtractTool())
        registry.register(VisionAnalyzerTool())
    return registry


# Global registry singleton
_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global singleton registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = create_default_registry()
    return _global_registry
