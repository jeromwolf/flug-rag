"""Base tool interface for MCP tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolParamType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    name: str
    type: ToolParamType
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """MCP-compatible tool definition."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: str = "general"

    def to_schema(self) -> dict:
        """Convert to JSON Schema format for MCP protocol."""
        properties = {}
        required = []
        for param in self.parameters:
            prop = {"type": param.type.value, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for MCP tools."""

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Return tool definition for MCP protocol."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    @property
    def name(self) -> str:
        return self.get_definition().name
