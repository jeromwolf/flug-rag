"""Workflow data models for Agent Builder."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Types of nodes in a workflow."""
    START = "start"
    LLM = "llm"              # LLM generation node
    RETRIEVAL = "retrieval"   # RAG retrieval node
    TOOL = "tool"             # MCP tool execution
    CONDITION = "condition"   # Branching logic
    OUTPUT = "output"         # Final output
    TRANSFORM = "transform"   # Data transformation


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NodeConfig:
    """Configuration for a workflow node."""
    node_type: NodeType
    label: str
    config: dict = field(default_factory=dict)
    # config examples:
    # LLM: {"provider": "vllm", "model": "...", "system_prompt": "...", "temperature": 0.7}
    # RETRIEVAL: {"top_k": 5, "filters": {}}
    # TOOL: {"tool_name": "calculator", "arguments_template": {}}
    # CONDITION: {"condition_type": "confidence", "threshold": 0.5}
    # TRANSFORM: {"template": "결과: {input}"}
    position: dict = field(default_factory=lambda: {"x": 0, "y": 0})


@dataclass
class Edge:
    """Connection between two nodes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""        # source node_id
    target: str = ""        # target node_id
    label: str = ""         # edge label (e.g., "true", "false" for conditions)
    condition: str | None = None  # Optional condition expression


@dataclass
class WorkflowNode:
    """A node instance in a workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: NodeConfig = field(default_factory=lambda: NodeConfig(node_type=NodeType.START, label="Start"))


@dataclass
class Workflow:
    """A complete workflow definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    nodes: list[WorkflowNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    def get_node(self, node_id: str) -> WorkflowNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_start_node(self) -> WorkflowNode | None:
        for node in self.nodes:
            if node.config.node_type == NodeType.START:
                return node
        return None

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_id]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.config.node_type.value,
                    "label": n.config.label,
                    "config": n.config.config,
                    "position": n.config.position,
                }
                for n in self.nodes
            ],
            "edges": [
                {"id": e.id, "source": e.source, "target": e.target, "label": e.label}
                for e in self.edges
            ],
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class NodeExecutionResult:
    """Result from executing a single node."""
    node_id: str
    status: ExecutionStatus
    output: Any = None
    error: str | None = None
    duration_ms: int = 0


@dataclass
class WorkflowExecutionResult:
    """Result from executing a complete workflow."""
    workflow_id: str
    status: ExecutionStatus
    node_results: list[NodeExecutionResult] = field(default_factory=list)
    final_output: Any = None
    total_duration_ms: int = 0
    error: str | None = None
