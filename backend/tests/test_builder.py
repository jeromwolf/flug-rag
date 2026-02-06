"""Tests for Agent Builder workflow engine."""

import pytest

from agent.builder import (
    Edge, ExecutionStatus, NodeConfig, NodeType, Workflow,
    WorkflowEngine, WorkflowNode, WorkflowStatus,
    get_preset, list_presets,
)


class TestWorkflowModels:
    """Test workflow data models."""

    def test_workflow_creation(self):
        """Test creating a workflow with nodes and edges."""
        start = WorkflowNode(id="start", config=NodeConfig(
            node_type=NodeType.START, label="Start",
        ))
        transform = WorkflowNode(id="transform", config=NodeConfig(
            node_type=NodeType.TRANSFORM, label="Transform",
            config={"template": "Result: {input}"},
        ))
        output = WorkflowNode(id="output", config=NodeConfig(
            node_type=NodeType.OUTPUT, label="Output",
        ))

        workflow = Workflow(
            name="Test Workflow",
            description="A test workflow",
            nodes=[start, transform, output],
            edges=[
                Edge(source="start", target="transform"),
                Edge(source="transform", target="output"),
            ],
            status=WorkflowStatus.ACTIVE,
        )

        assert workflow.name == "Test Workflow"
        assert len(workflow.nodes) == 3
        assert len(workflow.edges) == 2

    def test_workflow_to_dict(self):
        """Test converting workflow to dictionary."""
        start = WorkflowNode(id="start", config=NodeConfig(
            node_type=NodeType.START, label="Start",
        ))
        workflow = Workflow(
            name="Test Workflow",
            nodes=[start],
            edges=[],
        )

        data = workflow.to_dict()
        assert data["name"] == "Test Workflow"
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["type"] == "start"
        assert data["nodes"][0]["label"] == "Start"
        assert "id" in data
        assert "created_at" in data

    def test_get_node(self):
        """Test getting a node by ID."""
        node1 = WorkflowNode(id="node1", config=NodeConfig(
            node_type=NodeType.START, label="Node 1",
        ))
        node2 = WorkflowNode(id="node2", config=NodeConfig(
            node_type=NodeType.OUTPUT, label="Node 2",
        ))
        workflow = Workflow(nodes=[node1, node2], edges=[])

        found = workflow.get_node("node1")
        assert found is not None
        assert found.id == "node1"

        not_found = workflow.get_node("node3")
        assert not_found is None

    def test_get_start_node(self):
        """Test finding the start node."""
        start = WorkflowNode(id="start", config=NodeConfig(
            node_type=NodeType.START, label="Start",
        ))
        transform = WorkflowNode(id="transform", config=NodeConfig(
            node_type=NodeType.TRANSFORM, label="Transform",
        ))
        workflow = Workflow(nodes=[start, transform], edges=[])

        start_node = workflow.get_start_node()
        assert start_node is not None
        assert start_node.id == "start"
        assert start_node.config.node_type == NodeType.START

    def test_get_outgoing_edges(self):
        """Test getting outgoing edges from a node."""
        workflow = Workflow(
            nodes=[
                WorkflowNode(id="a", config=NodeConfig(node_type=NodeType.START, label="A")),
                WorkflowNode(id="b", config=NodeConfig(node_type=NodeType.OUTPUT, label="B")),
                WorkflowNode(id="c", config=NodeConfig(node_type=NodeType.OUTPUT, label="C")),
            ],
            edges=[
                Edge(source="a", target="b"),
                Edge(source="a", target="c"),
                Edge(source="b", target="c"),
            ],
        )

        edges_from_a = workflow.get_outgoing_edges("a")
        assert len(edges_from_a) == 2
        targets = {e.target for e in edges_from_a}
        assert targets == {"b", "c"}

        edges_from_b = workflow.get_outgoing_edges("b")
        assert len(edges_from_b) == 1
        assert edges_from_b[0].target == "c"


class TestWorkflowPresets:
    """Test preset workflow templates."""

    def test_list_presets_returns_three(self):
        """Test that list_presets returns 3 preset workflows."""
        presets = list_presets()
        assert len(presets) == 3
        assert all("id" in p and "name" in p and "description" in p for p in presets)

    def test_get_preset_returns_valid_workflow(self):
        """Test that get_preset returns a valid workflow."""
        workflow = get_preset("simple_rag")
        assert workflow is not None
        assert isinstance(workflow, Workflow)
        assert workflow.name == "간단 RAG"
        assert len(workflow.nodes) > 0
        assert workflow.status == WorkflowStatus.ACTIVE

    def test_preset_workflows_have_start_node(self):
        """Test that all preset workflows have a start node."""
        for preset_id in ["simple_rag", "routing", "quality_check"]:
            workflow = get_preset(preset_id)
            assert workflow is not None
            start_node = workflow.get_start_node()
            assert start_node is not None, f"Preset {preset_id} missing start node"
            assert start_node.config.node_type == NodeType.START

    def test_get_preset_nonexistent(self):
        """Test getting a non-existent preset."""
        workflow = get_preset("nonexistent")
        assert workflow is None


class TestWorkflowEngine:
    """Test workflow execution engine."""

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self):
        """Test executing a simple START → TRANSFORM → OUTPUT workflow."""
        start = WorkflowNode(id="start", config=NodeConfig(
            node_type=NodeType.START, label="Start",
        ))
        transform = WorkflowNode(id="transform", config=NodeConfig(
            node_type=NodeType.TRANSFORM, label="Transform",
            config={"template": "Result: {input}"},
        ))
        output = WorkflowNode(id="output", config=NodeConfig(
            node_type=NodeType.OUTPUT, label="Output",
        ))

        workflow = Workflow(
            name="Simple Workflow",
            nodes=[start, transform, output],
            edges=[
                Edge(source="start", target="transform"),
                Edge(source="transform", target="output"),
            ],
        )

        engine = WorkflowEngine()
        result = await engine.execute(workflow, {"query": "test input"})

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_output == "Result: test input"
        assert len(result.node_results) == 3
        assert all(nr.status == ExecutionStatus.COMPLETED for nr in result.node_results)

    @pytest.mark.asyncio
    async def test_execute_with_missing_start_node(self):
        """Test that execution fails when start node is missing."""
        # Workflow with no start node
        transform = WorkflowNode(id="transform", config=NodeConfig(
            node_type=NodeType.TRANSFORM, label="Transform",
            config={"template": "Result: {input}"},
        ))
        output = WorkflowNode(id="output", config=NodeConfig(
            node_type=NodeType.OUTPUT, label="Output",
        ))

        workflow = Workflow(
            name="Missing Start",
            nodes=[transform, output],
            edges=[Edge(source="transform", target="output")],
        )

        engine = WorkflowEngine()
        result = await engine.execute(workflow, {"query": "test"})

        assert result.status == ExecutionStatus.FAILED
        assert result.error == "No start node found in workflow"

    @pytest.mark.asyncio
    async def test_execute_multiple_outputs(self):
        """Test workflow with multiple transform nodes."""
        start = WorkflowNode(id="start", config=NodeConfig(
            node_type=NodeType.START, label="Start",
        ))
        transform1 = WorkflowNode(id="t1", config=NodeConfig(
            node_type=NodeType.TRANSFORM, label="Transform 1",
            config={"template": "Step1: {input}"},
        ))
        transform2 = WorkflowNode(id="t2", config=NodeConfig(
            node_type=NodeType.TRANSFORM, label="Transform 2",
            config={"template": "Step2: {input}"},
        ))
        output = WorkflowNode(id="output", config=NodeConfig(
            node_type=NodeType.OUTPUT, label="Output",
        ))

        workflow = Workflow(
            name="Multi-Transform",
            nodes=[start, transform1, transform2, output],
            edges=[
                Edge(source="start", target="t1"),
                Edge(source="t1", target="t2"),
                Edge(source="t2", target="output"),
            ],
        )

        engine = WorkflowEngine()
        result = await engine.execute(workflow, {"query": "hello"})

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_output == "Step2: Step1: hello"


class TestTransformNode:
    """Test transform node specifically."""

    @pytest.mark.asyncio
    async def test_simple_template_substitution(self):
        """Test simple {input} substitution in transform node."""
        workflow = Workflow(
            nodes=[
                WorkflowNode(id="start", config=NodeConfig(
                    node_type=NodeType.START, label="Start",
                )),
                WorkflowNode(id="transform", config=NodeConfig(
                    node_type=NodeType.TRANSFORM, label="Transform",
                    config={"template": "Transformed: {input}"},
                )),
                WorkflowNode(id="output", config=NodeConfig(
                    node_type=NodeType.OUTPUT, label="Output",
                )),
            ],
            edges=[
                Edge(source="start", target="transform"),
                Edge(source="transform", target="output"),
            ],
        )

        engine = WorkflowEngine()
        result = await engine.execute(workflow, {"query": "data"})

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_output == "Transformed: data"

    @pytest.mark.asyncio
    async def test_template_with_no_placeholder(self):
        """Test template without {input} placeholder."""
        workflow = Workflow(
            nodes=[
                WorkflowNode(id="start", config=NodeConfig(
                    node_type=NodeType.START, label="Start",
                )),
                WorkflowNode(id="transform", config=NodeConfig(
                    node_type=NodeType.TRANSFORM, label="Transform",
                    config={"template": "Static text"},
                )),
                WorkflowNode(id="output", config=NodeConfig(
                    node_type=NodeType.OUTPUT, label="Output",
                )),
            ],
            edges=[
                Edge(source="start", target="transform"),
                Edge(source="transform", target="output"),
            ],
        )

        engine = WorkflowEngine()
        result = await engine.execute(workflow, {"query": "ignored"})

        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_output == "Static text"
