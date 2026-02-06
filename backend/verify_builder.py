#!/usr/bin/env python3
"""Quick verification script for Agent Builder module."""

import sys


def verify_imports():
    """Verify all imports work correctly."""
    try:
        # Test models import
        from agent.builder.models import (
            Edge, ExecutionStatus, NodeConfig, NodeType,
            Workflow, WorkflowNode, WorkflowStatus,
        )
        print("✓ Models imported successfully")

        # Test engine import
        from agent.builder.engine import WorkflowEngine
        print("✓ Engine imported successfully")

        # Test presets import
        from agent.builder.presets import (
            PRESET_WORKFLOWS, get_preset, list_presets,
        )
        print("✓ Presets imported successfully")

        # Test package import
        from agent.builder import (
            WorkflowEngine, Workflow, WorkflowNode, NodeConfig, NodeType,
            Edge, WorkflowStatus, ExecutionStatus,
            PRESET_WORKFLOWS, get_preset, list_presets,
        )
        print("✓ Package-level imports successful")

        # Test preset functionality
        presets = list_presets()
        print(f"✓ Found {len(presets)} preset workflows")
        for p in presets:
            print(f"  - {p['id']}: {p['name']}")

        # Test creating a simple workflow
        workflow = get_preset("simple_rag")
        if workflow:
            print(f"✓ Loaded preset workflow: {workflow.name}")
            print(f"  - Nodes: {len(workflow.nodes)}")
            print(f"  - Edges: {len(workflow.edges)}")
            start = workflow.get_start_node()
            if start:
                print(f"  - Has start node: {start.config.label}")

        # Test workflow creation
        test_workflow = Workflow(
            name="Test Workflow",
            nodes=[
                WorkflowNode(id="start", config=NodeConfig(
                    node_type=NodeType.START, label="Start"
                )),
            ],
        )
        print(f"✓ Created test workflow: {test_workflow.name}")

        # Test to_dict
        data = test_workflow.to_dict()
        print(f"✓ Workflow to_dict: {len(data)} keys")

        print("\n✅ All verification checks passed!")
        return True

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)
