"""Workflow endpoints for Agent Builder."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.builder import WorkflowEngine, get_preset, list_presets
from agent.builder.models import Edge, NodeConfig, NodeType, Workflow, WorkflowNode, WorkflowStatus
from agent.builder.workflow_store import get_workflow_store
from api.schemas import WorkflowRunRequest
from auth.dependencies import get_current_user, require_role
from auth.models import Role, User

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class WorkflowSaveRequest(BaseModel):
    name: str
    description: str = ""
    nodes: list = []
    edges: list = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_workflow_from_saved(data: dict) -> Workflow:
    """Convert a saved workflow dict (from SQLite) into a Workflow model."""
    nodes = []
    for n in data.get("nodes", []):
        node_type_str = n.get("type", "start")
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            node_type = NodeType.START
        nodes.append(WorkflowNode(
            id=n["id"],
            config=NodeConfig(
                node_type=node_type,
                label=n.get("label", node_type_str),
                config=n.get("config", {}),
                position=n.get("position", {"x": 0, "y": 0}),
            ),
        ))

    edges = []
    for e in data.get("edges", []):
        edges.append(Edge(
            id=e.get("id", ""),
            source=e.get("source", ""),
            target=e.get("target", ""),
            label=e.get("label", ""),
            condition=e.get("condition"),
        ))

    return Workflow(
        id=data.get("id", ""),
        name=data.get("name", ""),
        description=data.get("description", ""),
        nodes=nodes,
        edges=edges,
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Preset endpoints (existing)
# ---------------------------------------------------------------------------

@router.get("/workflows/presets")
async def get_presets(current_user: User | None = Depends(get_current_user)):
    return {"presets": list_presets()}


def _convert_frontend_nodes(frontend_nodes: list) -> list:
    """Convert ReactFlow frontend node format to backend format."""
    result = []
    for n in frontend_nodes:
        data = n.get("data", {})
        result.append({
            "id": n.get("id", ""),
            "type": data.get("nodeType", "start"),
            "label": data.get("label", ""),
            "config": data.get("config", {}),
            "position": n.get("position", {"x": 0, "y": 0}),
        })
    return result


@router.post("/workflows/run")
async def run_workflow(request: WorkflowRunRequest, current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER]))):
    if request.preset:
        workflow = get_preset(request.preset)
        if not workflow:
            raise HTTPException(404, f"Preset not found: {request.preset}")
    elif request.workflow_id:
        store = await get_workflow_store()
        saved = await store.get(request.workflow_id)
        if not saved:
            raise HTTPException(404, f"Workflow not found: {request.workflow_id}")
        workflow = _build_workflow_from_saved(saved)
    elif request.nodes and request.edges is not None:
        workflow = _build_workflow_from_saved({
            "id": "inline",
            "name": "Inline Workflow",
            "description": "",
            "nodes": _convert_frontend_nodes(request.nodes),
            "edges": [
                {
                    "id": e.get("id", ""),
                    "source": e.get("source", ""),
                    "target": e.get("target", ""),
                    "label": e.get("label", ""),
                    "condition": e.get("condition"),
                }
                for e in request.edges
            ],
        })
    else:
        raise HTTPException(400, "Either preset, workflow_id, or nodes+edges is required")

    engine = WorkflowEngine()
    result = await engine.execute(workflow, request.input_data)

    return {
        "workflow_id": result.workflow_id,
        "status": result.status.value,
        "final_output": result.final_output,
        "total_duration_ms": result.total_duration_ms,
        "node_results": [
            {
                "node_id": nr.node_id,
                "status": nr.status.value,
                "duration_ms": nr.duration_ms,
            }
            for nr in result.node_results
        ],
        "error": result.error,
    }


# ---------------------------------------------------------------------------
# CRUD endpoints for user-saved workflows
# ---------------------------------------------------------------------------

@router.get("/workflows")
async def list_workflows(current_user: User | None = Depends(get_current_user)):
    """List all saved workflows."""
    store = await get_workflow_store()
    user_id = current_user.id if current_user else None
    workflows = await store.list_all(user_id=user_id)
    return {"workflows": workflows}


@router.post("/workflows", status_code=201)
async def create_workflow(
    request: WorkflowSaveRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Save a new workflow."""
    store = await get_workflow_store()
    user_id = current_user.id
    workflow = await store.create(
        name=request.name,
        description=request.description,
        nodes=request.nodes,
        edges=request.edges,
        user_id=user_id,
    )
    return workflow


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str, current_user: User | None = Depends(get_current_user)):
    """Get a single saved workflow by ID."""
    store = await get_workflow_store()
    workflow = await store.get(workflow_id)
    if workflow is None:
        raise HTTPException(404, f"Workflow not found: {workflow_id}")
    return workflow


@router.put("/workflows/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    request: WorkflowSaveRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Update an existing saved workflow."""
    store = await get_workflow_store()
    existing = await store.get(workflow_id)
    if existing is None:
        raise HTTPException(404, f"Workflow not found: {workflow_id}")
    workflow = await store.update(
        workflow_id=workflow_id,
        name=request.name,
        description=request.description,
        nodes=request.nodes,
        edges=request.edges,
    )
    return workflow


@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str, current_user: User = Depends(require_role([Role.ADMIN]))):
    """Delete a saved workflow."""
    store = await get_workflow_store()
    deleted = await store.delete(workflow_id)
    if not deleted:
        raise HTTPException(404, f"Workflow not found: {workflow_id}")
