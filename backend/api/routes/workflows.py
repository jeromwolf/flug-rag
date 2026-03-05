"""Workflow endpoints for Agent Builder."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent.builder import WorkflowEngine, get_preset, list_presets
from agent.builder.workflow_store import get_workflow_store
from api.schemas import WorkflowRunRequest
from auth.dependencies import get_current_user
from auth.models import User

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
# Preset endpoints (existing)
# ---------------------------------------------------------------------------

@router.get("/workflows/presets")
async def get_presets(current_user: User | None = Depends(get_current_user)):
    return {"presets": list_presets()}


@router.post("/workflows/run")
async def run_workflow(request: WorkflowRunRequest, current_user: User | None = Depends(get_current_user)):
    if request.preset:
        workflow = get_preset(request.preset)
        if not workflow:
            raise HTTPException(404, f"Preset not found: {request.preset}")
    else:
        raise HTTPException(400, "Either preset or workflow_id is required")

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
    current_user: User | None = Depends(get_current_user),
):
    """Save a new workflow."""
    store = await get_workflow_store()
    user_id = current_user.id if current_user else None
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
    current_user: User | None = Depends(get_current_user),
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
async def delete_workflow(workflow_id: str, current_user: User | None = Depends(get_current_user)):
    """Delete a saved workflow."""
    store = await get_workflow_store()
    deleted = await store.delete(workflow_id)
    if not deleted:
        raise HTTPException(404, f"Workflow not found: {workflow_id}")
