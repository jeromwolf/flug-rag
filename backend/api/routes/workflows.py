"""Workflow endpoints for Agent Builder."""

from fastapi import APIRouter, HTTPException

from agent.builder import WorkflowEngine, get_preset, list_presets
from api.schemas import WorkflowRunRequest

router = APIRouter()


@router.get("/workflows/presets")
async def get_presets():
    return {"presets": list_presets()}


@router.post("/workflows/run")
async def run_workflow(request: WorkflowRunRequest):
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
