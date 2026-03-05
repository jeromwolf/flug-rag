"""Session and conversation history endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from agent import get_memory
from api.schemas import (
    MessageResponse,
    SessionCreate,
    SessionListResponse,
    SessionResponse,
)
from auth.dependencies import get_current_user
from auth.models import User

router = APIRouter()


def _get_memory():
    return get_memory()


async def _verify_session_owner(session_id: str, user_id: str) -> dict:
    """Verify session exists and belongs to the user."""
    session = await _get_memory().get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session_owner = session.get("user_id", "")
    if session_owner and session_owner != user_id:
        raise HTTPException(403, "Access denied")
    return session


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate, current_user: User | None = Depends(get_current_user)):
    user_id = current_user.id if current_user else ""
    session_id = await _get_memory().create_session(title=request.title, user_id=user_id)
    session = await _get_memory().get_session(session_id)
    return SessionResponse(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), current_user: User | None = Depends(get_current_user)):
    user_id = current_user.id if current_user else ""
    sessions = await _get_memory().get_sessions(limit=limit, offset=offset, user_id=user_id)
    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s["id"],
                title=s["title"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                message_count=s.get("message_count", 0),
            )
            for s in sessions
        ]
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, current_user: User | None = Depends(get_current_user)):
    user_id = current_user.id if current_user else ""
    return await _verify_session_owner(session_id, user_id)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(session_id: str, limit: int = Query(50, ge=1, le=500), current_user: User | None = Depends(get_current_user)):
    user_id = current_user.id if current_user else ""
    await _verify_session_owner(session_id, user_id)
    messages = await _get_memory().get_history(session_id, limit=limit)
    return [
        MessageResponse(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            metadata=m.get("metadata", {}),
            created_at=m["created_at"],
        )
        for m in messages
    ]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user: User | None = Depends(get_current_user)):
    user_id = current_user.id if current_user else ""
    await _verify_session_owner(session_id, user_id)
    await _get_memory().delete_session(session_id)
    return {"status": "deleted", "id": session_id}


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, title: str, current_user: User | None = Depends(get_current_user)):
    user_id = current_user.id if current_user else ""
    await _verify_session_owner(session_id, user_id)
    await _get_memory().update_session_title(session_id, title)
    return {"status": "updated", "id": session_id}
