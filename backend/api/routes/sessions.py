"""Session and conversation history endpoints."""

from fastapi import APIRouter, HTTPException

from agent import ConversationMemory
from api.schemas import (
    MessageResponse,
    SessionCreate,
    SessionListResponse,
    SessionResponse,
)

router = APIRouter()
_memory = ConversationMemory()


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    session_id = await _memory.create_session(title=request.title)
    session = await _memory.get_session(session_id)
    return SessionResponse(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = 50, offset: int = 0):
    sessions = await _memory.get_sessions(limit=limit, offset=offset)
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
async def get_session(session_id: str):
    session = await _memory.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(session_id: str, limit: int = 50):
    messages = await _memory.get_history(session_id, limit=limit)
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
async def delete_session(session_id: str):
    await _memory.delete_session(session_id)
    return {"status": "deleted", "id": session_id}


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, title: str):
    await _memory.update_session_title(session_id, title)
    return {"status": "updated", "id": session_id}
