"""Chat endpoints with SSE streaming support."""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from agent import QueryRouter, get_memory
from api.schemas import ChatRequest, ChatResponse
from auth.dependencies import get_current_user
from auth.models import User
from rag import RAGChain
from rag.access_control import get_access_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared instances with thread-safe initialization
_memory = None
_router = None
_rag_chain = None
_init_lock = asyncio.Lock()


def _get_memory():
    global _memory
    if _memory is None:
        _memory = get_memory()
    return _memory


async def _get_router():
    global _router
    if _router is None:
        async with _init_lock:
            if _router is None:
                _router = QueryRouter()
    return _router


async def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        async with _init_lock:
            if _rag_chain is None:
                _rag_chain = RAGChain()
    return _rag_chain


async def _build_merged_filters(user, request_filters):
    """사용자 접근 권한 필터와 요청 필터를 병합."""
    if user is None:
        return request_filters

    try:
        manager = await get_access_manager()
        access_filter = await manager.build_access_filter(
            user_id=user.id,
            user_role=user.role.value.upper(),
            user_department=user.department or None,
        )
    except Exception as e:
        logger.error("Access filter build failed (denying access): %s", e)
        return {"folder_id": "__no_access__"}

    # ADMIN gets no filter (full access)
    if access_filter is None:
        return request_filters

    # Include user's personal documents alongside folder-based access
    personal_filter = {
        "$and": [
            {"user_id": user.id},
            {"owner_type": "personal"},
        ]
    }
    combined_filter = {"$or": [access_filter, personal_filter]}

    if request_filters is None:
        return combined_filter

    # Merge: combined access + request filters
    return {"$and": [combined_filter, request_filters]}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: User | None = Depends(get_current_user)):
    """Non-streaming chat endpoint."""
    # Create or get session
    session_id = request.session_id
    if not session_id:
        session_id = await _get_memory().create_session(title=request.message[:50])

    # Save user message
    await _get_memory().add_message(session_id, "user", request.message)

    # Get history
    history = await _get_memory().get_history(session_id)

    # Route query
    rag_chain = await _get_rag_chain()
    query_router = await _get_router()

    if request.mode == "auto":
        routing = await query_router.route(request.message, history)
        mode = (
            "rag"
            if routing.category.value in ("document_search", "complex_task")
            else "direct"
        )
    else:
        mode = request.mode

    # Build merged filters (access control + request filters)
    merged_filters = await _build_merged_filters(current_user, request.filters)

    # Generate response
    response = await rag_chain.query(
        question=request.message,
        mode=mode,
        filters=merged_filters,
        provider=request.provider,
        model=request.model,
        temperature=request.temperature,
    )

    # Save assistant message
    msg_id = await _get_memory().add_message(
        session_id,
        "assistant",
        response.content,
        metadata=response.metadata,
    )

    return ChatResponse(
        message_id=msg_id,
        content=response.content,
        sources=response.sources,
        confidence=response.confidence,
        confidence_level=response.confidence_level,
        safety_warning=response.safety_warning,
        session_id=session_id,
        metadata=response.metadata,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, current_user: User | None = Depends(get_current_user)):
    """SSE streaming chat endpoint."""
    session_id = request.session_id
    if not session_id:
        session_id = await _get_memory().create_session(title=request.message[:50])

    await _get_memory().add_message(session_id, "user", request.message)
    history = await _get_memory().get_history(session_id)

    rag_chain = await _get_rag_chain()
    query_router = await _get_router()

    if request.mode == "auto":
        routing = await query_router.route(request.message, history)
        mode = (
            "rag"
            if routing.category.value in ("document_search", "complex_task")
            else "direct"
        )
    else:
        mode = request.mode

    # Build merged filters (access control + request filters)
    merged_filters = await _build_merged_filters(current_user, request.filters)

    async def event_generator():
        full_content = ""
        async for event in rag_chain.stream_query(
            question=request.message,
            mode=mode,
            filters=merged_filters,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
        ):
            if event["event"] == "chunk":
                full_content += event["data"].get("content", "")
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

        # Save assistant message after streaming
        await _get_memory().add_message(session_id, "assistant", full_content)

    return EventSourceResponse(event_generator())
