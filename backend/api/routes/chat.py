"""Chat endpoints with SSE streaming support."""

import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from agent import ConversationMemory, QueryRouter
from api.schemas import ChatRequest, ChatResponse
from rag import RAGChain

router = APIRouter()

# Shared instances (will be proper DI later)
_memory = ConversationMemory()
_router = None
_rag_chain = None


def _get_router():
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    # Create or get session
    session_id = request.session_id
    if not session_id:
        session_id = await _memory.create_session(title=request.message[:50])

    # Save user message
    await _memory.add_message(session_id, "user", request.message)

    # Get history
    history = await _memory.get_history(session_id)

    # Route query
    rag_chain = _get_rag_chain()
    query_router = _get_router()

    if request.mode == "auto":
        routing = await query_router.route(request.message, history)
        mode = (
            "rag"
            if routing.category.value in ("document_search", "complex_task")
            else "direct"
        )
    else:
        mode = request.mode

    # Generate response
    response = await rag_chain.query(
        question=request.message,
        mode=mode,
        filters=request.filters,
        provider=request.provider,
        model=request.model,
    )

    # Save assistant message
    msg_id = await _memory.add_message(
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
async def chat_stream(request: ChatRequest):
    """SSE streaming chat endpoint."""
    session_id = request.session_id
    if not session_id:
        session_id = await _memory.create_session(title=request.message[:50])

    await _memory.add_message(session_id, "user", request.message)
    history = await _memory.get_history(session_id)

    rag_chain = _get_rag_chain()
    query_router = _get_router()

    if request.mode == "auto":
        routing = await query_router.route(request.message, history)
        mode = (
            "rag"
            if routing.category.value in ("document_search", "complex_task")
            else "direct"
        )
    else:
        mode = request.mode

    async def event_generator():
        full_content = ""
        async for event in rag_chain.stream_query(
            question=request.message,
            mode=mode,
            filters=request.filters,
            provider=request.provider,
            model=request.model,
        ):
            if event["event"] == "chunk":
                full_content += event["data"].get("content", "")
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

        # Save assistant message after streaming
        await _memory.add_message(session_id, "assistant", full_content)

    return EventSourceResponse(event_generator())
