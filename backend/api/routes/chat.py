"""Chat endpoints with SSE streaming support."""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from agent import QueryRouter, get_memory
from api.schemas import ChatRequest, ChatResponse
from auth.dependencies import get_current_user, require_password_changed
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
async def chat(request: ChatRequest, current_user: User | None = Depends(require_password_changed)):
    """Non-streaming chat endpoint."""
    # Create or get session
    session_id = request.session_id
    if not session_id:
        user_id = current_user.id if current_user else ""
        session_id = await _get_memory().create_session(title=request.message[:50], user_id=user_id)

    # Save user message
    await _get_memory().add_message(session_id, "user", request.message)

    # Get history
    history = await _get_memory().get_history(session_id)

    # Route query
    rag_chain = await _get_rag_chain()
    query_router = await _get_router()

    # Tool selection (keyword-based, before LLM routing)
    from agent.tool_selector import select_tool

    tool_selection = select_tool(request.message) if request.mode != "direct" else None

    if tool_selection:
        mode = "tool"
    elif request.mode == "auto":
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

    # Tool execution path
    if mode == "tool":
        from agent.mcp.registry import get_registry

        registry = get_registry()
        result = await registry.execute(tool_selection.tool_name, **tool_selection.arguments)

        if result.success:
            content_key = next(
                (k for k in ("draft", "material") if k in (result.data or {})),
                None,
            )
            content = result.data.get(content_key, str(result.data)) if content_key else str(result.data)
        else:
            content = result.error or "도구 실행 중 오류가 발생했습니다."

        msg_id = await _get_memory().add_message(session_id, "assistant", content)
        return ChatResponse(
            message_id=msg_id,
            content=content,
            sources=[],
            confidence=0.9 if result.success else 0.0,
            confidence_level="high" if result.success else "low",
            session_id=session_id,
            metadata={"tool_used": tool_selection.tool_name, **(result.metadata or {})},
        )

    # Generate response (RAG / direct)
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
async def chat_stream(request: ChatRequest, current_user: User | None = Depends(require_password_changed)):
    """SSE streaming chat endpoint."""
    session_id = request.session_id
    if not session_id:
        user_id = current_user.id if current_user else ""
        session_id = await _get_memory().create_session(title=request.message[:50], user_id=user_id)

    await _get_memory().add_message(session_id, "user", request.message)
    history = await _get_memory().get_history(session_id)

    rag_chain = await _get_rag_chain()
    query_router = await _get_router()

    # Tool selection (keyword-based, before LLM routing)
    from agent.tool_selector import select_tool

    tool_selection = select_tool(request.message) if request.mode != "direct" else None

    if tool_selection:
        mode = "tool"
    elif request.mode == "auto":
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
        # Send start event with session_id so the frontend can track the session
        msg_id = str(uuid.uuid4())
        yield {
            "event": "start",
            "data": json.dumps({"message_id": msg_id, "session_id": session_id}, ensure_ascii=False),
        }

        full_content = ""
        try:
            if mode == "tool":
                # ── MCP tool execution with chunked streaming ──
                from agent.mcp.registry import get_registry

                registry = get_registry()
                tool_label = {
                    "report_draft": "보고서 초안",
                    "training_material": "교육자료",
                }.get(tool_selection.tool_name, tool_selection.tool_name)

                yield {
                    "event": "tool_start",
                    "data": json.dumps(
                        {
                            "tool_name": tool_selection.tool_name,
                            "message": f"{tool_label}을(를) 생성 중입니다...",
                        },
                        ensure_ascii=False,
                    ),
                }

                result = await registry.execute(
                    tool_selection.tool_name, **tool_selection.arguments
                )

                if result.success:
                    content_key = next(
                        (k for k in ("draft", "material") if k in (result.data or {})),
                        None,
                    )
                    full_content = (
                        result.data.get(content_key, str(result.data))
                        if content_key
                        else str(result.data)
                    )

                    # Stream in chunks for smooth UX
                    for i in range(0, len(full_content), 80):
                        yield {
                            "event": "chunk",
                            "data": json.dumps(
                                {"content": full_content[i : i + 80]},
                                ensure_ascii=False,
                            ),
                        }

                    # Emit sources from tool metadata
                    for src in (result.metadata or {}).get("sources", []):
                        yield {
                            "event": "source",
                            "data": json.dumps(
                                {"filename": src, "content": "", "score": 0},
                                ensure_ascii=False,
                            ),
                        }

                    yield {
                        "event": "tool_end",
                        "data": json.dumps(
                            {"tool_name": tool_selection.tool_name, "success": True},
                            ensure_ascii=False,
                        ),
                    }
                    yield {
                        "event": "end",
                        "data": json.dumps(
                            {"confidence_score": 0.9, "latency_ms": 0},
                            ensure_ascii=False,
                        ),
                    }
                else:
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "message": result.error
                                or "도구 실행 중 오류가 발생했습니다.",
                            },
                            ensure_ascii=False,
                        ),
                    }
            else:
                # ── Existing RAG / direct streaming flow ──
                async for event in rag_chain.stream_query(
                    question=request.message,
                    mode=mode,
                    filters=merged_filters,
                    provider=request.provider,
                    model=request.model,
                    temperature=request.temperature,
                ):
                    if event["event"] == "start":
                        # Skip the rag_chain's own start event; we already sent one
                        continue
                    if event["event"] == "chunk":
                        full_content += event["data"].get("content", "")
                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"], ensure_ascii=False),
                    }
        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"message": "응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."},
                    ensure_ascii=False,
                ),
            }

        # Save assistant message after streaming (only if content was produced)
        if full_content:
            await _get_memory().add_message(session_id, "assistant", full_content)

    return EventSourceResponse(event_generator())
