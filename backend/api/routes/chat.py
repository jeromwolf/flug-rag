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


def _extract_tool_content(tool_name: str, result_data) -> str:
    """Extract displayable text from ToolResult.data for any tool."""
    if result_data is None:
        return ""
    if isinstance(result_data, str):
        return result_data
    if isinstance(result_data, dict):
        # LLM 생성 도구: 텍스트 키 직접 추출
        for key in ("draft", "material", "report", "checklist", "summary", "translation", "review"):
            if key in result_data:
                return result_data[key]
        # 이메일: 제목 + 본문 포맷팅
        if "email" in result_data:
            return _format_email(result_data)
        # 계산기
        if "result" in result_data and "expression" in result_data:
            return f"계산 결과: {result_data['expression']} = {result_data['result']}"
        # System DB query → 사람이 읽기 좋은 포맷
        if tool_name == "system_db_query":
            return _format_system_db(result_data)
        # ERP/EHSQ/Groupware → 포맷팅
        if tool_name in ("erp_lookup", "ehsq_lookup", "groupware_lookup"):
            return _format_enterprise_tool(tool_name, result_data)
        # 설비/자재 관리 시스템 (외부 API)
        if tool_name == "asset_management":
            return _format_asset_management(result_data)
        import json as _json
        return _json.dumps(result_data, ensure_ascii=False, indent=2)
    return str(result_data)


def _format_email(data: dict) -> str:
    """Format email_composer result into readable markdown."""
    import re as _re
    email_body = data.get("email", "")
    email_body = email_body.replace("\\n", "\n")
    # LLM이 코드블록으로 감싼 경우 제거 (```text, ```markdown, ``` 등)
    email_body = _re.sub(r'^```\w*\n?', '', email_body)
    email_body = _re.sub(r'\n?```$', '', email_body)
    # 이메일 본문을 마크다운으로 자연스럽게 표시
    lines = ["## ✉️ 이메일 초안\n"]
    for line in email_body.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
        elif stripped.startswith("수신자:") or stripped.startswith("제목:") or stripped.startswith("발신자:"):
            lines.append(f"**{stripped}**")
        elif stripped.startswith("-"):
            lines.append(stripped)
        else:
            lines.append(stripped)
    return "\n".join(lines)


def _format_system_db(data: dict) -> str:
    """Format system_db_query result into readable markdown."""
    summary = data.get("시스템 현황 요약", {})
    lines = ["## 📊 시스템 현황\n"]
    if summary:
        lines.append(f"- **사용자**: {summary.get('사용자', '-')}")
        lines.append(f"- **세션**: {summary.get('세션', '-')}")
        lines.append(f"- **메시지**: {summary.get('메시지', '-')}")
        lines.append(f"- **세션당 평균 대화**: {summary.get('세션당 평균 대화', '-')}")
        lines.append(f"- **감사 로그**: {summary.get('감사 로그', '-')}")
        lines.append("")

    daily = data.get("일별 질의 추이", [])
    if daily:
        lines.append("### 일별 질의 추이")
        lines.append("| 날짜 | 질의 수 |")
        lines.append("|------|---------|")
        for d in daily:
            lines.append(f"| {d.get('날짜', '')} | {d.get('질의 수', 0):,} |")
        lines.append("")

    users = data.get("최근 7일 사용자 활동", [])
    if users:
        lines.append("### 사용자 활동 (최근 7일)")
        lines.append("| 사용자 | 활동 수 | 최근 활동 |")
        lines.append("|--------|---------|----------|")
        for u in users:
            last = u.get("최근 활동", "")[:10]
            lines.append(f"| {u.get('사용자', '')} | {u.get('활동 수', 0):,} | {last} |")
        lines.append("")

    query_time = data.get("조회 시간", "")
    if query_time:
        lines.append(f"*조회 시간: {query_time[:19]}*")

    return "\n".join(lines)


def _format_enterprise_tool(tool_name: str, data: dict) -> str:
    """Format ERP/EHSQ/Groupware results into readable markdown."""
    if tool_name == "erp_lookup":
        return _format_erp(data)
    if tool_name == "ehsq_lookup":
        return _format_ehsq(data)
    if tool_name == "groupware_lookup":
        return _format_groupware(data)
    import json as _json
    return _json.dumps(data, ensure_ascii=False, indent=2)


def _format_erp(data: dict) -> str:
    """Format ERP lookup result."""
    qtype = data.get("query_type", "")
    keyword = data.get("keyword", "")
    result = data.get("result", {})
    lines = ["## 🏢 ERP 조회 결과\n"]

    if qtype == "budget":
        if isinstance(result, dict):
            for k in ("회계연도", "총예산", "집행액", "잔액", "집행률", "부문", "비고"):
                if k in result:
                    lines.append(f"- **{k}**: {result[k]}")
            items = result.get("부문별 현황") or result.get("세부항목") or []
            if items:
                lines.append("\n| 항목 | 예산 | 집행 | 집행률 |")
                lines.append("|------|------|------|--------|")
                for item in items:
                    name = item.get("부문") or item.get("항목", "")
                    lines.append(f"| {name} | {item.get('예산', '-')} | {item.get('집행', '-')} | {item.get('집행률', '-')} |")
    elif qtype == "project":
        items = result if isinstance(result, list) else []
        lines.append(f"**검색 키워드**: {keyword} | **결과**: {len(items)}건\n")
        for p in items:
            lines.append(f"### {p.get('프로젝트명', '')}")
            for k in ("프로젝트코드", "발주처", "계약금액", "진행률", "착공일", "준공예정일", "담당PM", "상태"):
                if k in p:
                    lines.append(f"- **{k}**: {p[k]}")
            lines.append("")
    elif qtype == "vendor":
        items = result if isinstance(result, list) else []
        lines.append(f"**검색 키워드**: {keyword} | **결과**: {len(items)}건\n")
        if items:
            lines.append("| 업체명 | 분류 | 등급 | 등록상태 | 누적계약액 |")
            lines.append("|--------|------|------|----------|------------|")
            for v in items:
                lines.append(f"| {v.get('업체명', '')} | {v.get('분류', '')} | {v.get('평가등급', '')} | {v.get('등록상태', '')} | {v.get('누적계약액', '')} |")

    return "\n".join(lines)


def _format_ehsq(data: dict) -> str:
    """Format EHSQ lookup result."""
    action = data.get("action", "")
    facility = data.get("facility", "전체")
    result = data.get("result", {})
    lines = ["## 🛡️ EHSQ 조회 결과\n"]

    if action == "incident_report":
        summary = result.get("summary", {})
        lines.append(f"**대상**: {facility} | **기준일**: {summary.get('기준일', '-')}\n")
        for k in ("집계기간", "총 사고건수", "중대재해", "경미사고", "아차사고", "무재해 누적일수", "전년 동기 대비"):
            if k in summary:
                lines.append(f"- **{k}**: {summary[k]}")
        incidents = result.get("incidents", [])
        if incidents:
            lines.append("\n### 사고 이력")
            for inc in incidents:
                lines.append(f"\n**{inc.get('사고번호', '')}** ({inc.get('분류', '')})")
                for k in ("발생일", "시설", "내용", "조치상태", "담당자"):
                    if k in inc:
                        lines.append(f"- {k}: {inc[k]}")

    elif action == "safety_status":
        lines.append(f"**대상**: {facility}")
        for k in ("기준일", "안전등급"):
            if k in result:
                lines.append(f"- **{k}**: {result[k]}")
        kpi = result.get("핵심지표", {})
        if kpi:
            lines.append("\n### 핵심 지표")
            for k, v in kpi.items():
                lines.append(f"- **{k}**: {v}")
        facilities = result.get("시설별 상태", [])
        if facilities:
            lines.append("\n### 시설별 안전 등급")
            lines.append("| 시설명 | 등급 | 최근점검일 | 지적사항 |")
            lines.append("|--------|------|------------|----------|")
            for f in facilities:
                lines.append(f"| {f.get('시설명', '')} | {f.get('등급', '')} | {f.get('최근점검일', '')} | {f.get('지적사항', 0)} |")
        upcoming = result.get("이달 예정 점검", [])
        if upcoming:
            lines.append("\n### 이달 예정 점검")
            for u in upcoming:
                lines.append(f"- **{u.get('시설명', '')}**: {u.get('예정일', '')} ({u.get('점검유형', '')})")

    elif action == "compliance_check":
        lines.append(f"**대상**: {facility}")
        for k in ("기준일", "전체 준수율"):
            if k in result:
                lines.append(f"- **{k}**: {result[k]}")
        obligations = result.get("법정의무 이행", [])
        if obligations:
            lines.append("\n### 법정의무 이행 현황")
            lines.append("| 항목 | 기한 | 상태 |")
            lines.append("|------|------|------|")
            for o in obligations:
                lines.append(f"| {o.get('항목', '')} | {o.get('기한', '')} | {o.get('상태', '')} |")
        violations = result.get("미이행 항목", [])
        if violations:
            lines.append("\n### ⚠️ 미이행 항목")
            for v in violations:
                lines.append(f"- **{v.get('항목', '')}** (기한: {v.get('기한', '')})")
                lines.append(f"  - 사유: {v.get('사유', '')}")
                lines.append(f"  - 조치계획: {v.get('조치계획', '')}")
        note = result.get("비고")
        if note:
            lines.append(f"\n> {note}")

    return "\n".join(lines)


def _format_groupware(data: dict) -> str:
    """Format Groupware lookup result."""
    action = data.get("action", "")
    keyword = data.get("keyword", "")
    result = data.get("result", [])
    lines = ["## 📅 그룹웨어 조회 결과\n"]

    if action == "schedule":
        lines.append(f"**검색**: {keyword} | **결과**: {len(result)}건\n")
        for s in result:
            lines.append(f"### 📌 {s.get('제목', '')}")
            lines.append(f"- **일시**: {s.get('일시', '')}")
            lines.append(f"- **장소**: {s.get('장소', '')}")
            lines.append(f"- **주관**: {s.get('주관', '')}")
            participants = s.get("참석", [])
            if participants:
                lines.append(f"- **참석**: {', '.join(participants)}")
            if s.get("비고"):
                lines.append(f"- **비고**: {s['비고']}")
            lines.append("")

    elif action == "approval":
        summary = data.get("summary", {})
        lines.append(f"**검색**: {keyword} | 진행중 {summary.get('진행중', 0)} / 완료 {summary.get('완료', 0)} / 반려 {summary.get('반려', 0)}\n")
        for a in result:
            status_icon = {"진행중": "🔄", "완료": "✅", "반려": "❌"}.get(a.get("상태", ""), "")
            lines.append(f"### {status_icon} {a.get('문서명', '')}")
            lines.append(f"- **결재번호**: {a.get('결재번호', '')}")
            lines.append(f"- **기안자**: {a.get('기안자', '')} | **기안일**: {a.get('기안일', '')}")
            if a.get("결재금액") and a["결재금액"] != "없음":
                lines.append(f"- **결재금액**: {a['결재금액']}")
            lines.append(f"- **현재단계**: {a.get('현재단계', '')}")
            if a.get("반려사유"):
                lines.append(f"- **반려사유**: {a['반려사유']}")
            lines.append("")

    elif action == "notice":
        lines.append(f"**검색**: {keyword} | **결과**: {len(result)}건\n")
        for n in result:
            urgency = {"긴급": "🔴", "중요": "🟡", "일반": "🟢"}.get(n.get("중요도", ""), "")
            lines.append(f"### {urgency} {n.get('제목', '')}")
            lines.append(f"- **작성부서**: {n.get('작성부서', '')} | **게시일**: {n.get('게시일', '')}")
            if n.get("마감일"):
                lines.append(f"- **마감일**: {n['마감일']}")
            lines.append(f"- {n.get('내용요약', '')}")
            attachments = n.get("첨부", [])
            if attachments:
                lines.append(f"- **첨부**: {', '.join(attachments)}")
            lines.append("")

    return "\n".join(lines)


def _format_asset_management(data: dict) -> str:
    """Format asset_management (외부 API) result into readable markdown."""
    action = data.get("action", "")
    period = data.get("조회기간", "")
    keyword = data.get("keyword", "")
    total = data.get("total", 0)
    count = data.get("반환건수", 0)
    result = data.get("result", [])

    lines = ["## 📦 설비/자재 관리 시스템 조회 결과\n"]
    lines.append(f"**조회기간**: {period} | **총 자재**: {total:,}건 | **반환**: {count}건")
    if keyword:
        lines.append(f" | **검색어**: {keyword}")
    lines.append("\n")

    if action == "detail" and isinstance(result, dict):
        # 단일 자재 상세
        for k, v in result.items():
            if k == "조회기간":
                continue
            lines.append(f"- **{k}**: {v}")
    elif isinstance(result, list) and result:
        # 테이블로 표시
        lines.append("| 자재코드 | 자재명 | 제조사 | 분류 | 단가(USD) | 재고 | 재고상태 |")
        lines.append("|----------|--------|--------|------|-----------|------|----------|")
        for item in result:
            code = item.get("자재코드", "")
            name = item.get("자재명", "")
            brand = item.get("제조사", "")
            cat = item.get("분류", "")
            price = item.get("단가(USD)", 0)
            stock = item.get("재고수량", 0)
            status = item.get("재고상태", "")
            lines.append(f"| {code} | {name} | {brand} | {cat} | ${price:,.2f} | {stock} | {status} |")
    else:
        lines.append("조회 결과가 없습니다.")

    lines.append("\n> *외부 설비/자재 관리 API 실시간 연동 결과입니다.*")
    return "\n".join(lines)


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


async def _generate_follow_up_questions(
    question: str, answer: str
) -> list[str]:
    """Generate 3 follow-up question suggestions based on Q&A context."""
    from core.llm.factory import create_llm

    llm = create_llm()
    prompt = (
        "다음 질문과 답변을 보고, 사용자가 이어서 물어볼 만한 후속 질문 3개를 생성하세요.\n"
        "각 질문은 한 줄에 하나씩, 번호 없이 질문만 작성하세요.\n"
        "질문은 구체적이고 원래 주제와 관련되어야 합니다.\n\n"
        f"질문: {question}\n"
        f"답변: {answer[:500]}\n\n"
        "후속 질문 3개:"
    )
    try:
        result = await llm.agenerate(prompt, max_tokens=200, temperature=0.7)
        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in result.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return lines[:3]
    except Exception as e:
        logger.debug("Follow-up generation failed: %s", e)
        return []


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


@router.get("/chat/config")
async def chat_config():
    """Public client config for chat (OCR limits, file size limits, etc.)."""
    from config.settings import settings
    return {
        "ocr_max_chars": settings.ocr_max_chars,
        "file_max_size_mb": 50,
    }


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

    # Pre-classify query (rule-based, <1ms)
    from rag.query_classifier import QueryClassifier, QueryClass

    classifier = QueryClassifier()
    classification = classifier.classify(request.message)
    query_class_value = classification.category.value

    # Immediate response for identity/dangerous (no LLM needed)
    if classification.category in (QueryClass.IDENTITY, QueryClass.DANGEROUS):
        content = classification.immediate_response
        msg_id = await _get_memory().add_message(session_id, "assistant", content)
        return ChatResponse(
            message_id=msg_id,
            content=content,
            sources=[],
            confidence=1.0 if classification.category == QueryClass.IDENTITY else 0.0,
            confidence_level="high" if classification.category == QueryClass.IDENTITY else "low",
            session_id=session_id,
            metadata={"query_class": query_class_value},
        )

    # Tool selection (keyword-based, before LLM routing)
    from agent.tool_selector import select_tool

    tool_selection = select_tool(request.message) if request.mode != "direct" else None
    routing = None

    if tool_selection:
        mode = "tool"
    elif classification.category in (QueryClass.CHITCHAT, QueryClass.GENERAL):
        # Skip QueryRouter for non-RAG queries
        mode = "direct"
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
            content = _extract_tool_content(tool_selection.tool_name, result.data)
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

    # Agent pipeline for complex / tool-required queries
    if routing and routing.category.value in ("complex_task", "tool_required"):
        from agent.planner import TaskPlanner
        from agent.executor import PlanExecutor

        planner = TaskPlanner(llm=rag_chain.llm)
        plan = await planner.plan(request.message)
        executor = PlanExecutor(rag_chain=rag_chain)
        content = await executor.execute(
            plan,
            filters=merged_filters,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
        )
        msg_id = await _get_memory().add_message(session_id, "assistant", content)
        return ChatResponse(
            message_id=msg_id,
            content=content,
            sources=[],
            confidence=0.8,
            confidence_level="high",
            session_id=session_id,
            metadata={"agent_pipeline": True, "plan_steps": len(plan.steps)},
        )

    # Generate response (RAG / direct)
    response = await rag_chain.query(
        question=request.message,
        mode=mode,
        filters=merged_filters,
        provider=request.provider,
        model=request.model,
        temperature=request.temperature,
        query_class=query_class_value,
    )

    # Add query_class to metadata
    response.metadata["query_class"] = query_class_value

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

    # Pre-classify query (rule-based, <1ms)
    from rag.query_classifier import QueryClassifier, QueryClass

    classifier = QueryClassifier()
    classification = classifier.classify(request.message)
    query_class_value = classification.category.value

    # Immediate response for identity/dangerous (no LLM needed)
    if classification.category in (QueryClass.IDENTITY, QueryClass.DANGEROUS):
        content = classification.immediate_response

        async def immediate_generator():
            msg_id = str(uuid.uuid4())
            await _get_memory().add_message(session_id, "assistant", content)
            yield {
                "event": "start",
                "data": json.dumps({"message_id": msg_id, "session_id": session_id}, ensure_ascii=False),
            }
            yield {"event": "chunk", "data": json.dumps({"content": content}, ensure_ascii=False)}
            yield {
                "event": "end",
                "data": json.dumps({
                    "confidence_score": 1.0 if classification.category == QueryClass.IDENTITY else 0.0,
                    "confidence_level": "high" if classification.category == QueryClass.IDENTITY else "low",
                    "query_class": query_class_value,
                }, ensure_ascii=False),
            }

        return EventSourceResponse(immediate_generator())

    # Tool selection (keyword-based, before LLM routing)
    from agent.tool_selector import select_tool

    tool_selection = select_tool(request.message) if request.mode != "direct" else None
    routing = None

    if tool_selection:
        mode = "tool"
        # 이전 대화 참조 ("이 내용으로", "위 내용" 등) → 마지막 어시스턴트 응답을 도구 컨텍스트에 주입
        _context_refs = ("이 내용", "위 내용", "위의 내용", "이걸로", "이것을", "이것으로", "해당 내용", "이 문서", "이 결과", "위 결과", "조회 결과")
        if any(ref in request.message for ref in _context_refs) and history:
            last_assistant = next(
                (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
                None,
            )
            if last_assistant:
                prev_context = last_assistant[:3000]  # 컨텍스트 길이 제한
                args = tool_selection.arguments
                # 도구별 컨텍스트 주입 필드
                if "body_context" in args:
                    args["body_context"] = f"[이전 조회 결과]\n{prev_context}\n\n[사용자 요청]\n{request.message}"
                elif "topic" in args:
                    args["topic"] = f"[이전 조회 결과]\n{prev_context}\n\n[사용자 요청]\n{request.message}"
                elif "text" in args:
                    args["text"] = f"[이전 조회 결과]\n{prev_context}\n\n[사용자 요청]\n{request.message}"
                elif "document_text" in args:
                    args["document_text"] = f"[이전 조회 결과]\n{prev_context}\n\n[사용자 요청]\n{request.message}"
    elif classification.category in (QueryClass.CHITCHAT, QueryClass.GENERAL):
        # Skip QueryRouter for non-RAG queries
        mode = "direct"
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
        end_metadata = {}
        source_list = []
        try:
            if routing and routing.category.value in ("complex_task", "tool_required"):
                # ── Agent pipeline: plan → execute steps → stream final ──
                from agent.planner import TaskPlanner
                from agent.executor import PlanExecutor

                planner = TaskPlanner(llm=rag_chain.llm)
                plan = await planner.plan(request.message)
                executor = PlanExecutor(rag_chain=rag_chain)
                async for evt in executor.stream_execute(
                    plan,
                    filters=merged_filters,
                    provider=request.provider,
                    model=request.model,
                    temperature=request.temperature,
                ):
                    event_type = evt.get("event", "")
                    event_data = evt.get("data", {})
                    if event_type == "start":
                        # Skip executor's start event; we already sent one
                        continue
                    if event_type == "chunk":
                        full_content += event_data.get("content", "")
                    yield {
                        "event": event_type,
                        "data": json.dumps(event_data, ensure_ascii=False),
                    }
            elif mode == "tool":
                # ── MCP tool execution with chunked streaming ──
                from agent.mcp.registry import get_registry

                registry = get_registry()
                tool_label = {
                    "report_draft": "보고서 초안",
                    "training_material": "교육자료",
                    "regulation_review": "규정 검토",
                    "safety_checklist": "안전 체크리스트",
                    "calculator": "수식 계산",
                    "data_analyzer": "데이터 분석",
                    "asset_management": "설비/자재 조회",
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
                    full_content = _extract_tool_content(tool_selection.tool_name, result.data)

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
                    query_class=query_class_value,
                ):
                    if event["event"] == "start":
                        # Skip the rag_chain's own start event; we already sent one
                        continue
                    if event["event"] == "chunk":
                        full_content += event["data"].get("content", "")
                    if event["event"] == "end":
                        end_metadata = event["data"]
                    if event["event"] == "source":
                        source_list.append(event["data"])
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
            msg_meta = {}
            if end_metadata:
                msg_meta["confidence"] = end_metadata.get("confidence_score")
                msg_meta["latency_ms"] = end_metadata.get("latency_ms")
                msg_meta["model"] = end_metadata.get("model_tier")
            if source_list:
                msg_meta["sources"] = source_list
            await _get_memory().add_message(session_id, "assistant", full_content, metadata=msg_meta if msg_meta else None)

        # Generate follow-up question suggestions (best-effort, non-blocking)
        if full_content and request.message:
            try:
                suggestions = await _generate_follow_up_questions(
                    request.message, full_content
                )
                if suggestions:
                    yield {
                        "event": "suggested_questions",
                        "data": json.dumps(
                            {"questions": suggestions}, ensure_ascii=False
                        ),
                    }
            except Exception as e:
                logger.debug("Follow-up question generation skipped: %s", e)

    return EventSourceResponse(event_generator())
