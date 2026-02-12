"""Pre-built workflow templates for common patterns."""

from agent.builder.models import (
    Edge, NodeConfig, NodeType, Workflow, WorkflowNode, WorkflowStatus,
)


def create_simple_rag_workflow() -> Workflow:
    """Simple RAG: query → retrieve → generate → output."""
    start = WorkflowNode(id="start", config=NodeConfig(
        node_type=NodeType.START, label="시작",
        position={"x": 0, "y": 0},
    ))
    retrieve = WorkflowNode(id="retrieve", config=NodeConfig(
        node_type=NodeType.RETRIEVAL, label="문서 검색",
        config={"top_k": 5},
        position={"x": 200, "y": 0},
    ))
    output = WorkflowNode(id="output", config=NodeConfig(
        node_type=NodeType.OUTPUT, label="출력",
        position={"x": 400, "y": 0},
    ))

    return Workflow(
        name="간단 RAG",
        description="질문 → 문서 검색 → 답변 생성",
        nodes=[start, retrieve, output],
        edges=[
            Edge(source="start", target="retrieve"),
            Edge(source="retrieve", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


def create_routing_workflow() -> Workflow:
    """Routing workflow: classify → branch → execute."""
    start = WorkflowNode(id="start", config=NodeConfig(
        node_type=NodeType.START, label="시작",
    ))
    classify = WorkflowNode(id="classify", config=NodeConfig(
        node_type=NodeType.LLM, label="질문 분류",
        config={
            "system_prompt": "사용자 질문을 분류하세요: document_search 또는 general_query",
            "prompt_template": "질문 분류: {input}",
            "temperature": 0.1,
        },
    ))
    rag = WorkflowNode(id="rag", config=NodeConfig(
        node_type=NodeType.RETRIEVAL, label="문서 검색",
        config={"top_k": 5},
    ))
    direct = WorkflowNode(id="direct", config=NodeConfig(
        node_type=NodeType.LLM, label="직접 답변",
        config={"system_prompt": "한국어로 친절하게 답변하세요.", "prompt_template": "{input}"},
    ))
    output = WorkflowNode(id="output", config=NodeConfig(
        node_type=NodeType.OUTPUT, label="출력",
    ))

    return Workflow(
        name="라우팅 워크플로우",
        description="질문 분류 → 문서검색/직접답변 분기 → 출력",
        nodes=[start, classify, rag, direct, output],
        edges=[
            Edge(source="start", target="classify"),
            Edge(source="classify", target="rag", condition="true"),
            Edge(source="classify", target="direct", condition="false"),
            Edge(source="rag", target="output"),
            Edge(source="direct", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


def create_quality_check_workflow() -> Workflow:
    """Quality check: retrieve → check confidence → branch."""
    start = WorkflowNode(id="start", config=NodeConfig(
        node_type=NodeType.START, label="시작",
    ))
    retrieve = WorkflowNode(id="retrieve", config=NodeConfig(
        node_type=NodeType.RETRIEVAL, label="문서 검색",
        config={"top_k": 5},
    ))
    check = WorkflowNode(id="check", config=NodeConfig(
        node_type=NodeType.CONDITION, label="신뢰도 확인",
        config={"condition_type": "confidence", "threshold": 0.5},
    ))
    high_output = WorkflowNode(id="high_output", config=NodeConfig(
        node_type=NodeType.OUTPUT, label="정상 출력",
    ))
    low_transform = WorkflowNode(id="low_transform", config=NodeConfig(
        node_type=NodeType.TRANSFORM, label="안전장치 적용",
        config={"template": "⚠️ 확인이 필요한 답변입니다.\n\n{input}"},
    ))
    low_output = WorkflowNode(id="low_output", config=NodeConfig(
        node_type=NodeType.OUTPUT, label="경고 출력",
    ))

    return Workflow(
        name="품질 검증 워크플로우",
        description="문서 검색 → 신뢰도 확인 → 조건부 출력",
        nodes=[start, retrieve, check, high_output, low_transform, low_output],
        edges=[
            Edge(source="start", target="retrieve"),
            Edge(source="retrieve", target="check"),
            Edge(source="check", target="high_output", condition="true"),
            Edge(source="check", target="low_transform", condition="false"),
            Edge(source="low_transform", target="low_output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


def create_regulation_review_workflow() -> Workflow:
    """규정 검토 워크플로우: 문서 입력 → 규정 검색 → 대조 분석 → 검토 보고서."""
    start = WorkflowNode(id="start", config=NodeConfig(
        node_type=NodeType.START, label="문서 입력",
        config={"description": "검토할 문서를 입력하세요"},
    ))
    regulation_search = WorkflowNode(id="regulation_search", config=NodeConfig(
        node_type=NodeType.TOOL, label="규정 검토",
        config={
            "tool_name": "regulation_review",
            "arguments_template": {
                "document_text": "{{input}}",
                "regulation_category": "전체",
                "review_depth": "standard",
            },
        },
        position={"x": 250, "y": 0},
    ))
    output = WorkflowNode(id="output", config=NodeConfig(
        node_type=NodeType.OUTPUT, label="검토 보고서",
        config={"format": "markdown"},
        position={"x": 500, "y": 0},
    ))

    return Workflow(
        name="규정 검토 에이전트",
        description="업로드 문서를 사내 규정과 대조하여 위반/불일치 항목을 식별합니다.",
        nodes=[start, regulation_search, output],
        edges=[
            Edge(source="start", target="regulation_search"),
            Edge(source="regulation_search", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


def create_safety_checklist_workflow() -> Workflow:
    """안전 체크리스트 워크플로우: 설비 유형 → 규정 검색 → 체크리스트 생성."""
    start = WorkflowNode(id="start", config=NodeConfig(
        node_type=NodeType.START, label="설비 유형 선택",
        config={"description": "점검 대상 설비 유형을 입력하세요"},
    ))
    checklist_gen = WorkflowNode(id="checklist_gen", config=NodeConfig(
        node_type=NodeType.TOOL, label="체크리스트 생성",
        config={
            "tool_name": "safety_checklist",
            "arguments_template": {
                "equipment_type": "{{input}}",
                "output_format": "markdown",
            },
        },
        position={"x": 250, "y": 0},
    ))
    output = WorkflowNode(id="output", config=NodeConfig(
        node_type=NodeType.OUTPUT, label="체크리스트 출력",
        config={"format": "markdown"},
        position={"x": 500, "y": 0},
    ))

    return Workflow(
        name="안전 체크리스트 생성",
        description="설비 유형별 안전 점검 체크리스트와 관련 규정을 자동 생성합니다.",
        nodes=[start, checklist_gen, output],
        edges=[
            Edge(source="start", target="checklist_gen"),
            Edge(source="checklist_gen", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


PRESET_WORKFLOWS = {
    "simple_rag": create_simple_rag_workflow,
    "routing": create_routing_workflow,
    "quality_check": create_quality_check_workflow,
    "regulation_review": create_regulation_review_workflow,
    "safety_checklist": create_safety_checklist_workflow,
}


def list_presets() -> list[dict]:
    """List available preset workflows."""
    return [
        {"id": key, "name": fn().name, "description": fn().description}
        for key, fn in PRESET_WORKFLOWS.items()
    ]


def get_preset(name: str) -> Workflow | None:
    """Get a preset workflow by name."""
    factory = PRESET_WORKFLOWS.get(name)
    return factory() if factory else None
