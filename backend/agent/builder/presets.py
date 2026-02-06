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


PRESET_WORKFLOWS = {
    "simple_rag": create_simple_rag_workflow,
    "routing": create_routing_workflow,
    "quality_check": create_quality_check_workflow,
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
