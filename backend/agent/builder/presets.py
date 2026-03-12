"""Pre-built workflow templates for 6 core Agent scenarios.

Each workflow is a DAG with proper node types, tool parameters,
and KOGAS domain-specific prompts.
"""

from agent.builder.models import (
    Edge, NodeConfig, NodeType, Workflow, WorkflowNode, WorkflowStatus,
)
from config.settings import settings


# ---------------------------------------------------------------------------
# 1. 기술문서 검색·요약
# ---------------------------------------------------------------------------

def create_tech_doc_summary_workflow() -> Workflow:
    """기술문서 검색·요약: 검색 → 신뢰도 확인 → 요약 또는 안내 → 출력."""
    return Workflow(
        name="기술문서 검색·요약",
        description="기술문서를 검색하고 신뢰도를 검증한 뒤 핵심 내용을 구조화된 요약 보고서로 생성합니다.",
        nodes=[
            WorkflowNode(id="start", config=NodeConfig(
                node_type=NodeType.START, label="검색어 입력",
                config={"description": "요약할 기술문서 주제를 입력하세요 (예: 배관 용접 검사 기준)"},
                position={"x": 40, "y": 220},
            )),
            WorkflowNode(id="retrieve", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="기술문서 검색",
                config={"top_k": 7},
                position={"x": 260, "y": 220},
            )),
            WorkflowNode(id="confidence_check", config=NodeConfig(
                node_type=NodeType.CONDITION, label="검색 품질 확인",
                config={"condition_type": "confidence", "threshold": 0.3},
                position={"x": 480, "y": 220},
            )),
            WorkflowNode(id="summarize", config=NodeConfig(
                node_type=NodeType.LLM, label="요약 보고서 생성",
                config={
                    "system_prompt": (
                        f"당신은 {settings.platform_name}의 기술문서 분석 전문가입니다.\n"
                        "검색된 기술문서를 바탕으로 다음 구조로 요약하세요:\n\n"
                        "## 핵심 요약\n- 3줄 이내 핵심 내용\n\n"
                        "## 주요 사항\n- 번호 매겨서 핵심 포인트 나열\n\n"
                        "## 관련 규정\n- 적용되는 내부규정 및 조항 명시\n\n"
                        "## 참고 사항\n- 추가 확인이 필요한 부분"
                    ),
                    "temperature": 0.1,
                },
                position={"x": 700, "y": 120},
            )),
            WorkflowNode(id="fallback", config=NodeConfig(
                node_type=NodeType.TRANSFORM, label="검색 결과 부족 안내",
                config={"template": "검색된 문서의 신뢰도가 낮습니다.\n\n검색 결과:\n{input}\n\n더 구체적인 키워드로 다시 검색하거나, 관련 문서가 등록되어 있는지 확인해 주세요."},
                position={"x": 700, "y": 340},
            )),
            WorkflowNode(id="output", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="요약 보고서",
                config={"format": "markdown"},
                position={"x": 940, "y": 120},
            )),
            WorkflowNode(id="output_low", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="안내 출력",
                config={"format": "markdown"},
                position={"x": 940, "y": 340},
            )),
        ],
        edges=[
            Edge(source="start", target="retrieve"),
            Edge(source="retrieve", target="confidence_check"),
            Edge(source="confidence_check", target="summarize", condition="true", label="신뢰도 충분"),
            Edge(source="confidence_check", target="fallback", condition="false", label="신뢰도 부족"),
            Edge(source="summarize", target="output"),
            Edge(source="fallback", target="output_low"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# 2. 규정 비교 분석
# ---------------------------------------------------------------------------

def create_regulation_comparison_workflow() -> Workflow:
    """규정 비교 분석: 두 규정 병렬 검색 → 비교 분석 → 컴플라이언스 검토 → 출력."""
    return Workflow(
        name="규정 비교 분석",
        description="두 규정 문서를 병렬 검색하고 공통점·차이점·충돌 사항을 분석한 뒤 컴플라이언스 리뷰를 수행합니다.",
        nodes=[
            WorkflowNode(id="start", config=NodeConfig(
                node_type=NodeType.START, label="비교 요청 입력",
                config={"description": "비교할 두 규정 항목을 입력하세요 (예: 안전관리규정 vs 시설관리규정)"},
                position={"x": 40, "y": 220},
            )),
            WorkflowNode(id="retrieve_a", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="규정 A 검색",
                config={"top_k": 5},
                position={"x": 280, "y": 100},
            )),
            WorkflowNode(id="retrieve_b", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="규정 B 검색",
                config={"top_k": 5},
                position={"x": 280, "y": 340},
            )),
            WorkflowNode(id="compare_llm", config=NodeConfig(
                node_type=NodeType.LLM, label="비교 분석",
                config={
                    "system_prompt": (
                        f"당신은 {settings.platform_name} 규정 분석 전문가입니다.\n"
                        "두 규정을 다음 형식으로 비교 분석하세요:\n\n"
                        "## 규정 개요\n| 항목 | 규정 A | 규정 B |\n\n"
                        "## 공통점\n- 두 규정이 동일하게 규정하는 사항\n\n"
                        "## 차이점\n| 비교 항목 | 규정 A | 규정 B | 비고 |\n\n"
                        "## 충돌 사항\n- 두 규정 간 상충되는 내용 (있을 경우)\n\n"
                        "## 적용 시 유의사항\n- 실무에서 두 규정을 동시 적용 시 주의점"
                    ),
                    "temperature": 0.1,
                },
                position={"x": 540, "y": 220},
            )),
            WorkflowNode(id="output", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="비교 분석 보고서",
                config={"format": "markdown"},
                position={"x": 780, "y": 220},
            )),
        ],
        edges=[
            Edge(source="start", target="retrieve_a", label="규정 A"),
            Edge(source="start", target="retrieve_b", label="규정 B"),
            Edge(source="retrieve_a", target="compare_llm"),
            Edge(source="retrieve_b", target="compare_llm"),
            Edge(source="compare_llm", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# 3. 설비 점검이력 조회 및 리포팅
# ---------------------------------------------------------------------------

def create_equipment_inspection_workflow() -> Workflow:
    """설비 점검이력: EHSQ 이력 + 안전 규정 병렬 조회 → 분석 → 보고서 → 출력."""
    return Workflow(
        name="설비 점검이력 조회 및 리포팅",
        description="EHSQ 시스템에서 설비 점검이력을 조회하고, 관련 안전 규정과 대조 분석하여 점검 리포트를 자동 생성합니다.",
        nodes=[
            WorkflowNode(id="start", config=NodeConfig(
                node_type=NodeType.START, label="설비 정보 입력",
                config={"description": "조회할 설비명 또는 시설을 입력하세요 (예: 인천 LNG 기지 배관)"},
                position={"x": 40, "y": 220},
            )),
            WorkflowNode(id="ehsq_lookup", config=NodeConfig(
                node_type=NodeType.TOOL, label="EHSQ 점검이력 조회",
                config={
                    "tool_name": "ehsq_lookup",
                    "arguments_template": {
                        "action": "incident_report",
                        "facility": "{input}",
                    },
                },
                position={"x": 270, "y": 100},
            )),
            WorkflowNode(id="retrieve_safety", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="안전 규정 검색",
                config={"top_k": 5},
                position={"x": 270, "y": 340},
            )),
            WorkflowNode(id="analyze_llm", config=NodeConfig(
                node_type=NodeType.LLM, label="점검이력 분석",
                config={
                    "system_prompt": (
                        f"당신은 {settings.platform_name}의 설비 안전 분석 전문가입니다.\n"
                        "EHSQ 점검이력과 관련 안전 규정을 종합하여 분석하세요:\n\n"
                        "## 점검 현황 요약\n- 총 점검 횟수, 최근 점검일, 점검 주기 준수 여부\n\n"
                        "## 주요 결함 유형\n- 발견된 결함을 유형별로 분류하고 빈도 표시\n\n"
                        "## 재발 패턴 분석\n- 동일 결함의 반복 발생 여부 및 원인 추정\n\n"
                        "## 규정 대조 결과\n- 관련 안전 규정 준수 여부\n\n"
                        "## 권고 조치사항\n- 즉시 조치, 단기(1개월), 중기(분기) 구분"
                    ),
                    "temperature": 0.1,
                },
                position={"x": 520, "y": 220},
            )),
            WorkflowNode(id="report_tool", config=NodeConfig(
                node_type=NodeType.TOOL, label="점검보고서 생성",
                config={
                    "tool_name": "report_draft",
                    "arguments_template": {
                        "topic": "{input}",
                        "report_type": "점검보고서",
                    },
                },
                position={"x": 770, "y": 220},
            )),
            WorkflowNode(id="output", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="점검 리포트",
                config={"format": "markdown"},
                position={"x": 1010, "y": 220},
            )),
        ],
        edges=[
            Edge(source="start", target="ehsq_lookup", label="EHSQ"),
            Edge(source="start", target="retrieve_safety", label="규정"),
            Edge(source="ehsq_lookup", target="analyze_llm"),
            Edge(source="retrieve_safety", target="analyze_llm"),
            Edge(source="analyze_llm", target="report_tool"),
            Edge(source="report_tool", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# 4. 경영평가 자료 자동 수집
# ---------------------------------------------------------------------------

def create_management_evaluation_workflow() -> Workflow:
    """경영평가 자료: ERP 데이터 + 관련 규정 병렬 조회 → 종합 분석 → 현황보고서 → 출력."""
    return Workflow(
        name="경영평가 자료 자동 수집",
        description="ERP 시스템의 예산·실적 데이터와 내부규정을 결합하여 경영평가 자료를 자동으로 수집·분석합니다.",
        nodes=[
            WorkflowNode(id="start", config=NodeConfig(
                node_type=NodeType.START, label="평가 항목 입력",
                config={"description": "수집할 경영평가 항목을 입력하세요 (예: 2026년 상반기 안전관리 실적)"},
                position={"x": 40, "y": 220},
            )),
            WorkflowNode(id="erp_budget", config=NodeConfig(
                node_type=NodeType.TOOL, label="ERP 예산·실적 조회",
                config={
                    "tool_name": "erp_lookup",
                    "arguments_template": {
                        "query_type": "budget",
                        "keyword": "{input}",
                    },
                },
                position={"x": 280, "y": 100},
            )),
            WorkflowNode(id="retrieve_regulation", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="평가 기준 규정 검색",
                config={"top_k": 5},
                position={"x": 280, "y": 340},
            )),
            WorkflowNode(id="analyze_llm", config=NodeConfig(
                node_type=NodeType.LLM, label="종합 분석",
                config={
                    "system_prompt": (
                        f"당신은 {settings.platform_name}의 경영평가 분석 전문가입니다.\n"
                        "ERP에서 수집한 경영 데이터와 내부규정을 종합하여 분석하세요:\n\n"
                        "## 평가 항목 개요\n- 해당 경영평가 항목의 배경과 목적\n\n"
                        "## 정량 실적\n| 지표 | 목표 | 실적 | 달성률 |\n"
                        "- ERP 데이터 기반 수치 근거 명시\n\n"
                        "## 정성 실적\n- 주요 추진 사항 및 성과\n\n"
                        "## 규정 근거\n- 각 실적의 내부규정 근거 조항\n\n"
                        "## 개선 과제\n- 미달 항목에 대한 개선 방안"
                    ),
                    "temperature": 0.1,
                },
                position={"x": 540, "y": 220},
            )),
            WorkflowNode(id="report_tool", config=NodeConfig(
                node_type=NodeType.TOOL, label="현황보고서 생성",
                config={
                    "tool_name": "report_draft",
                    "arguments_template": {
                        "topic": "{input}",
                        "report_type": "현황보고서",
                    },
                },
                position={"x": 780, "y": 220},
            )),
            WorkflowNode(id="output", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="경영평가 자료",
                config={"format": "markdown"},
                position={"x": 1020, "y": 220},
            )),
        ],
        edges=[
            Edge(source="start", target="erp_budget", label="ERP"),
            Edge(source="start", target="retrieve_regulation", label="규정"),
            Edge(source="erp_budget", target="analyze_llm"),
            Edge(source="retrieve_regulation", target="analyze_llm"),
            Edge(source="analyze_llm", target="report_tool"),
            Edge(source="report_tool", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# 5. 신규직원 온보딩 가이드 생성
# ---------------------------------------------------------------------------

def create_onboarding_guide_workflow() -> Workflow:
    """온보딩 가이드: 내부규정 검색 → 핵심 규정 정리 → 교육자료 생성 → 출력."""
    return Workflow(
        name="신규직원 온보딩 가이드 생성",
        description="내부규정을 검색하여 핵심 내용을 정리하고, 신규직원 맞춤형 온보딩 가이드를 자동 생성합니다.",
        nodes=[
            WorkflowNode(id="start", config=NodeConfig(
                node_type=NodeType.START, label="부서·직무 입력",
                config={"description": "신규직원의 부서 및 직무를 입력하세요 (예: 시설안전부 배관검사)"},
                position={"x": 40, "y": 220},
            )),
            WorkflowNode(id="retrieve", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="내부규정 검색",
                config={"top_k": 7},
                position={"x": 260, "y": 220},
            )),
            WorkflowNode(id="organize_llm", config=NodeConfig(
                node_type=NodeType.LLM, label="핵심 규정 정리",
                config={
                    "system_prompt": (
                        f"당신은 {settings.platform_name}의 인사교육 담당자입니다.\n"
                        "검색된 내부규정 중 신규직원이 반드시 알아야 할 핵심 내용을 정리하세요:\n\n"
                        "1. 조직 및 직무 관련 규정\n"
                        "2. 안전 관련 필수 규정\n"
                        "3. 복무 및 근태 규정\n"
                        "4. 보안 및 정보보호 규정\n\n"
                        "각 항목에 해당 규정의 조항 번호를 명시하세요."
                    ),
                    "temperature": 0.2,
                },
                position={"x": 480, "y": 220},
            )),
            WorkflowNode(id="training_tool", config=NodeConfig(
                node_type=NodeType.TOOL, label="온보딩 교육자료 생성",
                config={
                    "tool_name": "training_material",
                    "arguments_template": {
                        "topic": "{input}",
                        "level": "신입",
                        "format": "종합",
                    },
                },
                position={"x": 720, "y": 220},
            )),
            WorkflowNode(id="output", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="온보딩 가이드",
                config={"format": "markdown"},
                position={"x": 960, "y": 220},
            )),
        ],
        edges=[
            Edge(source="start", target="retrieve"),
            Edge(source="retrieve", target="organize_llm"),
            Edge(source="organize_llm", target="training_tool"),
            Edge(source="training_tool", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# 6. ISO 인증 대응 문서 초안
# ---------------------------------------------------------------------------

def create_iso_compliance_workflow() -> Workflow:
    """ISO 대응 문서: 준수현황 + ISO규정 병렬조회 → Gap분석 → 대응문서 생성 → 출력."""
    return Workflow(
        name="ISO 인증 대응 문서 초안",
        description="EHSQ 준수 현황과 ISO 관련 내부규정을 결합하여 Gap 분석 및 대응 문서 초안을 자동 생성합니다.",
        nodes=[
            WorkflowNode(id="start", config=NodeConfig(
                node_type=NodeType.START, label="ISO 항목 입력",
                config={"description": "대응이 필요한 ISO 인증 항목을 입력하세요 (예: ISO 14001 환경경영 4.4 운영 계획)"},
                position={"x": 40, "y": 220},
            )),
            WorkflowNode(id="ehsq_compliance", config=NodeConfig(
                node_type=NodeType.TOOL, label="준수 현황 조회",
                config={
                    "tool_name": "ehsq_lookup",
                    "arguments_template": {
                        "action": "compliance_check",
                        "facility": "{input}",
                    },
                },
                position={"x": 280, "y": 100},
            )),
            WorkflowNode(id="retrieve_iso", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="ISO 관련 규정 검색",
                config={"top_k": 5},
                position={"x": 280, "y": 340},
            )),
            WorkflowNode(id="gap_analysis", config=NodeConfig(
                node_type=NodeType.LLM, label="Gap 분석",
                config={
                    "system_prompt": (
                        f"당신은 {settings.platform_name}의 ISO 인증 심사 대응 전문가입니다.\n"
                        "EHSQ 준수 현황과 내부규정을 대조하여 Gap 분석을 수행하세요:\n\n"
                        "## ISO 요건 요약\n- 해당 ISO 조항의 핵심 요구사항\n\n"
                        "## 현재 준수 현황\n| 요구사항 | 현황 | 적합/부적합 |\n\n"
                        "## Gap 분석\n- 미충족 항목과 원인 분석\n\n"
                        "## 개선 계획\n| 개선 항목 | 담당부서 | 목표일 | 조치 내용 |\n\n"
                        "## 필요 증적 자료 목록\n- 심사 시 제출해야 할 증빙 문서"
                    ),
                    "temperature": 0.1,
                },
                position={"x": 540, "y": 220},
            )),
            WorkflowNode(id="report_tool", config=NodeConfig(
                node_type=NodeType.TOOL, label="공식 대응 문서 생성",
                config={
                    "tool_name": "report_draft",
                    "arguments_template": {
                        "topic": "{input}",
                        "report_type": "분석보고서",
                    },
                },
                position={"x": 800, "y": 220},
            )),
            WorkflowNode(id="output", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="ISO 대응 문서",
                config={"format": "markdown"},
                position={"x": 1040, "y": 220},
            )),
        ],
        edges=[
            Edge(source="start", target="ehsq_compliance", label="EHSQ"),
            Edge(source="start", target="retrieve_iso", label="규정"),
            Edge(source="ehsq_compliance", target="gap_analysis"),
            Edge(source="retrieve_iso", target="gap_analysis"),
            Edge(source="gap_analysis", target="report_tool"),
            Edge(source="report_tool", target="output"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# 7. 보고서 자동 생성
# ---------------------------------------------------------------------------

def create_report_generation_workflow() -> Workflow:
    """보고서 자동 생성: 주제 입력 → RAG 검색 → 보고서 초안 → 출력."""
    return Workflow(
        name="보고서 자동 생성",
        description="주제를 입력하면 관련 문서를 검색하고, 전문적인 보고서 초안을 자동 생성합니다. PDF/DOCX로 내보내기 가능합니다.",
        nodes=[
            WorkflowNode(id="start", config=NodeConfig(
                node_type=NodeType.START, label="보고서 주제 입력",
                config={"description": "작성할 보고서 주제를 입력하세요 (예: 2026년 1분기 안전점검 결과 보고서)"},
                position={"x": 40, "y": 220},
            )),
            WorkflowNode(id="retrieve", config=NodeConfig(
                node_type=NodeType.RETRIEVAL, label="관련 문서 검색",
                config={"top_k": 7},
                position={"x": 260, "y": 220},
            )),
            WorkflowNode(id="confidence_check", config=NodeConfig(
                node_type=NodeType.CONDITION, label="검색 품질 확인",
                config={"condition_type": "confidence", "threshold": 0.3},
                position={"x": 480, "y": 220},
            )),
            WorkflowNode(id="draft_llm", config=NodeConfig(
                node_type=NodeType.LLM, label="보고서 초안 작성",
                config={
                    "system_prompt": (
                        f"당신은 {settings.platform_name}의 전문 보고서 작성 어시스턴트입니다.\n"
                        "검색된 문서를 바탕으로 한국 공공기관 형식의 보고서 초안을 작성하세요.\n\n"
                        "보고서 구조:\n"
                        "## 1. 개요\n- 보고서 목적 및 배경\n\n"
                        "## 2. 주요 내용\n- 핵심 사항을 번호 매겨 정리\n\n"
                        "## 3. 세부 분석\n- 관련 규정 근거와 함께 상세 분석\n\n"
                        "## 4. 결론 및 제언\n- 요약 및 향후 조치사항\n\n"
                        "## 5. 참고 자료\n- 인용 문서 목록\n\n"
                        "사실에 기반하여 작성하고 불확실한 내용은 '[확인 필요]'로 표시하세요."
                    ),
                    "temperature": 0.2,
                },
                position={"x": 700, "y": 120},
            )),
            WorkflowNode(id="fallback", config=NodeConfig(
                node_type=NodeType.TRANSFORM, label="검색 결과 부족 안내",
                config={"template": "관련 문서의 검색 결과가 부족합니다.\n\n{input}\n\n더 구체적인 주제로 다시 시도하거나, 관련 문서가 등록되어 있는지 확인해 주세요."},
                position={"x": 700, "y": 340},
            )),
            WorkflowNode(id="output", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="보고서 출력",
                config={"format": "markdown"},
                position={"x": 940, "y": 120},
            )),
            WorkflowNode(id="output_low", config=NodeConfig(
                node_type=NodeType.OUTPUT, label="안내 출력",
                config={"format": "markdown"},
                position={"x": 940, "y": 340},
            )),
        ],
        edges=[
            Edge(source="start", target="retrieve"),
            Edge(source="retrieve", target="confidence_check"),
            Edge(source="confidence_check", target="draft_llm", condition="true", label="신뢰도 충분"),
            Edge(source="confidence_check", target="fallback", condition="false", label="신뢰도 부족"),
            Edge(source="draft_llm", target="output"),
            Edge(source="fallback", target="output_low"),
        ],
        status=WorkflowStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRESET_WORKFLOWS = {
    "tech_doc_summary": create_tech_doc_summary_workflow,
    "regulation_comparison": create_regulation_comparison_workflow,
    "equipment_inspection": create_equipment_inspection_workflow,
    "management_evaluation": create_management_evaluation_workflow,
    "onboarding_guide": create_onboarding_guide_workflow,
    "iso_compliance": create_iso_compliance_workflow,
    "report_generation": create_report_generation_workflow,
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
