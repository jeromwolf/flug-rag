"""Tool selector: keyword-based mapping of user messages to MCP tools.

Runs BEFORE the LLM router to deterministically detect tool-invocation
intent from the user message.  Only fires when both a *domain keyword*
and an *action keyword* are present (e.g. "보고서" + "만들어줘").
"""

import re
from dataclasses import dataclass


@dataclass
class ToolSelection:
    """Result of tool selection."""

    tool_name: str
    arguments: dict[str, str]
    confidence: float


# ── Report draft ────────────────────────────────────────────────────
_REPORT_TYPE_MAP = {
    "결과": "결과보고서",
    "현황": "현황보고서",
    "분석": "분석보고서",
    "점검": "점검보고서",
}

_REPORT_TRIGGERS = [
    "보고서 만들",
    "보고서 작성",
    "보고서 초안",
    "보고서 생성",
    "보고서를 만들",
    "보고서를 작성",
    "리포트 만들",
    "초안 만들",
    "초안 작성",
    "초안 생성",
]

# ── Training material ──────────────────────────────────────────────
_LEVEL_MAP = {
    "신입": "신입",
    "초급": "신입",
    "기초": "신입",
    "중급": "중급",
    "고급": "고급",
    "심화": "고급",
}

_FORMAT_MAP = {
    "교안": "교안",
    "체크리스트": "체크리스트",
    "퀴즈": "퀴즈",
    "종합": "종합",
}

_TRAINING_TRIGGERS = [
    "교육자료 만들",
    "교육자료 작성",
    "교육자료 생성",
    "교육자료를 만들",
    "교육자료를 작성",
    "교육 자료 만들",
    "교안 만들",
    "교안 작성",
    "교안을 만들",
    "교안을 작성",
    "학습 자료 만들",
    "교육 콘텐츠 만들",
]

_PERIOD_RE = re.compile(r"(\d{4}년[\s]*(?:\d{1,2}월)?[\s]*(?:\d분기)?)")

# ── Regulation review ───────────────────────────────────────────────
_REGULATION_TRIGGERS = [
    "규정 검토", "규정검토", "규정 위반", "규정과 대조",
    "준수 여부", "규정 확인", "법규 검토", "내부규정 검토",
]

# ── Safety checklist ────────────────────────────────────────────────
_SAFETY_TRIGGERS = [
    "안전 체크리스트", "안전체크리스트", "점검 체크리스트",
    "점검표 만들", "점검표 생성", "안전 점검표", "안전점검표",
    "체크리스트 만들", "체크리스트 생성",
]

_EQUIPMENT_MAP = {
    "배관": "배관", "정압기": "정압기", "저장탱크": "저장탱크", "공급설비": "공급설비",
}

# ── Calculator ──────────────────────────────────────────────────────
_CALC_TRIGGERS = [
    "계산해", "계산해줘", "계산 해줘", "얼마야",
    "수식 계산", "값을 구해",
]

# ── Data analyzer ───────────────────────────────────────────────────
_DATA_TRIGGERS = [
    "데이터 분석", "데이터분석", "통계 분석", "통계분석",
    "평균 구해", "분포 분석",
]


# ── Public API ─────────────────────────────────────────────────────
def select_tool(message: str) -> ToolSelection | None:
    """Select MCP tool based on message keywords.

    Returns *ToolSelection* if a generation tool matches, ``None`` otherwise.
    """
    msg = message.strip()

    if any(kw in msg for kw in _REPORT_TRIGGERS):
        return _select_report(msg)

    if any(kw in msg for kw in _TRAINING_TRIGGERS):
        return _select_training(msg)

    if any(kw in msg for kw in _REGULATION_TRIGGERS):
        return _select_regulation(msg)

    if any(kw in msg for kw in _SAFETY_TRIGGERS):
        return _select_safety(msg)

    if any(kw in msg for kw in _CALC_TRIGGERS):
        return _select_calculator(msg)

    if any(kw in msg for kw in _DATA_TRIGGERS):
        return _select_data_analyzer(msg)

    return None


# ── Internal helpers ───────────────────────────────────────────────
def _select_report(msg: str) -> ToolSelection:
    report_type = "결과보고서"
    for kw, rtype in _REPORT_TYPE_MAP.items():
        if kw in msg:
            report_type = rtype
            break

    period = ""
    m = _PERIOD_RE.search(msg)
    if m:
        period = m.group(1).strip()

    return ToolSelection(
        tool_name="report_draft",
        arguments={"topic": msg, "report_type": report_type, "period": period},
        confidence=0.8,
    )


def _select_training(msg: str) -> ToolSelection:
    level = "신입"
    for kw, lvl in _LEVEL_MAP.items():
        if kw in msg:
            level = lvl
            break

    fmt = "종합"
    for kw, f in _FORMAT_MAP.items():
        if kw in msg:
            fmt = f
            break

    return ToolSelection(
        tool_name="training_material",
        arguments={"topic": msg, "level": level, "format": fmt},
        confidence=0.8,
    )


def _select_regulation(msg: str) -> ToolSelection:
    category = "전체"
    cat_map = {"안전": "안전관리", "시설": "시설기준", "운영": "운영규정"}
    for kw, cat in cat_map.items():
        if kw in msg:
            category = cat
            break
    return ToolSelection(
        tool_name="regulation_review",
        arguments={"document_text": msg, "regulation_category": category, "review_depth": "standard"},
        confidence=0.8,
    )


def _select_safety(msg: str) -> ToolSelection:
    equipment = "일반"
    for kw, eq in _EQUIPMENT_MAP.items():
        if kw in msg:
            equipment = eq
            break
    return ToolSelection(
        tool_name="safety_checklist",
        arguments={"equipment_type": equipment, "output_format": "markdown"},
        confidence=0.8,
    )


def _select_calculator(msg: str) -> ToolSelection:
    expr_re = re.compile(r'[\d\.\+\-\*\/\(\)\s\^%]+')
    m = expr_re.search(msg)
    expression = m.group(0).strip() if m else msg
    return ToolSelection(
        tool_name="calculator",
        arguments={"expression": expression},
        confidence=0.75,
    )


def _select_data_analyzer(msg: str) -> ToolSelection:
    return ToolSelection(
        tool_name="data_analyzer",
        arguments={"data": [], "analysis_type": "statistics"},
        confidence=0.7,
    )
