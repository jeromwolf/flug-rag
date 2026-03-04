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
