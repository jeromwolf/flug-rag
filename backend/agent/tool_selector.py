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

# ── Summarizer ──────────────────────────────────────────────────────
_SUMMARIZE_TRIGGERS = [
    "요약해", "요약해줘", "요약 해줘", "요약해주세요",
    "요약해 줘", "요약해 주세요", "요약하면",
    "summarize", "요약본", "핵심 정리", "핵심정리",
    "간략히 정리", "간단히 정리", "줄여서 말해", "짧게 정리",
]

_SUMMARIZE_MODE_MAP = {
    "핵심 문장": "extractive",
    "핵심문장": "extractive",
    "문장 추출": "extractive",
    "추출": "extractive",
}

# ── Translator ──────────────────────────────────────────────────────
_TRANSLATE_TRIGGERS = [
    "번역해", "번역해줘", "번역 해줘", "번역해주세요",
    "번역해 줘", "번역해 주세요", "translate",
    "영어로", "영어로 번역", "일본어로", "일본어로 번역",
    "중국어로", "중국어로 번역", "한국어로 번역",
    "영문으로", "영문 번역", "번역본 만들",
]

_TRANSLATE_SOURCE_MAP = {
    "영어": "en", "english": "en", "영문": "en",
    "일본어": "ja", "일어": "ja", "japanese": "ja",
    "중국어": "zh", "chinese": "zh", "중문": "zh",
    "한국어": "ko", "korean": "ko", "한글": "ko",
}

_TRANSLATE_TARGET_MAP = {
    "영어로": "en", "영문으로": "en", "영어 번역": "en",
    "일본어로": "ja", "일어로": "ja",
    "중국어로": "zh", "중문으로": "zh",
    "한국어로": "ko", "한글로": "ko",
}

# ── Email composer ──────────────────────────────────────────────────
_EMAIL_TRIGGERS = [
    "이메일 작성", "이메일 써줘", "이메일 써주세요",
    "이메일을 작성", "이메일을 써줘", "이메일을 써주세요",
    "메일 작성", "메일 써줘", "메일 써주세요",
    "메일을 작성", "메일을 써줘", "메일을 써주세요",
    "이메일 초안", "메일 초안", "메일 만들어줘",
    "이메일 만들어줘", "이메일 만들어주세요", "메일 만들어주세요",
]

_EMAIL_TONE_MAP = {
    "격식": "formal", "공식": "formal", "공문": "formal",
    "반격식": "semi-formal", "일반": "semi-formal",
    "캐주얼": "casual", "친근": "casual", "편하게": "casual",
}

# ── Report generator (template-based) ──────────────────────────────
_REPORT_GEN_TRIGGERS = [
    "안전 점검 보고서", "안전점검보고서", "안전점검 보고서",
    "월간 보고서", "월간보고서", "월간 요약",
    "사고 보고서", "사고보고서", "이상 보고서", "이상보고서",
]

_REPORT_GEN_TEMPLATE_MAP = {
    "안전 점검": "safety_inspection",
    "안전점검": "safety_inspection",
    "월간": "monthly_summary",
    "사고": "incident_report",
    "이상": "incident_report",
}

# ── Knowledge base / Search ─────────────────────────────────────────
_SEARCH_TRIGGERS = [
    "문서 검색", "문서검색", "지식베이스 검색", "지식 베이스 검색",
    "관련 문서 찾아", "관련문서 찾아", "자료 검색", "자료검색",
    "검색해줘", "검색해 줘", "검색해주세요",
]

_KB_TRIGGERS = [
    "문서 몇 개", "총 문서 수", "지식베이스 현황", "지식 베이스 현황",
    "청크 수", "등록된 문서", "인덱스 현황",
]

# ── ERP lookup ────────────────────────────────────────────────────
_ERP_TRIGGERS = [
    "ERP 조회", "ERP조회", "예산 조회", "예산조회", "예산 현황", "예산현황",
    "프로젝트 현황", "프로젝트현황", "프로젝트 조회", "협력업체 조회",
    "협력업체 현황", "벤더 조회", "집행률", "예산 집행",
]

_ERP_TYPE_MAP = {
    "예산": "budget", "집행": "budget",
    "프로젝트": "project", "공사": "project",
    "협력업체": "vendor", "벤더": "vendor", "업체": "vendor",
}

# ── EHSQ lookup ───────────────────────────────────────────────────
_EHSQ_TRIGGERS = [
    "EHSQ 조회", "EHSQ조회", "안전 사고", "안전사고", "사고 현황", "사고현황",
    "안전 등급", "안전등급", "시설 안전", "컴플라이언스 현황", "준수 현황",
    "안전 현황", "안전현황", "재해 현황", "무재해",
]

_EHSQ_ACTION_MAP = {
    "사고": "incident_report", "재해": "incident_report", "무재해": "incident_report",
    "등급": "safety_status", "시설": "safety_status",
    "컴플라이언스": "compliance_check", "준수": "compliance_check",
}

# ── Groupware lookup ──────────────────────────────────────────────
_GROUPWARE_TRIGGERS = [
    "그룹웨어 조회", "그룹웨어조회", "일정 조회", "일정조회",
    "결재 현황", "결재현황", "결재 조회", "공지사항 조회", "공지 조회",
    "회의 일정", "회의일정", "오늘 일정", "금주 일정",
]

_GROUPWARE_TYPE_MAP = {
    "일정": "schedule", "회의": "schedule",
    "결재": "approval",
    "공지": "notice",
}

# ── System DB query ───────────────────────────────────────────────
_SYSTEM_DB_TRIGGERS = [
    "시스템 현황", "시스템현황", "사용 현황", "사용현황",
    "DB 조회", "DB조회", "데이터베이스 조회",
    "질의 통계", "질의통계", "사용자 현황", "사용자현황",
    "감사 로그 조회", "접속 현황", "접속현황",
    "운영 현황", "운영현황", "시스템 통계", "시스템통계",
]

_SYSTEM_DB_TYPE_MAP = {
    "사용자": "user_stats",
    "질의": "query_stats", "통계": "query_stats",
    "문서": "document_stats",
    "감사": "audit_log", "로그": "audit_log", "접속": "audit_log",
}


# ── Public API ─────────────────────────────────────────────────────
def select_tool(message: str) -> ToolSelection | None:
    """Select MCP tool based on message keywords.

    Returns *ToolSelection* if a generation tool matches, ``None`` otherwise.
    Evaluation order: more-specific patterns first to avoid false matches.
    """
    msg = message.strip()

    # Enterprise system integrations (highest priority for demo)
    if any(kw in msg for kw in _SYSTEM_DB_TRIGGERS):
        return _select_system_db(msg)

    if any(kw in msg for kw in _ERP_TRIGGERS):
        return _select_erp(msg)

    if any(kw in msg for kw in _EHSQ_TRIGGERS):
        return _select_ehsq(msg)

    if any(kw in msg for kw in _GROUPWARE_TRIGGERS):
        return _select_groupware(msg)

    # Higher-specificity triggers first
    if any(kw in msg for kw in _REPORT_GEN_TRIGGERS):
        return _select_report_generator(msg)

    if any(kw in msg for kw in _REPORT_TRIGGERS):
        return _select_report(msg)

    if any(kw in msg for kw in _TRAINING_TRIGGERS):
        return _select_training(msg)

    if any(kw in msg for kw in _REGULATION_TRIGGERS):
        return _select_regulation(msg)

    if any(kw in msg for kw in _SAFETY_TRIGGERS):
        return _select_safety(msg)

    if any(kw in msg for kw in _EMAIL_TRIGGERS):
        return _select_email(msg)

    if any(kw in msg for kw in _TRANSLATE_TRIGGERS):
        return _select_translator(msg)

    if any(kw in msg for kw in _SUMMARIZE_TRIGGERS):
        return _select_summarizer(msg)

    if any(kw in msg for kw in _KB_TRIGGERS):
        return _select_knowledge_base(msg)

    if any(kw in msg for kw in _SEARCH_TRIGGERS):
        return _select_search(msg)

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


def _select_summarizer(msg: str) -> ToolSelection:
    """Select document_summarizer tool with mode detection."""
    mode = "abstractive"
    for kw, m in _SUMMARIZE_MODE_MAP.items():
        if kw in msg:
            mode = m
            break
    return ToolSelection(
        tool_name="document_summarizer",
        arguments={"text": msg, "mode": mode, "language": "ko"},
        confidence=0.75,
    )


def _select_translator(msg: str) -> ToolSelection:
    """Select translator tool with language-pair detection."""
    # Detect target language from target-direction keywords first
    target_lang = "en"  # default: translate to English
    for kw, lang in _TRANSLATE_TARGET_MAP.items():
        if kw in msg:
            target_lang = lang
            break

    # Infer source language: opposite of target, default ko
    source_lang = "ko" if target_lang != "ko" else "en"
    # Override source if explicitly mentioned
    for kw, lang in _TRANSLATE_SOURCE_MAP.items():
        if kw in msg and lang != target_lang:
            source_lang = lang
            break

    return ToolSelection(
        tool_name="translator",
        arguments={"text": msg, "source_lang": source_lang, "target_lang": target_lang},
        confidence=0.8,
    )


def _select_email(msg: str) -> ToolSelection:
    """Select email_composer tool with tone detection."""
    tone = "formal"  # default
    for kw, t in _EMAIL_TONE_MAP.items():
        if kw in msg:
            tone = t
            break

    # Try to extract a subject hint from the message
    subject_hint = msg if len(msg) <= 80 else msg[:80]

    return ToolSelection(
        tool_name="email_composer",
        arguments={
            "subject": subject_hint,
            "recipients": [],
            "body_context": msg,
            "tone": tone,
        },
        confidence=0.8,
    )


def _select_report_generator(msg: str) -> ToolSelection:
    """Select report_generator tool with template detection."""
    template = "monthly_summary"  # safe default
    for kw, tmpl in _REPORT_GEN_TEMPLATE_MAP.items():
        if kw in msg:
            template = tmpl
            break
    return ToolSelection(
        tool_name="report_generator",
        arguments={"template_name": template, "data": {"description": msg}, "output_format": "markdown"},
        confidence=0.75,
    )


def _select_search(msg: str) -> ToolSelection:
    """Select search_documents tool."""
    return ToolSelection(
        tool_name="search_documents",
        arguments={"query": msg, "top_k": 5},
        confidence=0.7,
    )


def _select_knowledge_base(msg: str) -> ToolSelection:
    """Select knowledge_base tool for status/count queries."""
    return ToolSelection(
        tool_name="knowledge_base",
        arguments={"action": "count"},
        confidence=0.7,
    )


def _select_erp(msg: str) -> ToolSelection:
    """Select ERP lookup tool with query type detection."""
    query_type = "budget"  # default
    for kw, qt in _ERP_TYPE_MAP.items():
        if kw in msg:
            query_type = qt
            break
    return ToolSelection(
        tool_name="erp_lookup",
        arguments={"query_type": query_type, "keyword": msg},
        confidence=0.85,
    )


def _select_ehsq(msg: str) -> ToolSelection:
    """Select EHSQ lookup tool with action detection."""
    action = "safety_status"  # default
    for kw, act in _EHSQ_ACTION_MAP.items():
        if kw in msg:
            action = act
            break
    return ToolSelection(
        tool_name="ehsq_lookup",
        arguments={"action": action, "facility": msg},
        confidence=0.85,
    )


def _select_groupware(msg: str) -> ToolSelection:
    """Select groupware lookup tool with type detection."""
    action = "schedule"  # default
    for kw, qt in _GROUPWARE_TYPE_MAP.items():
        if kw in msg:
            action = qt
            break
    return ToolSelection(
        tool_name="groupware_lookup",
        arguments={"action": action, "keyword": ""},
        confidence=0.85,
    )


def _select_system_db(msg: str) -> ToolSelection:
    """Select system DB query tool with type detection."""
    query_type = "system_summary"  # default: 전체 요약
    for kw, qt in _SYSTEM_DB_TYPE_MAP.items():
        if kw in msg:
            query_type = qt
            break
    return ToolSelection(
        tool_name="system_db_query",
        arguments={"query_type": query_type, "days": 7},
        confidence=0.9,
    )
