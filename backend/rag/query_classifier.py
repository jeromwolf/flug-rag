"""Fast rule-based query pre-classifier.

Classifies user queries BEFORE the LLM-based QueryRouter to:
- Immediately respond to identity/dangerous queries (no LLM needed)
- Route chitchat/general queries to direct LLM (skip RAG)
- Pass rag queries to the full RAG pipeline

Performance: <1ms (pure regex/keyword matching, no LLM call)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from config.settings import settings


class QueryClass(str, Enum):
    RAG = "rag"  # Internal regulation questions → full RAG pipeline
    GENERAL = "general"  # General knowledge → direct LLM
    IDENTITY = "identity"  # System/model info → immediate canned response
    DANGEROUS = "dangerous"  # Prompt injection/jailbreak → immediate block
    CHITCHAT = "chitchat"  # Greetings/thanks → direct LLM (lightweight)


@dataclass
class ClassificationResult:
    category: QueryClass
    confidence: float
    immediate_response: str | None = None  # For identity/dangerous only


# ── Canned Responses ──

def _identity_response() -> str:
    return (
        f"저는 {settings.platform_name}의 AI 어시스턴트입니다. "
        "내부규정 및 업무 관련 질문에 답변하도록 설계되었습니다. "
        "궁금한 점이 있으시면 편하게 질문해주세요."
    )

DANGEROUS_RESPONSE = (
    "죄송합니다. 해당 요청은 처리할 수 없습니다. "
    "업무 관련 질문을 부탁드립니다."
)

# ── Pattern Definitions ──

# 1. Dangerous: prompt injection / jailbreak attempts
_DANGEROUS_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"역할을?\s*무시", re.IGNORECASE),
    re.compile(r"지시를?\s*무시", re.IGNORECASE),
    re.compile(r"명령을?\s*무시", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?instructions", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are", re.IGNORECASE),
    re.compile(r"act\s+as\s+if", re.IGNORECASE),
    re.compile(r"새로운\s*역할", re.IGNORECASE),
    re.compile(r"프롬프트\s*(를\s*)?알려", re.IGNORECASE),
    re.compile(r"시스템\s*메시지", re.IGNORECASE),
]

# 2. Identity: system/model info questions
_IDENTITY_KEYWORDS = [
    "이름이 뭐",
    "이름이 뭘",
    "이름이 뭔",
    "넌 누구",
    "너는 누구",
    "당신은 누구",
    "어떤 모델",
    "무슨 모델",
    "모델 이름",
    "만든 사람",
    "개발자가 누구",
    "누가 만들",
    "어떤 AI",
    "무슨 AI",
    "뭐로 만들",
    "어디서 만들",
]
_IDENTITY_PATTERNS = [
    re.compile(r"(너|당신|자기).*(소개|설명)", re.IGNORECASE),
    re.compile(r"(GPT|ChatGPT|챗지피티|클로드|Claude|Gemini|제미나이)", re.IGNORECASE),
]

# 3. Chitchat: greetings and casual conversation
_CHITCHAT_KEYWORDS = [
    "안녕",
    "감사합니다",
    "감사해",
    "고마워",
    "고맙습니다",
    "반가워",
    "반갑습니다",
    "잘가",
    "수고",
    "ㅎㅎ",
    "ㅋㅋ",
    "ㅠㅠ",
    "ㅜㅜ",
    "좋은 하루",
    "좋은 아침",
    "화이팅",
    "파이팅",
    "하이",
    "헬로",
    "hello",
    "hi",
    "thanks",
    "thank you",
    "bye",
]

# 4. General: non-work questions
_GENERAL_KEYWORDS = [
    "날씨",
    "기온",
    "비가 오",
    "눈이 오",
    "오늘 몇일",
    "오늘 몇 일",
    "오늘 날짜",
    "지금 몇시",
    "지금 몇 시",
    "현재 시간",
    "환율",
    "주가",
    "주식",
    "코인",
    "비트코인",
    "로또",
    "맛집",
    "영화",
    "드라마",
    "축구",
    "야구",
    "농구",
    "올림픽",
    "월드컵",
]
_GENERAL_PATTERNS = [
    # Programming/tech (not work-related regulations)
    re.compile(r"(파이썬|자바|자바스크립트|리액트|python|java|javascript|react)\s*(이란|이 뭐|이 뭔|이란|이란\?|코드|문법)", re.IGNORECASE),
    # Math
    re.compile(r"(\d+)\s*[+\-×÷*/]\s*(\d+)", re.IGNORECASE),
    # Translation
    re.compile(r"(번역|translate)", re.IGNORECASE),
]

# 5. RAG: work-related domain keywords
_RAG_KEYWORDS = [
    "규정", "규칙", "규범", "매뉴얼", "절차", "기준", "지침", "조례",
    "안전", "가스", "점검", "검사", "설비", "배관", "시설",
    "휴가", "연차", "출장", "여비", "수당", "급여", "보수", "퇴직",
    "징계", "보상", "재해", "산재", "보험",
    "교육", "훈련", "자격", "면허", "인사", "승진", "전보",
    "계약", "입찰", "조달", "예산", "결산", "회계",
    "감사원", "감사규정", "내부감사", "내부통제", "윤리", "갑질", "직무발명",
    "문서", "보고서", "서식", "양식",
    "조항", "조문", "별표", "별지",
    "KOGAS",
]
# Add platform name and its abbreviation dynamically
if settings.platform_name not in _RAG_KEYWORDS:
    _RAG_KEYWORDS.append(settings.platform_name)
_short_name = settings.platform_name.replace("한국", "")
if _short_name and _short_name != settings.platform_name and _short_name not in _RAG_KEYWORDS:
    _RAG_KEYWORDS.append(_short_name)
_RAG_PATTERNS = [
    re.compile(r"제\s*\d+\s*조", re.IGNORECASE),
    re.compile(r"(어떻게|어떤|무엇|뭐가|뭘).*(규정|절차|기준|방법|처리)", re.IGNORECASE),
]


class QueryClassifier:
    """Fast rule-based query pre-classifier.

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("너 이름이 뭐야?")
        if result.immediate_response:
            return result.immediate_response  # No LLM needed
    """

    def classify(self, query: str) -> ClassificationResult:
        """Classify a user query into one of 5 categories.

        Order of checks matters — earlier checks take priority.
        """
        q = query.strip()
        q_lower = q.lower()

        # Very short queries are likely chitchat
        if len(q) <= 2:
            return ClassificationResult(
                category=QueryClass.CHITCHAT,
                confidence=0.9,
            )

        # 1. Dangerous check (highest priority)
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(q):
                return ClassificationResult(
                    category=QueryClass.DANGEROUS,
                    confidence=0.95,
                    immediate_response=DANGEROUS_RESPONSE,
                )

        # 2. Identity check
        for kw in _IDENTITY_KEYWORDS:
            if kw in q_lower:
                return ClassificationResult(
                    category=QueryClass.IDENTITY,
                    confidence=0.95,
                    immediate_response=_identity_response(),
                )
        for pattern in _IDENTITY_PATTERNS:
            if pattern.search(q):
                return ClassificationResult(
                    category=QueryClass.IDENTITY,
                    confidence=0.90,
                    immediate_response=_identity_response(),
                )

        # 3. Check RAG keywords BEFORE chitchat/general
        # (prevents "안전 규정 감사합니다" from being classified as chitchat)
        has_rag_keyword = any(kw in q_lower for kw in _RAG_KEYWORDS)
        has_rag_pattern = any(p.search(q) for p in _RAG_PATTERNS)

        if has_rag_keyword or has_rag_pattern:
            return ClassificationResult(
                category=QueryClass.RAG,
                confidence=0.85 if has_rag_keyword else 0.75,
            )

        # 4. Chitchat check (only if no RAG keywords)
        for kw in _CHITCHAT_KEYWORDS:
            if kw in q_lower:
                return ClassificationResult(
                    category=QueryClass.CHITCHAT,
                    confidence=0.90,
                )

        # 5. General check
        for kw in _GENERAL_KEYWORDS:
            if kw in q_lower:
                return ClassificationResult(
                    category=QueryClass.GENERAL,
                    confidence=0.85,
                )
        for pattern in _GENERAL_PATTERNS:
            if pattern.search(q):
                return ClassificationResult(
                    category=QueryClass.GENERAL,
                    confidence=0.80,
                )

        # 6. Default: RAG (false negative is worse than false positive)
        return ClassificationResult(
            category=QueryClass.RAG,
            confidence=0.50,
        )
