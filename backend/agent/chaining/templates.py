"""Pre-built chain templates for common multi-agent patterns."""

from typing import Any

from agent.chaining.chain import AgentChain, ChainStep


def create_research_chain(
    router_fn=None,
    retriever_fn=None,
    summarizer_fn=None,
    formatter_fn=None,
) -> AgentChain:
    """Research chain: Router -> Retriever -> Summarizer -> Formatter.

    Designed for research queries that need document retrieval
    and summarized answers.
    """
    chain = AgentChain(
        name="research_chain",
        description="연구/조사 체인: 라우터 -> 검색 -> 요약 -> 포맷팅",
    )

    if router_fn:
        chain.add(
            agent_id="router",
            name="질문 라우팅",
            execute_fn=router_fn,
        )
    if retriever_fn:
        chain.add(
            agent_id="retriever",
            name="문서 검색",
            execute_fn=retriever_fn,
        )
    if summarizer_fn:
        chain.add(
            agent_id="summarizer",
            name="결과 요약",
            execute_fn=summarizer_fn,
        )
    if formatter_fn:
        chain.add(
            agent_id="formatter",
            name="결과 포맷팅",
            execute_fn=formatter_fn,
        )

    return chain


def create_analysis_chain(
    analyzer_fn=None,
    reporter_fn=None,
    email_fn=None,
) -> AgentChain:
    """Analysis chain: DataAnalyzer -> Reporter -> EmailComposer.

    Designed for data analysis workflows that end with
    a report or email.
    """
    chain = AgentChain(
        name="analysis_chain",
        description="분석 체인: 데이터 분석 -> 보고서 생성 -> 이메일 작성",
    )

    if analyzer_fn:
        chain.add(
            agent_id="data_analyzer",
            name="데이터 분석",
            execute_fn=analyzer_fn,
        )
    if reporter_fn:
        chain.add(
            agent_id="report_generator",
            name="보고서 생성",
            execute_fn=reporter_fn,
        )
    if email_fn:
        chain.add(
            agent_id="email_composer",
            name="이메일 작성",
            execute_fn=email_fn,
            condition=lambda data: isinstance(data, dict) and data.get("send_email", False),
        )

    return chain


def create_qa_chain(
    router_fn=None,
    planner_fn=None,
    retriever_fn=None,
    merger_fn=None,
    quality_fn=None,
) -> AgentChain:
    """QA chain: Router -> Planner -> Retrieval -> Merger -> QualityCheck.

    Designed for complex Q&A that may need planning and
    multi-source retrieval.
    """
    chain = AgentChain(
        name="qa_chain",
        description="QA 체인: 라우터 -> 플래너 -> 검색 -> 병합 -> 품질 검증",
    )

    if router_fn:
        chain.add(
            agent_id="router",
            name="질문 분류",
            execute_fn=router_fn,
        )
    if planner_fn:
        chain.add(
            agent_id="planner",
            name="실행 계획 수립",
            execute_fn=planner_fn,
            condition=lambda data: isinstance(data, dict) and data.get("needs_planning", False),
        )
    if retriever_fn:
        chain.add(
            agent_id="retriever",
            name="문서 검색",
            execute_fn=retriever_fn,
        )
    if merger_fn:
        chain.add(
            agent_id="merger",
            name="결과 병합",
            execute_fn=merger_fn,
        )
    if quality_fn:
        chain.add(
            agent_id="quality_check",
            name="품질 검증",
            execute_fn=quality_fn,
        )

    return chain


def create_translation_chain(
    detector_fn=None,
    translator_fn=None,
    quality_fn=None,
) -> AgentChain:
    """Translation chain: Language Detector -> Translator -> QualityCheck.

    Designed for translation workflows with automatic
    language detection and quality verification.
    """
    chain = AgentChain(
        name="translation_chain",
        description="번역 체인: 언어 감지 -> 번역 -> 품질 검증",
    )

    if detector_fn:
        chain.add(
            agent_id="language_detector",
            name="언어 감지",
            execute_fn=detector_fn,
        )
    if translator_fn:
        chain.add(
            agent_id="translator",
            name="번역",
            execute_fn=translator_fn,
        )
    if quality_fn:
        chain.add(
            agent_id="quality_check",
            name="번역 품질 검증",
            execute_fn=quality_fn,
        )

    return chain


# Template metadata for listing
CHAIN_TEMPLATES = {
    "research_chain": {
        "name": "연구/조사 체인",
        "description": "라우터 -> 검색 -> 요약 -> 포맷팅",
        "steps": ["router", "retriever", "summarizer", "formatter"],
        "factory": create_research_chain,
    },
    "analysis_chain": {
        "name": "분석 체인",
        "description": "데이터 분석 -> 보고서 생성 -> 이메일 작성",
        "steps": ["data_analyzer", "report_generator", "email_composer"],
        "factory": create_analysis_chain,
    },
    "qa_chain": {
        "name": "QA 체인",
        "description": "라우터 -> 플래너 -> 검색 -> 병합 -> 품질 검증",
        "steps": ["router", "planner", "retriever", "merger", "quality_check"],
        "factory": create_qa_chain,
    },
    "translation_chain": {
        "name": "번역 체인",
        "description": "언어 감지 -> 번역 -> 품질 검증",
        "steps": ["language_detector", "translator", "quality_check"],
        "factory": create_translation_chain,
    },
}


def list_chain_templates() -> list[dict]:
    """List available chain templates."""
    return [
        {
            "id": key,
            "name": info["name"],
            "description": info["description"],
            "steps": info["steps"],
        }
        for key, info in CHAIN_TEMPLATES.items()
    ]


def get_chain_template(name: str) -> dict | None:
    """Get a chain template by name."""
    return CHAIN_TEMPLATES.get(name)
