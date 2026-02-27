"""
교육자료 자동 생성 MCP 도구.

사규, 기술 매뉴얼, ISO 규정을 기반으로 교육 콘텐츠를 구조화하여 생성.
예: "신입 엔지니어 배관 검사 교육자료 만들어줘"
"""
import logging

from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)

# 수준별 학습 깊이 지침
LEVEL_GUIDELINES = {
    "신입": {
        "description": "기초 개념 중심, 용어 설명 포함, 안전 수칙 강조",
        "objective_prefix": "기본 개념을 이해하고",
        "depth": "핵심 개념과 기본 절차 위주로 설명하세요. 전문 용어는 반드시 풀어서 설명하세요.",
    },
    "중급": {
        "description": "실무 적용 중심, 판단 기준 포함, 사례 기반",
        "objective_prefix": "실무에 적용할 수 있도록",
        "depth": "실무 적용 방법과 판단 기준을 중심으로 설명하세요. 실제 사례를 포함하세요.",
    },
    "고급": {
        "description": "심화 분석, 문제 해결, 규정 해석 능력",
        "objective_prefix": "심화 역량을 갖추고",
        "depth": "심화 내용과 예외 상황, 규정 해석 방법을 포함하세요. 문제 해결 접근법을 제시하세요.",
    },
}

# 자료 형식별 구성 지침
FORMAT_GUIDELINES = {
    "교안": {
        "description": "강의 진행용 교안 (학습목표 → 핵심내용 → 정리)",
        "sections": ["학습 목표", "사전 지식 확인", "핵심 개념", "상세 설명", "실습/적용", "핵심 정리", "참고 규정"],
    },
    "체크리스트": {
        "description": "현장 적용용 점검 체크리스트",
        "sections": ["준비사항 확인", "핵심 점검 항목", "주의사항", "완료 확인"],
    },
    "퀴즈": {
        "description": "학습 성취도 확인용 퀴즈",
        "sections": ["OX 문제", "단답형 문제", "서술형 문제", "정답 및 해설"],
    },
    "종합": {
        "description": "교안 + 체크리스트 + 퀴즈 통합 패키지",
        "sections": ["학습 목표", "핵심 개념", "절차 및 방법", "현장 체크리스트", "확인 퀴즈", "참고 규정"],
    },
}


class TrainingMaterialTool(BaseTool):
    """교육 주제와 수준에 맞는 교육자료를 RAG 검색 기반으로 자동 생성."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="training_material",
            description="교육자료를 자동 생성합니다. 관련 사규, ISO 규정, 기술 매뉴얼을 검색하여 학습 목표와 핵심 내용을 구조화합니다.",
            parameters=[
                ToolParameter(
                    name="topic",
                    type=ToolParamType.STRING,
                    description="교육 주제 (예: 배관 용접 검사, 가스 누출 대응 절차, 정압기 유지보수)",
                ),
                ToolParameter(
                    name="level",
                    type=ToolParamType.STRING,
                    description="교육 대상 수준",
                    required=False,
                    default="신입",
                    enum=list(LEVEL_GUIDELINES.keys()),
                ),
                ToolParameter(
                    name="format",
                    type=ToolParamType.STRING,
                    description="자료 형식 (교안: 강의용, 체크리스트: 현장용, 퀴즈: 평가용, 종합: 통합 패키지)",
                    required=False,
                    default="종합",
                    enum=list(FORMAT_GUIDELINES.keys()),
                ),
            ],
            category="generation",
        )

    async def execute(self, **kwargs) -> ToolResult:
        topic = kwargs.get("topic", "").strip()
        level = kwargs.get("level", "신입")
        fmt = kwargs.get("format", "종합")

        if not topic:
            return ToolResult(success=False, error="교육 주제를 입력해주세요.")

        try:
            # Step 1: RAG 검색으로 관련 규정/매뉴얼 수집
            from rag.retriever import HybridRetriever

            retriever = HybridRetriever()

            # 주제 관련 규정, 기술 매뉴얼, 절차서 검색
            search_queries = [
                f"{topic} 관련 규정 기준 절차",
                f"{topic} 안전 주의사항",
            ]

            all_results = []
            seen_contents = set()
            for query in search_queries:
                results = await retriever.retrieve(query=query)
                for r in results:
                    # 중복 제거 (앞 100자 기준)
                    key = r.content[:100]
                    if key not in seen_contents:
                        seen_contents.add(key)
                        all_results.append(r)

            # 상위 8개만 사용
            top_results = all_results[:8]

            # Step 2: 컨텍스트 구성
            context_parts = []
            for i, r in enumerate(top_results):
                filename = r.metadata.get("filename", "미상")
                page = r.metadata.get("page_number", "")
                ref = f"{filename}" + (f" (p.{page})" if page else "")
                context_parts.append(
                    f"[참고자료 {i+1}] 출처: {ref}\n{r.content}"
                )
            references_context = "\n\n".join(context_parts) if context_parts else "관련 문서 없음"

            # Step 3: LLM으로 교육자료 생성
            from core.llm import create_llm

            llm = create_llm()

            level_info = LEVEL_GUIDELINES.get(level, LEVEL_GUIDELINES["신입"])
            format_info = FORMAT_GUIDELINES.get(fmt, FORMAT_GUIDELINES["종합"])
            toc = "\n".join(f"  - {s}" for s in format_info["sections"])

            system_prompt = f"""당신은 한국가스기술공사의 전문 교육자료 개발자입니다.
주어진 주제와 수준에 맞는 실용적인 교육자료를 작성하세요.

교육 대상 수준: {level} ({level_info['description']})
자료 형식: {fmt} ({format_info['description']})

작성 지침:
- {level_info['depth']}
- 한국가스기술공사 현장 실무에 적합한 내용으로 작성하세요.
- 관련 규정 및 기준을 근거로 제시하세요.
- 안전 관련 사항은 반드시 강조 표시(⚠️ 또는 [주의])하세요.
- 내용이 불확실하거나 확인이 필요한 경우 '[확인 필요]'로 표시하세요."""

            user_prompt = f"""## 교육자료 생성 요청
- 주제: {topic}
- 대상: {level} 수준
- 형식: {fmt}

## 포함할 섹션
{toc}

## 참고 자료
{references_context}

위 참고 자료를 활용하여 {level} 대상 {fmt} 형식의 교육자료를 작성하세요.
- 학습 목표를 명확히 서술하세요.
- 핵심 내용을 구조화하여 전달하세요.
- 참고 자료가 있는 내용은 출처를 표기하세요.
- 없는 내용은 '[작성 필요]'로 표시하세요."""

            response = await llm.generate(prompt=user_prompt, system=system_prompt)

            # Step 4: 출처 목록 구성
            source_refs = [
                f"- {r.metadata.get('filename', '미상')}"
                + (f" (p.{r.metadata.get('page_number', '')})" if r.metadata.get("page_number") else "")
                + f" [신뢰도 {r.score:.2f}]"
                for r in top_results
            ]

            final_output = f"""{response.content}

---
## 참조 문서 목록
{chr(10).join(source_refs) if source_refs else '- 참조 문서 없음'}

*본 교육자료는 AI가 생성한 초안으로, 최종 배포 전 담당 전문가 검토가 필요합니다.*"""

            return ToolResult(
                success=True,
                data={"material": final_output, "reference_count": len(top_results)},
                metadata={
                    "topic": topic,
                    "level": level,
                    "format": fmt,
                    "sources": [r.metadata.get("filename", "미상") for r in top_results],
                },
            )

        except Exception as e:
            logger.error("Training material generation failed: %s", e)
            return ToolResult(success=False, error=f"교육자료 생성 중 오류가 발생했습니다: {str(e)}")
