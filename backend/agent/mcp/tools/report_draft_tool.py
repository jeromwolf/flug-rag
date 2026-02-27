"""
보고서 초안 자동 생성 MCP 도구.

사용자 요청에 따라 관련 문서를 RAG로 검색하고 보고서 초안을 자동 생성.
예: "이번 달 안전점검 결과 보고서 만들어줘"
"""
import logging
from datetime import datetime

from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)

# 보고서 유형별 기본 구조 템플릿
REPORT_STRUCTURES = {
    "결과보고서": [
        "1. 개요",
        "2. 추진 경위",
        "3. 주요 내용",
        "4. 결과 및 성과",
        "5. 향후 계획",
        "6. 참고 자료",
    ],
    "현황보고서": [
        "1. 현황 개요",
        "2. 세부 현황",
        "3. 주요 지표",
        "4. 문제점 및 개선사항",
        "5. 결론",
    ],
    "분석보고서": [
        "1. 분석 배경 및 목적",
        "2. 분석 범위 및 방법",
        "3. 분석 결과",
        "4. 시사점",
        "5. 결론 및 제언",
        "6. 참고 문헌",
    ],
    "점검보고서": [
        "1. 점검 개요",
        "2. 점검 대상 및 범위",
        "3. 점검 결과",
        "4. 부적합 사항",
        "5. 조치 현황",
        "6. 종합 의견",
    ],
}


class ReportDraftTool(BaseTool):
    """주제에 맞는 보고서 초안을 RAG 검색 기반으로 자동 생성."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="report_draft",
            description="주제에 맞는 보고서 초안을 자동 생성합니다. 관련 사규, 기술문서, 점검 이력을 검색하여 구조화된 보고서를 작성합니다.",
            parameters=[
                ToolParameter(
                    name="topic",
                    type=ToolParamType.STRING,
                    description="보고서 주제 (예: 배관 점검 결과 보고서, 2026년 안전관리 현황)",
                ),
                ToolParameter(
                    name="report_type",
                    type=ToolParamType.STRING,
                    description="보고서 유형",
                    required=False,
                    default="결과보고서",
                    enum=list(REPORT_STRUCTURES.keys()),
                ),
                ToolParameter(
                    name="period",
                    type=ToolParamType.STRING,
                    description="보고 기간 (예: 2026년 1월, 2025년 4분기)",
                    required=False,
                    default="",
                ),
            ],
            category="generation",
        )

    async def execute(self, **kwargs) -> ToolResult:
        topic = kwargs.get("topic", "").strip()
        report_type = kwargs.get("report_type", "결과보고서")
        period = kwargs.get("period", "")

        if not topic:
            return ToolResult(success=False, error="보고서 주제를 입력해주세요.")

        try:
            # Step 1: RAG 검색으로 관련 문서 수집
            from rag.retriever import HybridRetriever

            retriever = HybridRetriever()
            search_query = f"{topic} {report_type} 관련 내용 자료"
            results = await retriever.retrieve(query=search_query)

            if not results:
                logger.warning("No RAG results for report draft: %s", topic)

            # Step 2: 보고서 구조 결정
            structure = REPORT_STRUCTURES.get(report_type, REPORT_STRUCTURES["결과보고서"])

            # Step 3: 검색된 문서 컨텍스트 구성
            context_parts = []
            for i, r in enumerate(results[:6]):
                filename = r.metadata.get("filename", "미상")
                page = r.metadata.get("page_number", "")
                ref = f"{filename}" + (f" (p.{page})" if page else "")
                context_parts.append(
                    f"[참고문서 {i+1}] 출처: {ref}, 신뢰도: {r.score:.2f}\n{r.content}"
                )
            references_context = "\n\n".join(context_parts) if context_parts else "관련 문서 없음"

            # Step 4: LLM으로 보고서 초안 생성
            from core.llm import create_llm

            llm = create_llm()

            period_str = f"보고 기간: {period}\n" if period else ""
            today = datetime.now().strftime("%Y년 %m월 %d일")
            toc = "\n".join(f"  {s}" for s in structure)

            system_prompt = """당신은 한국가스기술공사의 전문 보고서 작성 어시스턴트입니다.
제공된 참고 문서와 주제를 바탕으로 전문적이고 체계적인 보고서 초안을 작성하세요.
- 한국 공공기관 문서 스타일을 준수합니다.
- 사실에 기반하여 작성하고 불확실한 내용은 '[확인 필요]'로 표시합니다.
- 각 섹션을 명확하게 구분하여 작성합니다.
- 출처가 있는 내용은 괄호 안에 출처를 표기합니다."""

            user_prompt = f"""## 보고서 작성 요청
- 주제: {topic}
- 유형: {report_type}
- {period_str}작성일: {today}

## 목차 구조
{toc}

## 참고 문서
{references_context}

위 목차 구조에 맞게 보고서 초안을 작성하세요.
참고 문서의 내용을 활용하되, 없는 내용은 '[작성 필요]' 표시로 남겨주세요.
보고서 제목과 작성 정보(수신, 발신, 제목, 작성일)를 포함해 주세요."""

            response = await llm.generate(prompt=user_prompt, system=system_prompt)

            # Step 5: 출처 목록 구성
            source_refs = [
                f"- {r.metadata.get('filename', '미상')}"
                + (f" (p.{r.metadata.get('page_number', '')})" if r.metadata.get("page_number") else "")
                + f" [신뢰도 {r.score:.2f}]"
                for r in results[:6]
            ]

            final_output = f"""{response.content}

---
## 참조 문서 목록
{chr(10).join(source_refs) if source_refs else '- 참조 문서 없음'}

*본 문서는 AI가 생성한 초안으로, 최종 보고 전 반드시 담당자 검토가 필요합니다.*"""

            return ToolResult(
                success=True,
                data={"draft": final_output, "reference_count": len(results)},
                metadata={
                    "topic": topic,
                    "report_type": report_type,
                    "period": period,
                    "sources": [r.metadata.get("filename", "미상") for r in results[:6]],
                },
            )

        except Exception as e:
            logger.error("Report draft generation failed: %s", e)
            return ToolResult(success=False, error=f"보고서 초안 생성 중 오류가 발생했습니다: {str(e)}")
