"""
규정 검토 에이전트 도구
SFR-013: 업로드 문서를 사내 규정과 대조 검토
"""
import logging
from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)


class RegulationReviewTool(BaseTool):
    """업로드 문서를 사내 규정과 대조하여 위반/불일치 항목을 식별."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="regulation_review",
            description="업로드된 문서의 내용을 관련 규정과 대조하여 위반/불일치 항목을 식별하고 검토 의견을 생성합니다.",
            parameters=[
                ToolParameter(
                    name="document_text",
                    type=ToolParamType.STRING,
                    description="검토 대상 문서 텍스트",
                ),
                ToolParameter(
                    name="regulation_category",
                    type=ToolParamType.STRING,
                    description="규정 카테고리 (예: 안전관리, 시설기준, 운영규정)",
                    required=False,
                    default="전체",
                ),
                ToolParameter(
                    name="review_depth",
                    type=ToolParamType.STRING,
                    description="검토 깊이: brief(요약), standard(표준), detailed(상세)",
                    required=False,
                    default="standard",
                    enum=["brief", "standard", "detailed"],
                ),
            ],
            category="analysis",
        )

    async def execute(self, **kwargs) -> ToolResult:
        document_text = kwargs.get("document_text", "")
        regulation_category = kwargs.get("regulation_category", "전체")
        review_depth = kwargs.get("review_depth", "standard")

        if not document_text.strip():
            return ToolResult(success=False, error="검토할 문서 텍스트가 비어있습니다.")

        try:
            # Step 1: RAG 검색으로 관련 규정 찾기
            from rag.retriever import HybridRetriever
            retriever = HybridRetriever()

            # 문서 핵심 키워드 추출 후 규정 검색
            search_query = f"{regulation_category} 규정 관련: {document_text[:500]}"
            filters = None
            if regulation_category != "전체":
                filters = {"category": regulation_category}

            results = await retriever.retrieve(query=search_query, filters=filters)

            if not results:
                return ToolResult(
                    success=False,
                    error="관련 규정을 찾을 수 없습니다. 규정 데이터가 인제스트되었는지 확인하세요.",
                )

            # Step 2: LLM으로 대조 분석
            from core.llm import create_llm

            llm = create_llm()
            regulations_context = "\n\n".join([
                f"[규정 {i+1}] (출처: {r.metadata.get('filename', '미상')}, 신뢰도: {r.score:.2f})\n{r.content}"
                for i, r in enumerate(results[:5])
            ])

            depth_instruction = {
                "brief": "주요 위반 사항만 간략히 나열하세요.",
                "standard": "각 위반/불일치 항목에 대해 근거 규정과 함께 설명하세요.",
                "detailed": "각 위반 항목에 대해 근거 규정, 위반 심각도, 개선 권고사항을 상세히 작성하세요.",
            }

            system_prompt = """당신은 한국가스기술공사의 규정 검토 전문가입니다.
제출된 문서를 관련 규정과 대조하여 다음을 식별하세요:
1. 규정 위반 항목 (위반 조항, 심각도)
2. 불일치 항목 (규정과 다른 부분)
3. 누락 항목 (규정에서 요구하지만 문서에 없는 사항)
4. 검토 의견 및 권고사항

결과는 구조화된 형식으로 작성하세요."""

            user_prompt = f"""## 검토 대상 문서
{document_text[:3000]}

## 관련 규정
{regulations_context}

## 검토 지시사항
{depth_instruction.get(review_depth, depth_instruction["standard"])}

위의 문서를 관련 규정과 대조하여 검토 보고서를 작성하세요."""

            response = await llm.generate(prompt=user_prompt, system=system_prompt)

            # Step 3: 결과 조합
            source_refs = [
                f"- {r.metadata.get('filename', '미상')} (p.{r.metadata.get('page_number', '?')}, 신뢰도: {r.score:.2f})"
                for r in results[:5]
            ]

            final_output = f"""# 규정 검토 보고서

## 검토 조건
- 카테고리: {regulation_category}
- 검토 깊이: {review_depth}
- 참조 규정 수: {len(results)}건

## 검토 결과
{response.content}

## 참조 규정 출처
{chr(10).join(source_refs)}
"""
            return ToolResult(
                success=True,
                data={"report": final_output, "regulation_count": len(results)},
                metadata={
                    "category": regulation_category,
                    "depth": review_depth,
                    "sources": [r.metadata.get("filename", "미상") for r in results[:5]],
                },
            )

        except Exception as e:
            logger.error("Regulation review failed: %s", e)
            return ToolResult(success=False, error=f"규정 검토 중 오류가 발생했습니다: {str(e)}")
