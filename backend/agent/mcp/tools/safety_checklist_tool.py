"""
안전 체크리스트 생성 에이전트 도구
SFR-013: 설비 유형별 안전 체크리스트 + 규정 매핑
"""
import logging
from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)

# 설비 유형별 기본 체크리스트 템플릿
EQUIPMENT_TEMPLATES = {
    "배관": [
        "배관 외관 부식 상태 점검",
        "용접부 균열 점검",
        "배관 지지대 상태 확인",
        "가스 누출 탐지 시험",
        "배관 두께 측정",
        "방식(防蝕) 설비 상태",
        "밸브 작동 상태 점검",
    ],
    "정압기": [
        "정압기 입/출구 압력 확인",
        "안전밸브 작동 시험",
        "필터 청소/교환 상태",
        "다이어프램 상태 점검",
        "배기관 상태 확인",
        "계기류 교정 상태",
        "외함 부식/손상 점검",
    ],
    "저장탱크": [
        "탱크 외관 점검",
        "기초 및 앵커볼트 상태",
        "안전밸브 작동 시험",
        "액면계 정상 작동 확인",
        "긴급차단밸브 작동 시험",
        "방류벽 상태 점검",
        "접지 저항 측정",
        "가스감지기 정상 작동 확인",
    ],
    "공급설비": [
        "가스공급 차단장치 점검",
        "압력조정기 작동 확인",
        "가스미터 정확도 점검",
        "배관 연결부 누출 검사",
        "환기설비 상태 확인",
        "가스감지 및 경보장치 점검",
    ],
    "일반": [
        "안전장비 비치 상태",
        "비상연락망 게시 여부",
        "소화기 비치 및 유효기간",
        "안전표지판 설치 상태",
        "작업자 보호구 착용 상태",
        "작업 허가서 발급 여부",
    ],
}


class SafetyChecklistTool(BaseTool):
    """설비 유형별 안전 체크리스트를 생성하고 관련 규정을 자동 매핑."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="safety_checklist",
            description="설비 유형에 맞는 안전 체크리스트를 생성하고 관련 규정을 RAG로 매핑합니다.",
            parameters=[
                ToolParameter(
                    name="equipment_type",
                    type=ToolParamType.STRING,
                    description="설비 유형: 배관, 정압기, 저장탱크, 공급설비, 일반",
                    enum=list(EQUIPMENT_TEMPLATES.keys()),
                ),
                ToolParameter(
                    name="additional_items",
                    type=ToolParamType.STRING,
                    description="추가 점검 항목 (쉼표 구분)",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="output_format",
                    type=ToolParamType.STRING,
                    description="출력 형식: markdown, text",
                    required=False,
                    default="markdown",
                    enum=["markdown", "text"],
                ),
            ],
            category="analysis",
        )

    async def execute(self, **kwargs) -> ToolResult:
        equipment_type = kwargs.get("equipment_type", "일반")
        additional_items = kwargs.get("additional_items", "")
        output_format = kwargs.get("output_format", "markdown")

        try:
            # Step 1: 기본 템플릿 로드
            checklist_items = list(EQUIPMENT_TEMPLATES.get(equipment_type, EQUIPMENT_TEMPLATES["일반"]))

            # 추가 항목 반영
            if additional_items:
                extras = [item.strip() for item in additional_items.split(",") if item.strip()]
                checklist_items.extend(extras)

            # Step 2: RAG로 각 항목에 관련 규정 매핑
            from rag.retriever import HybridRetriever
            retriever = HybridRetriever()

            checklist_with_regulations = []
            for item in checklist_items:
                search_query = f"{equipment_type} {item} 안전 규정 기준"
                results = await retriever.retrieve(query=search_query)

                regulation_ref = ""
                if results and results[0].score > 0.3:
                    top = results[0]
                    filename = top.metadata.get("filename", "")
                    page = top.metadata.get("page_number", "")
                    regulation_ref = f"{filename}"
                    if page:
                        regulation_ref += f" (p.{page})"

                checklist_with_regulations.append({
                    "item": item,
                    "regulation": regulation_ref,
                    "score": results[0].score if results else 0,
                })

            # Step 3: 출력 생성
            if output_format == "markdown":
                output = self._format_markdown(equipment_type, checklist_with_regulations)
            else:
                output = self._format_text(equipment_type, checklist_with_regulations)

            return ToolResult(success=True, data=output)

        except Exception as e:
            logger.error("Safety checklist generation failed: %s", e)
            return ToolResult(success=False, error=f"체크리스트 생성 중 오류: {str(e)}")

    def _format_markdown(self, equipment_type: str, items: list[dict]) -> str:
        lines = [
            f"# {equipment_type} 안전 점검 체크리스트",
            "",
            f"| No. | 점검 항목 | 적합 | 부적합 | 해당없음 | 근거 규정 |",
            f"|-----|----------|:----:|:------:|:-------:|----------|",
        ]
        for i, item in enumerate(items, 1):
            reg = item["regulation"] or "-"
            lines.append(f"| {i} | {item['item']} | [ ] | [ ] | [ ] | {reg} |")

        lines.extend([
            "",
            "## 점검 정보",
            "- 점검일: ____년 __월 __일",
            "- 점검자: ________________",
            "- 확인자: ________________",
            "",
            "## 특이사항",
            "",
            "________________",
        ])
        return "\n".join(lines)

    def _format_text(self, equipment_type: str, items: list[dict]) -> str:
        lines = [f"[{equipment_type} 안전 점검 체크리스트]", ""]
        for i, item in enumerate(items, 1):
            reg = f" (근거: {item['regulation']})" if item["regulation"] else ""
            lines.append(f"{i}. [ ] {item['item']}{reg}")
        lines.extend(["", "점검일:", "점검자:", "확인자:", "", "특이사항:"])
        return "\n".join(lines)
