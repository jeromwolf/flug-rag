"""Report generation tool with templates for KOGAS."""

import json
from datetime import datetime

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from core.llm import BaseLLM, create_llm


# Report templates for Korean Gas Safety Corporation
REPORT_TEMPLATES = {
    "safety_inspection": {
        "name": "안전 점검 보고서",
        "description": "가스 시설 안전 점검 결과 보고서",
        "required_fields": ["facility_name", "inspector", "date", "findings"],
        "system_prompt": (
            "당신은 한국가스기술공사의 안전 점검 보고서 작성 전문가입니다.\n"
            "제공된 데이터를 기반으로 전문적이고 체계적인 안전 점검 보고서를 작성하세요.\n"
            "보고서는 다음 섹션을 포함해야 합니다:\n"
            "1. 점검 개요\n2. 점검 결과\n3. 지적 사항\n4. 개선 권고사항\n5. 결론"
        ),
    },
    "monthly_summary": {
        "name": "월간 요약 보고서",
        "description": "월간 업무 요약 보고서",
        "required_fields": ["department", "period", "activities"],
        "system_prompt": (
            "당신은 한국가스기술공사의 월간 보고서 작성 전문가입니다.\n"
            "제공된 데이터를 기반으로 월간 요약 보고서를 작성하세요.\n"
            "보고서는 다음 섹션을 포함해야 합니다:\n"
            "1. 기간 및 부서 정보\n2. 주요 업무 실적\n3. 주요 이슈\n4. 다음 달 계획"
        ),
    },
    "incident_report": {
        "name": "사고/이상 보고서",
        "description": "가스 관련 사고 또는 이상 상황 보고서",
        "required_fields": ["incident_type", "location", "date", "description"],
        "system_prompt": (
            "당신은 한국가스기술공사의 사고 보고서 작성 전문가입니다.\n"
            "제공된 데이터를 기반으로 사고/이상 보고서를 작성하세요.\n"
            "보고서는 다음 섹션을 포함해야 합니다:\n"
            "1. 사고 개요\n2. 상세 경위\n3. 피해 현황\n4. 원인 분석\n5. 조치 사항\n6. 재발 방지 대책"
        ),
    },
}


class ReportGeneratorTool(BaseTool):
    """Generate structured reports using templates and LLM."""

    def __init__(self, llm: BaseLLM | None = None):
        self._llm = llm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def get_definition(self) -> ToolDefinition:
        template_names = list(REPORT_TEMPLATES.keys())
        return ToolDefinition(
            name="report_generator",
            description="보고서를 생성합니다. 안전 점검, 월간 요약, 사고 보고서 템플릿을 지원합니다.",
            category="document",
            parameters=[
                ToolParameter(
                    name="template_name",
                    type=ToolParamType.STRING,
                    description="보고서 템플릿 이름",
                    enum=template_names,
                ),
                ToolParameter(
                    name="data",
                    type=ToolParamType.OBJECT,
                    description="보고서에 포함할 데이터 (딕셔너리)",
                ),
                ToolParameter(
                    name="output_format",
                    type=ToolParamType.STRING,
                    description="출력 형식: markdown 또는 json",
                    required=False,
                    default="markdown",
                    enum=["markdown", "json"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        template_name = kwargs.get("template_name", "")
        data = kwargs.get("data", {})
        output_format = kwargs.get("output_format", "markdown")

        if not template_name:
            return ToolResult(success=False, error="template_name is required")
        if template_name not in REPORT_TEMPLATES:
            return ToolResult(
                success=False,
                error=f"Unknown template: {template_name}. Available: {list(REPORT_TEMPLATES.keys())}",
            )
        if not data:
            return ToolResult(success=False, error="data parameter is required")

        try:
            report = await self.generate_report(template_name, data, output_format)
            return ToolResult(
                success=True,
                data={
                    "report": report,
                    "template": template_name,
                    "format": output_format,
                    "generated_at": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Report generation error: {e}")

    async def generate_report(self, template_name: str, data: dict, output_format: str = "markdown") -> str | dict:
        """Generate a report from template and data."""
        template = REPORT_TEMPLATES[template_name]

        # Build prompt with data
        data_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)

        if output_format == "json":
            format_instruction = (
                "보고서를 JSON 형식으로 작성하세요. "
                "각 섹션을 키-값 쌍으로 구성하세요."
            )
        else:
            format_instruction = (
                "보고서를 마크다운 형식으로 작성하세요. "
                "제목, 소제목, 목록 등을 활용하여 가독성 있게 작성하세요."
            )

        prompt = (
            f"다음 데이터를 기반으로 '{template['name']}'을(를) 작성하세요.\n\n"
            f"데이터:\n{data_str}\n\n"
            f"{format_instruction}"
        )

        response = await self.llm.generate(
            prompt=prompt,
            system=template["system_prompt"],
            temperature=0.3,
        )

        content = response.content.strip()

        if output_format == "json":
            try:
                # Attempt to parse as JSON
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                return json.loads(content)
            except (json.JSONDecodeError, IndexError):
                # Return as structured dict if parsing fails
                return {"content": response.content.strip()}

        return content
