"""Email composition tool with Korean business templates."""

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from core.llm import BaseLLM, create_llm


TONE_PROMPTS = {
    "formal": (
        "격식체를 사용하여 공식적인 비즈니스 이메일을 작성하세요.\n"
        "경어체(-습니다, -시기 바랍니다)를 사용하고 정중한 인사말을 포함하세요."
    ),
    "semi-formal": (
        "반말이 아닌 존댓말을 사용하되, 다소 부드러운 톤으로 작성하세요.\n"
        "(-드립니다, -감사드립니다) 형태를 사용하세요."
    ),
    "casual": (
        "친근하면서도 예의 바른 톤으로 작성하세요.\n"
        "(-요, -습니다) 형태를 혼용해도 됩니다."
    ),
}

EMAIL_SYSTEM = """당신은 한국가스기술공사의 이메일 작성 전문가입니다.
비즈니스 이메일을 전문적으로 작성합니다.

이메일 구조:
1. 수신자 정보
2. 제목
3. 인사말
4. 본문 (요청 배경, 핵심 내용, 요청/안내 사항)
5. 마무리 인사
6. 발신자 정보

{tone_instruction}

이메일 내용만 작성하세요."""


class EmailComposerTool(BaseTool):
    """Compose professional Korean business emails."""

    def __init__(self, llm: BaseLLM | None = None):
        self._llm = llm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="email_composer",
            description="비즈니스 이메일 초안을 작성합니다. 격식체, 반격식체, 캐주얼 톤을 지원합니다.",
            category="document",
            parameters=[
                ToolParameter(
                    name="subject",
                    type=ToolParamType.STRING,
                    description="이메일 제목 또는 제목 힌트",
                ),
                ToolParameter(
                    name="recipients",
                    type=ToolParamType.ARRAY,
                    description="수신자 목록 (예: ['홍길동 팀장', '김철수 과장'])",
                ),
                ToolParameter(
                    name="body_context",
                    type=ToolParamType.STRING,
                    description="이메일 본문에 포함할 내용/맥락",
                ),
                ToolParameter(
                    name="tone",
                    type=ToolParamType.STRING,
                    description="이메일 톤: formal(격식), semi-formal(반격식), casual(캐주얼)",
                    required=False,
                    default="formal",
                    enum=["formal", "semi-formal", "casual"],
                ),
                ToolParameter(
                    name="cc_list",
                    type=ToolParamType.ARRAY,
                    description="참조(CC) 수신자 목록",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        subject = kwargs.get("subject", "")
        recipients = kwargs.get("recipients", [])
        body_context = kwargs.get("body_context", "")
        tone = kwargs.get("tone", "formal")
        cc_list = kwargs.get("cc_list", [])

        if not subject:
            return ToolResult(success=False, error="subject is required")
        if not body_context:
            return ToolResult(success=False, error="body_context is required")

        try:
            email = await self.compose_email(
                subject_hint=subject,
                recipients=recipients,
                context=body_context,
                tone=tone,
                cc_list=cc_list,
            )
            return ToolResult(
                success=True,
                data={
                    "email": email,
                    "subject": subject,
                    "recipients": recipients,
                    "cc_list": cc_list,
                    "tone": tone,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Email composition error: {e}")

    async def compose_email(
        self,
        subject_hint: str,
        recipients: list[str] | None = None,
        context: str = "",
        tone: str = "formal",
        cc_list: list[str] | None = None,
    ) -> str:
        """Compose a business email."""
        recipients = recipients or []
        cc_list = cc_list or []
        tone_instruction = TONE_PROMPTS.get(tone, TONE_PROMPTS["formal"])

        system = EMAIL_SYSTEM.format(tone_instruction=tone_instruction)

        parts = [f"제목: {subject_hint}"]
        if recipients:
            parts.append(f"수신자: {', '.join(recipients)}")
        if cc_list:
            parts.append(f"참조: {', '.join(cc_list)}")
        parts.append(f"내용/맥락: {context}")

        prompt = "\n".join(parts)

        response = await self.llm.generate(
            prompt=prompt,
            system=system,
            temperature=0.4,
        )
        return response.content.strip()
