"""Translation tool using LLM for multi-language support."""

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from core.llm import BaseLLM, create_llm


SUPPORTED_LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
}

SUPPORTED_PAIRS = {
    ("ko", "en"), ("en", "ko"),
    ("ko", "ja"), ("ja", "ko"),
    ("ko", "zh"), ("zh", "ko"),
}

TRANSLATOR_SYSTEM = """당신은 전문 번역가입니다.
주어진 텍스트를 정확하고 자연스럽게 번역하세요.
전문 용어는 가능한 한 정확하게 번역하고, 문맥에 맞는 표현을 사용하세요.
번역 결과만 출력하세요. 추가 설명이나 주석은 포함하지 마세요."""


class TranslatorTool(BaseTool):
    """Multi-language translation using LLM."""

    def __init__(self, llm: BaseLLM | None = None):
        self._llm = llm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="translator",
            description="텍스트를 다른 언어로 번역합니다. 한국어↔영어, 한국어↔일본어, 한국어↔중국어를 지원합니다.",
            category="nlp",
            parameters=[
                ToolParameter(
                    name="text",
                    type=ToolParamType.STRING,
                    description="번역할 텍스트",
                ),
                ToolParameter(
                    name="source_lang",
                    type=ToolParamType.STRING,
                    description="원본 언어 코드 (ko, en, ja, zh)",
                    enum=["ko", "en", "ja", "zh"],
                ),
                ToolParameter(
                    name="target_lang",
                    type=ToolParamType.STRING,
                    description="대상 언어 코드 (ko, en, ja, zh)",
                    enum=["ko", "en", "ja", "zh"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        source_lang = kwargs.get("source_lang", "")
        target_lang = kwargs.get("target_lang", "")

        if not text:
            return ToolResult(success=False, error="text parameter is required")
        if not source_lang or not target_lang:
            return ToolResult(success=False, error="source_lang and target_lang are required")
        if source_lang == target_lang:
            return ToolResult(
                success=True,
                data={"translated_text": text, "source_lang": source_lang, "target_lang": target_lang},
            )
        if (source_lang, target_lang) not in SUPPORTED_PAIRS:
            return ToolResult(
                success=False,
                error=f"Unsupported language pair: {source_lang} -> {target_lang}. "
                      f"Supported: {', '.join(f'{a}->{b}' for a, b in sorted(SUPPORTED_PAIRS))}",
            )

        try:
            translated = await self.translate(text, source_lang, target_lang)
            return ToolResult(
                success=True,
                data={
                    "translated_text": translated,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_language": SUPPORTED_LANGUAGES[source_lang],
                    "target_language": SUPPORTED_LANGUAGES[target_lang],
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Translation error: {e}")

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages."""
        source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

        prompt = (
            f"다음 {source_name} 텍스트를 {target_name}로 번역하세요.\n\n"
            f"원문:\n{text}"
        )
        response = await self.llm.generate(
            prompt=prompt,
            system=TRANSLATOR_SYSTEM,
            temperature=0.2,
        )
        return response.content.strip()
