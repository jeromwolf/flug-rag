"""Translation tool using LLM for multi-language support."""

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from core.llm import BaseLLM, create_llm


SUPPORTED_LANGUAGES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}

TRANSLATOR_SYSTEM = (
    "You are a professional translator. "
    "Translate the following text from {source_lang} to {target_lang}. "
    "Maintain the original meaning and tone. "
    "Return ONLY the translated text."
)


class TranslatorTool(BaseTool):
    """텍스트 번역 도구 -- LLM 기반 다국어 번역"""

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
            description="텍스트를 다른 언어로 번역합니다. 한국어, 영어, 일본어, 중국어를 지원합니다.",
            category="utility",
            help_text=(
                "LLM 기반 전문 번역 도구입니다.\n"
                "지원 언어: ko(한국어), en(영어), ja(일본어), zh(중국어)\n"
                "source_lang을 'auto'로 설정하면 자동 감지합니다.\n"
                "전문 용어를 정확히 번역하며, 번역 결과만 반환합니다."
            ),
            parameters=[
                ToolParameter(
                    name="text",
                    type=ToolParamType.STRING,
                    description="번역할 텍스트",
                ),
                ToolParameter(
                    name="source_lang",
                    type=ToolParamType.STRING,
                    description="원본 언어 코드 (auto, ko, en, ja, zh). 기본값: auto",
                    required=False,
                    default="auto",
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
        source_lang = kwargs.get("source_lang", "auto")
        target_lang = kwargs.get("target_lang", "")

        if not text:
            return ToolResult(success=False, error="text parameter is required")
        if not target_lang:
            return ToolResult(success=False, error="target_lang parameter is required")
        if target_lang not in SUPPORTED_LANGUAGES:
            return ToolResult(
                success=False,
                error=f"Unsupported target language: {target_lang}. "
                      f"Supported: {list(SUPPORTED_LANGUAGES.keys())}",
            )
        if source_lang != "auto" and source_lang not in SUPPORTED_LANGUAGES:
            return ToolResult(
                success=False,
                error=f"Unsupported source language: {source_lang}. "
                      f"Use 'auto' or one of: {list(SUPPORTED_LANGUAGES.keys())}",
            )
        if source_lang == target_lang:
            return ToolResult(
                success=True,
                data={
                    "translated_text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
            )

        try:
            translated = await self._translate(text, source_lang, target_lang)
            return ToolResult(
                success=True,
                data={
                    "translated_text": translated,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "target_language": SUPPORTED_LANGUAGES[target_lang],
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Translation error: {e}")

    async def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using the LLM."""
        source_name = SUPPORTED_LANGUAGES.get(source_lang, "auto-detected language")
        target_name = SUPPORTED_LANGUAGES[target_lang]

        system = TRANSLATOR_SYSTEM.format(
            source_lang=source_name,
            target_lang=target_name,
        )

        prompt = f"Translate the following text:\n\n{text}"

        response = await self.llm.generate(
            prompt=prompt,
            system=system,
            temperature=0.2,
        )
        return response.content.strip()
