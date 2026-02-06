"""Document summarization tool using LLM."""

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from core.llm import BaseLLM, create_llm


EXTRACTIVE_SYSTEM = """당신은 문서 요약 전문가입니다.
주어진 텍스트에서 핵심 키워드와 중요 문장을 추출하여 요약하세요.
원문의 핵심 문장을 그대로 사용하여 간결하게 정리하세요."""

ABSTRACTIVE_SYSTEM = """당신은 문서 요약 전문가입니다.
주어진 텍스트를 읽고 새로운 문장으로 핵심 내용을 요약하세요.
원문의 의미를 정확히 유지하면서 간결하고 자연스러운 한국어로 작성하세요."""


class SummarizerTool(BaseTool):
    """Document summarization using extractive or abstractive methods."""

    def __init__(self, llm: BaseLLM | None = None):
        self._llm = llm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="document_summarizer",
            description="문서 또는 텍스트를 요약합니다. 추출적(키워드 기반) 또는 추상적(LLM 기반) 요약을 지원합니다.",
            category="nlp",
            parameters=[
                ToolParameter(
                    name="text",
                    type=ToolParamType.STRING,
                    description="요약할 텍스트",
                ),
                ToolParameter(
                    name="mode",
                    type=ToolParamType.STRING,
                    description="요약 방식: extractive(추출적) 또는 abstractive(추상적)",
                    required=False,
                    default="abstractive",
                    enum=["extractive", "abstractive"],
                ),
                ToolParameter(
                    name="max_length",
                    type=ToolParamType.INTEGER,
                    description="요약 최대 길이 (글자 수)",
                    required=False,
                    default=500,
                ),
                ToolParameter(
                    name="language",
                    type=ToolParamType.STRING,
                    description="요약 언어",
                    required=False,
                    default="ko",
                    enum=["ko", "en"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        if not text:
            return ToolResult(success=False, error="text parameter is required")

        mode = kwargs.get("mode", "abstractive")
        max_length = kwargs.get("max_length", 500)
        language = kwargs.get("language", "ko")

        try:
            if mode == "extractive":
                summary = await self._extractive_summarize(text, max_length, language)
            else:
                summary = await self._abstractive_summarize(text, max_length, language)

            return ToolResult(
                success=True,
                data={
                    "summary": summary,
                    "mode": mode,
                    "original_length": len(text),
                    "summary_length": len(summary),
                },
                metadata={"language": language, "max_length": max_length},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Summarization error: {e}")

    async def _extractive_summarize(self, text: str, max_length: int, language: str) -> str:
        """Extract key sentences from text using LLM."""
        lang_instruction = "한국어로 요약하세요." if language == "ko" else "Summarize in English."
        prompt = (
            f"다음 텍스트에서 핵심 문장을 추출하여 {max_length}자 이내로 요약하세요.\n"
            f"{lang_instruction}\n\n"
            f"텍스트:\n{text}"
        )
        response = await self.llm.generate(
            prompt=prompt,
            system=EXTRACTIVE_SYSTEM,
            temperature=0.1,
        )
        return response.content.strip()[:max_length]

    async def _abstractive_summarize(self, text: str, max_length: int, language: str) -> str:
        """Generate new summary using LLM."""
        lang_instruction = "한국어로 요약하세요." if language == "ko" else "Summarize in English."
        prompt = (
            f"다음 텍스트를 {max_length}자 이내로 요약하세요.\n"
            f"{lang_instruction}\n\n"
            f"텍스트:\n{text}"
        )
        response = await self.llm.generate(
            prompt=prompt,
            system=ABSTRACTIVE_SYSTEM,
            temperature=0.3,
        )
        return response.content.strip()[:max_length]

    async def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Convenience method: summarize text with default settings."""
        result = await self.execute(text=text, max_length=max_length)
        if result.success:
            return result.data["summary"]
        raise RuntimeError(result.error)

    async def summarize_document(self, doc_id: str) -> str:
        """Summarize a document by ID. Requires vectorstore access."""
        from core.vectorstore import create_vectorstore
        vs = create_vectorstore()
        items = await vs.get(ids=[doc_id])
        if not items:
            raise ValueError(f"Document not found: {doc_id}")
        text = items[0].get("content", "") if isinstance(items[0], dict) else str(items[0])
        return await self.summarize_text(text)
