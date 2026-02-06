"""Anthropic Claude LLM provider."""

from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic

from .base import BaseLLM, LLMResponse


class AnthropicProvider(BaseLLM):
    """Anthropic Claude LLM provider."""

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.temperature,
            **kwargs,
        )

        content = response.content[0].text if response.content else ""

        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            raw=response.model_dump(),
        )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.temperature,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def close(self):
        await self._client.close()
