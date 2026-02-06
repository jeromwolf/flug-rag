"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class LLMResponse:
    """Standard LLM response."""
    content: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens
    raw: dict = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    provider_name: str = "base"

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 2048, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a complete response."""
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response token by token."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Generate and parse JSON response."""
        import json
        response = await self.generate(
            prompt=prompt,
            system=(system or "") + "\n\nJSON 형식으로만 응답하세요.",
            **kwargs,
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise ValueError(f"Failed to parse JSON from LLM response: {content[:200]}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
