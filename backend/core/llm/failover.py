"""LLM failover wrapper: on primary failure, auto-switch to fallback provider."""

import logging
from typing import AsyncIterator, Optional

import httpx

from .base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)

_FAILOVER_ERRORS: tuple = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
    ConnectionError,
    TimeoutError,
)

try:
    import openai
    _FAILOVER_ERRORS = _FAILOVER_ERRORS + (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
    )
except ImportError:
    pass


class FailoverLLM(BaseLLM):
    """Wraps a primary LLM with automatic fallback on failure."""

    provider_name = "failover"

    def __init__(self, primary: BaseLLM, fallback: BaseLLM):
        self.primary = primary
        self.fallback = fallback
        self.model = primary.model
        self.temperature = primary.temperature
        self.max_tokens = primary.max_tokens

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        try:
            return await self.primary.generate(
                prompt=prompt, system=system,
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            )
        except _FAILOVER_ERRORS as e:
            logger.warning(
                "Primary LLM (%s/%s) failed: %s — failover to (%s/%s)",
                self.primary.provider_name, self.primary.model, e,
                self.fallback.provider_name, self.fallback.model,
            )
            return await self.fallback.generate(
                prompt=prompt, system=system,
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        try:
            gen = self.primary.stream(
                prompt=prompt, system=system,
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            )
            first = True
            async for token in gen:
                if first:
                    first = False
                yield token
            if first:
                raise RuntimeError("Primary stream empty")
        except _FAILOVER_ERRORS as e:
            logger.warning(
                "Primary LLM stream (%s/%s) failed: %s — failover to fallback",
                self.primary.provider_name, self.primary.model, e,
            )
            async for token in self.fallback.stream(
                prompt=prompt, system=system,
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            ):
                yield token

    async def close(self):
        await self.primary.close()
        await self.fallback.close()
