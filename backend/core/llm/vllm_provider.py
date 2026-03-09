"""vLLM provider using OpenAI-compatible API."""

import json as _json
import logging
import re
from typing import AsyncIterator, Optional

import httpx

from .base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)

# Regex to extract max allowed input tokens from vLLM error message
_RE_MAX_INPUT = re.compile(r"maximum input length of (\d+) tokens")


class VLLMProvider(BaseLLM):
    """vLLM LLM provider via OpenAI-compatible endpoint."""

    provider_name = "vllm"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-placeholder",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def _handle_context_overflow(self, error_body: str, payload: dict) -> bool:
        """Handle context length overflow by reducing max_tokens.

        Returns True if payload was adjusted and should be retried.
        """
        match = _RE_MAX_INPUT.search(error_body)
        if match:
            max_input = int(match.group(1))
            # Set max_tokens to fit within context: leave at least 128 tokens for output
            new_max_tokens = max(128, payload["max_tokens"] - 100)
            logger.warning(
                "Context overflow (max_input=%d). Reducing max_tokens %d → %d",
                max_input, payload["max_tokens"], new_max_tokens,
            )
            payload["max_tokens"] = new_max_tokens
            return True
        return False

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
            **kwargs,
        }
        response = await self._client.post("chat/completions", json=payload)
        if response.status_code == 400:
            error_body = response.text[:500]
            logger.error("vLLM error %d: %s", response.status_code, error_body)
            if self._handle_context_overflow(error_body, payload):
                response = await self._client.post("chat/completions", json=payload)
        if response.status_code != 200:
            logger.error("vLLM error %d: %s", response.status_code, response.text[:500])
            response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            provider=self.provider_name,
            usage=data.get("usage", {}),
            raw=data,
        )

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True,
            **kwargs,
        }

        # Pre-check: try non-streaming first to detect context overflow
        check_payload = {**payload, "stream": False, "max_tokens": 1}
        check_resp = await self._client.post("chat/completions", json=check_payload)
        if check_resp.status_code == 400:
            error_body = check_resp.text[:500]
            logger.error("vLLM stream pre-check error: %s", error_body)
            if self._handle_context_overflow(error_body, payload):
                pass  # payload adjusted, continue with stream

        async with self._client.stream(
            "POST", "chat/completions", json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                error_text = body.decode("utf-8", errors="replace")[:500]
                logger.error("vLLM stream error %d: %s", response.status_code, error_text)
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    data = _json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content

    async def close(self):
        await self._client.aclose()
