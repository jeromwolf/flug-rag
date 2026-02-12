"""Ollama LLM provider."""

from typing import AsyncIterator, Optional

import httpx

from .base import BaseLLM, LLMResponse


# Module-level connection pool singleton
_pool_manager = None


def get_llm_pool_manager():
    """Get the LLM connection pool manager."""
    global _pool_manager
    return _pool_manager


def set_llm_pool_manager(manager):
    """Set the LLM connection pool manager (called from app startup)."""
    global _pool_manager
    _pool_manager = manager


class OllamaProvider(BaseLLM):
    """Ollama local LLM provider."""

    provider_name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._own_client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client, preferring shared pool if available."""
        pool = get_llm_pool_manager()
        if pool is not None:
            return pool.get_client(self.base_url, timeout=120.0)

        # Fallback: create own client
        if self._own_client is None:
            self._own_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,
            )
        return self._own_client

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

        client = self._get_client()
        response = await client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            provider=self.provider_name,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
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

        client = self._get_client()
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()
            import json
            async for line in response.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done", False):
                        break

    async def close(self):
        if self._own_client is not None:
            await self._own_client.aclose()
            self._own_client = None
