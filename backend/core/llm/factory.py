"""LLM Factory for creating provider instances from settings."""

from typing import Dict, List, Optional, Type

from config.settings import settings

from .base import BaseLLM
from .anthropic_provider import AnthropicProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .vllm_provider import VLLMProvider

_PROVIDERS: Dict[str, Type[BaseLLM]] = {
    "vllm": VLLMProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

# Default configs per provider (from settings)
_DEFAULT_CONFIGS: Dict[str, dict] = {
    "vllm": {
        "model": settings.vllm_model,
        "base_url": settings.vllm_base_url,
        "api_key": settings.vllm_api_key,
    },
    "ollama": {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
    },
    "openai": {
        "model": settings.openai_model,
        "api_key": settings.openai_api_key,
    },
    "anthropic": {
        "model": settings.anthropic_model,
        "api_key": settings.anthropic_api_key,
    },
}


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float | None = None,
    max_tokens: int = 2048,
    **kwargs,
) -> BaseLLM:
    """Create an LLM instance.

    Args:
        provider: Provider name (vllm, ollama, openai, anthropic).
                  Defaults to settings.default_llm_provider.
        model: Model name. Defaults to provider's default model.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        **kwargs: Extra provider-specific arguments.

    Returns:
        BaseLLM instance.
    """
    provider = provider or settings.default_llm_provider

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(_PROVIDERS.keys())}"
        )

    # Merge defaults with overrides
    config = {**_DEFAULT_CONFIGS.get(provider, {})}
    if model:
        config["model"] = model
    config["temperature"] = temperature if temperature is not None else settings.llm_temperature
    config["max_tokens"] = max_tokens
    config.update(kwargs)

    return _PROVIDERS[provider](**config)


def get_default_llm() -> BaseLLM:
    """Get the default LLM based on settings."""
    return create_llm()


def list_providers() -> List[str]:
    """List available provider names."""
    return list(_PROVIDERS.keys())
