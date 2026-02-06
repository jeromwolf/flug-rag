"""Unit tests for LLM abstraction layer."""

import pytest

from core.llm.base import BaseLLM, LLMResponse
from core.llm.factory import create_llm, list_providers


class TestLLMFactory:
    """Tests for LLM factory."""

    def test_list_providers(self):
        providers = list_providers()
        assert "vllm" in providers
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_create_vllm(self):
        llm = create_llm(provider="vllm")
        assert llm.provider_name == "vllm"
        assert isinstance(llm, BaseLLM)

    def test_create_ollama(self):
        llm = create_llm(provider="ollama")
        assert llm.provider_name == "ollama"
        assert isinstance(llm, BaseLLM)

    def test_create_openai(self):
        llm = create_llm(provider="openai", model="gpt-4o-mini")
        assert llm.provider_name == "openai"
        assert llm.model == "gpt-4o-mini"

    def test_create_anthropic(self):
        llm = create_llm(provider="anthropic")
        assert llm.provider_name == "anthropic"

    def test_create_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm(provider="unknown")

    def test_custom_temperature(self):
        llm = create_llm(provider="ollama", temperature=0.3)
        assert llm.temperature == 0.3

    def test_custom_max_tokens(self):
        llm = create_llm(provider="ollama", max_tokens=4096)
        assert llm.max_tokens == 4096

    def test_override_model(self):
        llm = create_llm(provider="ollama", model="llama3.1:8b")
        assert llm.model == "llama3.1:8b"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_basic_response(self):
        resp = LLMResponse(
            content="Hello",
            model="test-model",
            provider="test",
        )
        assert resp.content == "Hello"
        assert resp.model == "test-model"
        assert resp.provider == "test"
        assert resp.usage == {}
        assert resp.raw == {}

    def test_response_with_usage(self):
        resp = LLMResponse(
            content="World",
            model="gpt-4",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 5
