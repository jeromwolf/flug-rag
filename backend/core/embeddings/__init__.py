"""Embedding providers for flux-rag."""

from config.settings import settings

from .base import BaseEmbedding
from .local import LocalEmbedding
from .openai_embedder import OpenAIEmbedding


def create_embedder(provider: str = "local", **kwargs) -> BaseEmbedding:
    """Create an embedding instance.

    Args:
        provider: "local" (sentence-transformers) or "openai"
        **kwargs: Extra provider-specific arguments.
    """
    if provider == "local":
        return LocalEmbedding(
            model_name=kwargs.get("model_name", settings.embedding_model),
            device=kwargs.get("device"),
        )
    elif provider == "openai":
        return OpenAIEmbedding(
            api_key=kwargs.get("api_key", settings.openai_api_key),
            dimension=kwargs.get("dimension", settings.embedding_dimension),
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


__all__ = ["BaseEmbedding", "LocalEmbedding", "OpenAIEmbedding", "create_embedder"]
