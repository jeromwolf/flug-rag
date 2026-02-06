"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract base for embedding models."""

    model_name: str = "base"
    dimension: int = 0

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        ...

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batch."""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query (may differ from document embedding)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, dim={self.dimension})"
