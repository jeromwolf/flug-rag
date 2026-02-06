"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Single search result from vector store."""
    id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base for vector store implementations."""

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents to the store."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        ...

    @abstractmethod
    async def get(self, ids: list[str]) -> list[dict]:
        """Get documents by IDs."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get total document count."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all documents."""
        ...
