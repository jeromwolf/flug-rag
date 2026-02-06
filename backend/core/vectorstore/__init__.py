"""Vector store implementations for flux-rag."""

from config.settings import settings

from .base import BaseVectorStore, SearchResult
from .chroma import ChromaStore

try:
    from .milvus import MilvusStore
except ImportError:
    MilvusStore = None  # type: ignore[assignment,misc]


def create_vectorstore(
    store_type: str = "chroma",
    collection_name: str | None = None,
    **kwargs,
) -> BaseVectorStore:
    """Create a vector store instance.

    Args:
        store_type: "chroma" (default) or "milvus" for production.
        collection_name: Collection name override.
        **kwargs: Extra store-specific arguments.
    """
    if store_type == "chroma":
        return ChromaStore(
            persist_dir=kwargs.get("persist_dir", settings.chroma_persist_dir),
            collection_name=collection_name or settings.chroma_collection_name,
        )
    elif store_type == "milvus":
        if MilvusStore is None:
            raise ImportError("pymilvus is required for Milvus. Install: pip install pymilvus")
        return MilvusStore(
            host=kwargs.get("host", settings.milvus_host),
            port=kwargs.get("port", settings.milvus_port),
            collection_name=collection_name or settings.chroma_collection_name,
            dimension=settings.embedding_dimension,
            index_type=kwargs.get("index_type", settings.milvus_index_type),
            metric_type=kwargs.get("metric_type", settings.milvus_metric_type),
        )
    else:
        raise ValueError(
            f"Unknown vector store type: {store_type}. Available: ['chroma', 'milvus']"
        )


__all__ = ["BaseVectorStore", "SearchResult", "ChromaStore", "MilvusStore", "create_vectorstore"]
