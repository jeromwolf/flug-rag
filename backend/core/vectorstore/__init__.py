"""Vector store implementations for flux-rag."""

from config.settings import settings

from .base import BaseVectorStore, SearchResult
from .chroma import ChromaStore

try:
    from .milvus import MilvusStore
except ImportError:
    MilvusStore = None  # type: ignore[assignment,misc]


def create_vectorstore(
    store_type: str | None = None,
    collection_name: str | None = None,
    **kwargs,
) -> BaseVectorStore:
    """Create a vector store instance.

    Args:
        store_type: "chroma", "milvus_lite", or "milvus". Defaults to settings.vectorstore_type.
        collection_name: Collection name override.
        **kwargs: Extra store-specific arguments.
    """
    store_type = store_type or settings.vectorstore_type

    if store_type == "chroma":
        return ChromaStore(
            persist_dir=kwargs.get("persist_dir", settings.chroma_persist_dir),
            collection_name=collection_name or settings.chroma_collection_name,
        )
    elif store_type in ("milvus", "milvus_lite"):
        if MilvusStore is None:
            raise ImportError("pymilvus is required for Milvus. Install: pip install pymilvus")
        resolved_collection = collection_name or getattr(
            settings, "milvus_collection_name", None
        ) or settings.chroma_collection_name
        if store_type == "milvus_lite":
            uri = kwargs.get("uri", settings.milvus_store_uri)
        else:
            # For "milvus": use milvus_uri if it looks like a URL, else construct from host:port
            uri = kwargs.get("uri", settings.milvus_store_uri)
            if not uri.startswith("http"):
                uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        return MilvusStore(
            uri=uri,
            token=kwargs.get("token", settings.milvus_store_token),
            collection_name=resolved_collection,
            dimension=settings.embedding_dimension,
            metric_type=kwargs.get("metric_type", settings.milvus_metric_type),
        )
    else:
        raise ValueError(
            f"Unknown vector store type: {store_type}. Available: ['chroma', 'milvus_lite', 'milvus']"
        )


__all__ = ["BaseVectorStore", "SearchResult", "ChromaStore", "MilvusStore", "create_vectorstore"]
