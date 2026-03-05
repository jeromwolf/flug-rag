"""Vector distribution analysis for monitoring and debugging."""

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .base import BaseVectorStore
from . import create_vectorstore

# Optional numpy import with fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class VectorDistribution:
    """Statistics about vector distribution in the collection."""
    total_vectors: int
    dimension: int
    norm_stats: dict  # min, max, mean, std
    outlier_count: int
    outlier_ids: list[str]  # IDs of outlier vectors
    analyzed_at: str


@dataclass
class CollectionHealth:
    """Health metrics for vector collection."""
    total_vectors: int
    collection_name: str
    index_type: str
    estimated_size_mb: float
    metadata: dict


class VectorAnalyzer:
    """Analyze vector distribution and collection health."""

    def __init__(self, vectorstore: BaseVectorStore | None = None):
        """Initialize analyzer with optional vectorstore.

        Args:
            vectorstore: Vector store instance. Defaults to create_vectorstore().
        """
        self.vectorstore = vectorstore or create_vectorstore()

    async def get_distribution(self) -> VectorDistribution:
        """Analyze vector distribution and detect outliers.

        For ChromaDB: retrieves stored embeddings and computes L2 norm statistics.
        For Milvus: returns count-based stats (Milvus does not expose stored embeddings
        via the standard query API, so norm analysis is not available).

        Returns:
            VectorDistribution with statistics and outlier detection.
        """
        # ChromaDB path: retrieve embeddings directly
        if hasattr(self.vectorstore, "_collection"):
            collection = self.vectorstore._collection

            results = await asyncio.to_thread(
                collection.get,
                include=["embeddings", "metadatas"]
            )

            embeddings = results.get("embeddings", [])
            ids = results.get("ids", [])

            if not embeddings:
                return VectorDistribution(
                    total_vectors=0,
                    dimension=0,
                    norm_stats={"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
                    outlier_count=0,
                    outlier_ids=[],
                    analyzed_at=datetime.now(timezone.utc).isoformat()
                )

            dimension = len(embeddings[0])

            if HAS_NUMPY:
                norms = self._calculate_norms_numpy(embeddings)
            else:
                norms = self._calculate_norms_python(embeddings)

            norm_min = min(norms)
            norm_max = max(norms)
            norm_mean = sum(norms) / len(norms)

            if HAS_NUMPY:
                norm_std = float(np.std(norms))
            else:
                variance = sum((x - norm_mean) ** 2 for x in norms) / len(norms)
                norm_std = math.sqrt(variance)

            upper_threshold = norm_mean + 2 * norm_std
            lower_threshold = norm_mean - 2 * norm_std

            outlier_ids = []
            for i, norm in enumerate(norms):
                if norm > upper_threshold or norm < lower_threshold:
                    outlier_ids.append(ids[i])

            return VectorDistribution(
                total_vectors=len(embeddings),
                dimension=dimension,
                norm_stats={
                    "min": float(norm_min),
                    "max": float(norm_max),
                    "mean": float(norm_mean),
                    "std": float(norm_std)
                },
                outlier_count=len(outlier_ids),
                outlier_ids=outlier_ids,
                analyzed_at=datetime.now(timezone.utc).isoformat()
            )

        # Milvus path: embeddings are not returned by the query API.
        # Return count-based summary; norm analysis is not available.
        count = await self.vectorstore.count()
        # Try to get dimension from store metadata if available
        dimension = getattr(self.vectorstore, "dimension", 1024)
        return VectorDistribution(
            total_vectors=count,
            dimension=dimension,
            norm_stats={"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            outlier_count=0,
            outlier_ids=[],
            analyzed_at=datetime.now(timezone.utc).isoformat()
        )

    async def get_collection_health(self) -> CollectionHealth:
        """Get collection health metrics.

        Supports both ChromaDB and Milvus vectorstores.

        Returns:
            CollectionHealth with size and metadata info.
        """
        count = await self.vectorstore.count()

        # ChromaDB path
        if hasattr(self.vectorstore, "_collection"):
            collection = self.vectorstore._collection
            metadata = await asyncio.to_thread(lambda: collection.metadata)

            # Estimate dimension from a sample vector
            dimension = 1024  # Default for bge-m3
            if count > 0:
                sample = await asyncio.to_thread(
                    collection.get,
                    limit=1,
                    include=["embeddings"]
                )
                if sample.get("embeddings"):
                    dimension = len(sample["embeddings"][0])

            estimated_bytes = count * dimension * 4  # float32
            estimated_size_mb = estimated_bytes / (1024 * 1024)
            index_type = metadata.get("hnsw:space", "cosine")

            return CollectionHealth(
                total_vectors=count,
                collection_name=collection.name,
                index_type=index_type,
                estimated_size_mb=round(estimated_size_mb, 2),
                metadata=metadata
            )

        # Milvus path
        dimension = getattr(self.vectorstore, "dimension", 1024)
        metric_type = getattr(self.vectorstore, "metric_type", "COSINE")
        collection_name = getattr(self.vectorstore, "collection_name", "knowledge_base")
        uri = getattr(self.vectorstore, "uri", "")

        estimated_bytes = count * dimension * 4  # float32
        estimated_size_mb = estimated_bytes / (1024 * 1024)

        # Try to get richer info via get_collection_info() if available
        store_metadata: dict = {"uri": uri}
        if hasattr(self.vectorstore, "get_collection_info"):
            try:
                info = await asyncio.to_thread(self.vectorstore.get_collection_info)
                store_metadata.update(info)
            except Exception:
                pass

        return CollectionHealth(
            total_vectors=count,
            collection_name=collection_name,
            index_type=metric_type,
            estimated_size_mb=round(estimated_size_mb, 2),
            metadata=store_metadata
        )

    def _calculate_norms_numpy(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate L2 norms using numpy (fast)."""
        arr = np.array(embeddings)
        norms = np.linalg.norm(arr, axis=1)
        return norms.tolist()

    def _calculate_norms_python(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate L2 norms using pure Python (fallback)."""
        norms = []
        for vec in embeddings:
            norm = math.sqrt(sum(x * x for x in vec))
            norms.append(norm)
        return norms
