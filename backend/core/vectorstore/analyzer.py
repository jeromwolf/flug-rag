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

        Returns:
            VectorDistribution with statistics and outlier detection.
        """
        # Access ChromaDB collection (assuming ChromaStore)
        if not hasattr(self.vectorstore, "_collection"):
            raise ValueError("VectorAnalyzer currently supports ChromaStore only")

        collection = self.vectorstore._collection

        # Get all embeddings and metadata
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

        # Calculate L2 norms
        if HAS_NUMPY:
            norms = self._calculate_norms_numpy(embeddings)
        else:
            norms = self._calculate_norms_python(embeddings)

        # Calculate norm statistics
        norm_min = min(norms)
        norm_max = max(norms)
        norm_mean = sum(norms) / len(norms)

        # Standard deviation
        if HAS_NUMPY:
            norm_std = float(np.std(norms))
        else:
            variance = sum((x - norm_mean) ** 2 for x in norms) / len(norms)
            norm_std = math.sqrt(variance)

        # Detect outliers (vectors with norm > mean + 2*std or < mean - 2*std)
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

    async def get_collection_health(self) -> CollectionHealth:
        """Get collection health metrics.

        Returns:
            CollectionHealth with size and metadata info.
        """
        if not hasattr(self.vectorstore, "_collection"):
            raise ValueError("VectorAnalyzer currently supports ChromaStore only")

        collection = self.vectorstore._collection

        # Get collection info
        count = await self.vectorstore.count()
        metadata = await asyncio.to_thread(lambda: collection.metadata)

        # Estimate storage size (float32 = 4 bytes per dimension)
        # Get dimension from a sample vector
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

        # Extract index type from metadata
        index_type = metadata.get("hnsw:space", "unknown")

        return CollectionHealth(
            total_vectors=count,
            collection_name=collection.name,
            index_type=index_type,
            estimated_size_mb=round(estimated_size_mb, 2),
            metadata=metadata
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
