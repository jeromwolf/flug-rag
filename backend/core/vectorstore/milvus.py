"""Milvus vector store implementation using MilvusClient API.

Supports both Milvus Lite (embedded, local file) and Milvus Standalone (server).
- Lite: MilvusClient(uri="./data/milvus.db")
- Standalone: MilvusClient(uri="http://localhost:19530", token="...")
"""

import asyncio
import json
import logging
from typing import Any

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from .base import BaseVectorStore, SearchResult

logger = logging.getLogger(__name__)


class MilvusStore(BaseVectorStore):
    """Milvus vector store supporting Lite (embedded) and Standalone modes."""

    def __init__(
        self,
        uri: str = "./data/milvus.db",
        token: str = "",
        collection_name: str = "knowledge_base",
        dimension: int = 1024,
        metric_type: str = "COSINE",
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type

        # Connect using MilvusClient (works for both Lite and Standalone)
        self.client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)

        # Create collection if not exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection with schema if it doesn't exist."""
        if self.client.has_collection(self.collection_name):
            return

        # Define schema with source_type as first-class filterable field
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
        ])

        # Create index params
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Created Milvus collection: %s (dim=%d)", self.collection_name, self.dimension)

    async def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents with embeddings to Milvus."""
        if not metadatas:
            metadatas = [{} for _ in ids]

        # Build data list
        data = []
        for i in range(len(ids)):
            meta = metadatas[i] if metadatas else {}
            source_type = meta.pop("source_type", "") if meta else ""
            data.append({
                "id": ids[i],
                "content": texts[i],
                "embedding": embeddings[i],
                "source_type": str(source_type) if source_type else "",
                "metadata_json": json.dumps(meta, ensure_ascii=False),
            })

        await asyncio.to_thread(
            self.client.insert,
            collection_name=self.collection_name,
            data=data,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        filter_expr = self._build_filter_expr(filters) if filters else ""

        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "content", "source_type", "metadata_json"],
            filter=filter_expr,
        )

        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                metadata = json.loads(entity.get("metadata_json", "{}"))
                # Restore source_type into metadata for compatibility
                st = entity.get("source_type", "")
                if st:
                    metadata["source_type"] = st

                search_results.append(
                    SearchResult(
                        id=entity.get("id", hit.get("id", "")),
                        content=entity.get("content", ""),
                        score=hit.get("distance", 0.0),
                        metadata=metadata,
                    )
                )

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.collection_name,
            ids=ids,
        )

    async def get(self, ids: list[str]) -> list[dict]:
        """Get documents by IDs."""
        results = await asyncio.to_thread(
            self.client.get,
            collection_name=self.collection_name,
            ids=ids,
            output_fields=["id", "content", "source_type", "metadata_json"],
        )

        items = []
        for result in results:
            metadata = json.loads(result.get("metadata_json", "{}"))
            st = result.get("source_type", "")
            if st:
                metadata["source_type"] = st
            items.append({
                "id": result.get("id"),
                "content": result.get("content"),
                "metadata": metadata,
            })
        return items

    async def count(self) -> int:
        """Get total document count."""
        result = await asyncio.to_thread(
            self.client.query,
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if result and len(result) > 0:
            return result[0].get("count(*)", 0)
        return 0

    async def clear(self) -> None:
        """Clear all documents in collection."""
        await asyncio.to_thread(self.client.drop_collection, collection_name=self.collection_name)
        await asyncio.to_thread(self._ensure_collection)

    async def get_all_documents(self) -> list[dict]:
        """Get all documents for BM25 index building."""
        items = []
        # Use iterator for large collections
        batch_size = 1000
        offset = 0

        while True:
            batch = await asyncio.to_thread(
                self.client.query,
                collection_name=self.collection_name,
                filter="",
                output_fields=["id", "content", "source_type", "metadata_json"],
                limit=batch_size,
                offset=offset,
            )

            if not batch:
                break

            for doc in batch:
                metadata = json.loads(doc.get("metadata_json", "{}"))
                st = doc.get("source_type", "")
                if st:
                    metadata["source_type"] = st
                items.append({
                    "id": doc.get("id"),
                    "content": doc.get("content"),
                    "metadata": metadata,
                })

            if len(batch) < batch_size:
                break
            offset += batch_size

        return items

    def _build_filter_expr(self, filters: dict) -> str:
        """Build Milvus filter expression from dict.

        source_type is a first-class field and can be filtered directly.
        Other metadata keys stored in metadata_json cannot be filtered.
        """
        if not filters:
            return ""

        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            # source_type is a first-class field
            if key == "source_type":
                if isinstance(value, list):
                    escaped = [v.replace('"', '\\"') for v in value]
                    val_str = ", ".join(f'"{v}"' for v in escaped)
                    conditions.append(f"source_type in [{val_str}]")
                else:
                    escaped = str(value).replace('"', '\\"')
                    conditions.append(f'source_type == "{escaped}"')

        return " and ".join(conditions) if conditions else ""

    def get_collection_info(self) -> dict:
        """Get collection metadata and stats."""
        info = self.client.get_collection_stats(self.collection_name)
        return {
            "name": self.collection_name,
            "count": info.get("row_count", 0),
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "uri": self.uri,
        }
