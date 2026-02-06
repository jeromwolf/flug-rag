"""Milvus vector store implementation."""

import asyncio
import json
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from .base import BaseVectorStore, SearchResult


class MilvusStore(BaseVectorStore):
    """Milvus distributed vector store for production."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "knowledge_base",
        dimension: int = 1024,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        nlist: int = 128,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.nlist = nlist

        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port,
        )

        # Create collection if not exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection with schema if it doesn't exist."""
        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            self._collection.load()
        else:
            # Define schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=256,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dimension,
                ),
                FieldSchema(
                    name="metadata_json",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description=f"flux-rag {self.collection_name}",
            )

            # Create collection
            self._collection = Collection(
                name=self.collection_name,
                schema=schema,
            )

            # Create index on embedding field
            index_params = {
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"nlist": self.nlist},
            }
            self._collection.create_index(
                field_name="embedding",
                index_params=index_params,
            )

            # Load collection to memory
            self._collection.load()

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

        # Serialize metadata to JSON strings
        metadata_jsons = [json.dumps(meta) for meta in metadatas]

        # Prepare data for insertion
        entities = [
            ids,
            texts,
            embeddings,
            metadata_jsons,
        ]

        # Insert data (sync operation, run in thread)
        await asyncio.to_thread(
            self._collection.insert,
            entities,
        )

        # Flush to persist data
        await asyncio.to_thread(self._collection.flush)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search by cosine similarity."""
        search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": 10},
        }

        # Build filter expression
        expr = self._build_filter_expr(filters) if filters else None

        # Perform search (sync operation, run in thread)
        results = await asyncio.to_thread(
            self._collection.search,
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "content", "metadata_json"],
        )

        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                # Milvus returns similarity score directly for COSINE
                search_results.append(
                    SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.score,
                        metadata=json.loads(hit.entity.get("metadata_json", "{}")),
                    )
                )

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        expr = f"id in {ids}"
        await asyncio.to_thread(self._collection.delete, expr)
        await asyncio.to_thread(self._collection.flush)

    async def get(self, ids: list[str]) -> list[dict]:
        """Get documents by IDs."""
        expr = f"id in {ids}"
        results = await asyncio.to_thread(
            self._collection.query,
            expr=expr,
            output_fields=["id", "content", "metadata_json"],
        )

        items = []
        for result in results:
            items.append({
                "id": result.get("id"),
                "content": result.get("content"),
                "metadata": json.loads(result.get("metadata_json", "{}")),
            })
        return items

    async def count(self) -> int:
        """Get total document count."""
        return await asyncio.to_thread(self._collection.num_entities)

    async def clear(self) -> None:
        """Clear all documents in collection."""
        # Drop and recreate collection
        await asyncio.to_thread(utility.drop_collection, self.collection_name)
        await asyncio.to_thread(self._ensure_collection)

    def _build_filter_expr(self, filters: dict) -> str | None:
        """Build Milvus filter expression from dict.

        Note: Filters on metadata fields require the metadata_json field
        to be parsed. For production, consider adding dedicated fields
        for commonly filtered metadata keys.
        """
        conditions = []
        for key, value in filters.items():
            if value is not None:
                if isinstance(value, list):
                    # For list values, use 'in' operator
                    value_str = str(value).replace("'", '"')
                    conditions.append(f'{key} in {value_str}')
                elif isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return " and ".join(conditions)

    def get_collection_info(self) -> dict:
        """Get collection metadata and stats."""
        return {
            "name": self.collection_name,
            "count": self._collection.num_entities,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
        }
