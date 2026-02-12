"""ChromaDB vector store implementation."""

import asyncio
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .base import BaseVectorStore, SearchResult


class ChromaStore(BaseVectorStore):
    """ChromaDB persistent vector store."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "knowledge_base",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Ensure directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": f"flux-rag {collection_name}",
                "hnsw:space": "cosine",
            },
        )

    async def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents with embeddings to ChromaDB."""
        # ChromaDB is sync, run in thread
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search by cosine similarity."""
        where = self._build_where_filter(filters) if filters else None

        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances (lower = closer for cosine)
                # Convert to similarity score (1 - distance)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 - distance

                search_results.append(
                    SearchResult(
                        id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        score=score,
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    )
                )

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        await asyncio.to_thread(self._collection.delete, ids=ids)

    async def get(self, ids: list[str]) -> list[dict]:
        """Get documents by IDs."""
        results = await asyncio.to_thread(
            self._collection.get,
            ids=ids,
            include=["documents", "metadatas"],
        )
        items = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                items.append({
                    "id": doc_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return items

    async def count(self) -> int:
        """Get total document count."""
        return await asyncio.to_thread(self._collection.count)

    async def clear(self) -> None:
        """Clear all documents in collection."""
        # Recreate collection
        await asyncio.to_thread(self._client.delete_collection, self.collection_name)
        self._collection = await asyncio.to_thread(
            self._client.create_collection,
            name=self.collection_name,
            metadata={
                "description": f"flux-rag {self.collection_name}",
                "hnsw:space": "cosine",
            },
        )

    def _build_where_filter(self, filters: dict) -> dict | None:
        """Build ChromaDB where filter from dict.

        If the filter already contains ChromaDB operators ($and, $or, $not),
        pass through as-is (e.g. from access control merged filters).
        """
        if not filters:
            return None

        # If any top-level key is a ChromaDB operator, the filter is pre-built
        if any(key.startswith("$") for key in filters):
            return filters

        conditions = []
        for key, value in filters.items():
            if value is not None:
                if isinstance(value, list):
                    conditions.append({key: {"$in": value}})
                else:
                    conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    async def get_all_documents(self) -> list[dict]:
        """Get all documents for BM25 index building."""
        all_docs = await asyncio.to_thread(
            self._collection.get,
            include=["documents", "metadatas"],
        )
        items = []
        if all_docs["ids"]:
            for i, doc_id in enumerate(all_docs["ids"]):
                items.append({
                    "id": doc_id,
                    "content": all_docs["documents"][i] if all_docs["documents"] else "",
                    "metadata": all_docs["metadatas"][i] if all_docs["metadatas"] else {},
                })
        return items

    def get_collection_info(self) -> dict:
        """Get collection metadata and stats."""
        return {
            "name": self.collection_name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata,
        }
