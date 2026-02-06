"""Local embedding using sentence-transformers (bge-m3)."""

import asyncio
import hashlib
import logging
from functools import lru_cache

from .base import BaseEmbedding

logger = logging.getLogger(__name__)

# Embedding cache TTL: 1 hour (embeddings are deterministic for same input)
_EMBEDDING_CACHE_TTL = 3600


class LocalEmbedding(BaseEmbedding):
    """Local embedding via sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        self.dimension = 1024  # bge-m3 default
        self._device = device
        self._model = None

    def _get_model(self):
        """Lazy-load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._device)
            self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @staticmethod
    def _embedding_cache_key(text: str) -> str:
        """Build a cache key for an embedding vector."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embed:local:{text_hash}"

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text with caching (TTL: 3600s)."""
        # Try cache first
        cache_key = self._embedding_cache_key(text)
        try:
            from core.cache import get_cache

            cache = await get_cache()
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached
        except Exception:
            cache = None

        model = self._get_model()
        embedding = await asyncio.to_thread(
            model.encode, text, normalize_embeddings=True
        )
        result = embedding.tolist()

        # Store in cache
        if cache is not None:
            try:
                await cache.set(cache_key, result, ttl=_EMBEDDING_CACHE_TTL)
            except Exception:
                pass

        return result

    async def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts in batch."""
        model = self._get_model()
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query. For bge-m3, prepend instruction."""
        return await self.embed_text(query)
