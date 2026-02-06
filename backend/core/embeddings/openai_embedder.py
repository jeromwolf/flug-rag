"""OpenAI embedding provider."""

from openai import AsyncOpenAI

from .base import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str = "",
        dimension: int = 1024,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed_text(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=self.dimension,
        )
        return response.data[0].embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # OpenAI supports batch in single call (max 2048 inputs)
        batch_size = 2048
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimension,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        return await self.embed_text(query)

    async def close(self):
        await self._client.close()
