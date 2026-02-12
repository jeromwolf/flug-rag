"""Query expansion techniques for RAG retrieval.

Implements HyDE (Hypothetical Document Embeddings) and other expansion methods.
"""

import logging

from core.embeddings import BaseEmbedding, create_embedder
from core.llm import BaseLLM, create_llm
from rag.prompt import PromptManager

logger = logging.getLogger(__name__)


class QueryExpander:
    """Query expansion using HyDE and other techniques."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        embedder: BaseEmbedding | None = None,
        prompt_manager: PromptManager | None = None,
    ):
        self._llm = llm
        self._embedder = embedder
        self._prompt_manager = prompt_manager

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    @property
    def embedder(self) -> BaseEmbedding:
        if self._embedder is None:
            self._embedder = create_embedder()
        return self._embedder

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager()
        return self._prompt_manager

    async def expand_hyde(self, query: str) -> list[float]:
        """Generate a hypothetical document embedding for the query.

        HyDE Process:
        1. Ask LLM to generate a hypothetical answer
        2. Embed the hypothetical answer
        3. Return the embedding vector for retrieval

        Args:
            query: User's original question.

        Returns:
            Embedding vector of the hypothetical document.
        """
        # Step 1: Generate hypothetical answer
        prompt = self.prompt_manager.get_system_prompt("hyde_system").format(query=query)
        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
        )
        hypothetical_doc = response.content.strip()
        logger.debug("HyDE hypothetical doc: %s", hypothetical_doc[:100])

        # Step 2: Embed the hypothetical answer
        embedding = await self.embedder.embed_query(hypothetical_doc)
        return embedding

    async def expand_hyde_with_original(self, query: str) -> list[float]:
        """Generate HyDE embedding averaged with original query embedding.

        Combines the hypothetical document embedding with the original query
        embedding to balance between query intent and document similarity.

        Args:
            query: User's original question.

        Returns:
            Averaged embedding vector.
        """
        import numpy as np

        # Get both embeddings in parallel
        import asyncio
        hyde_emb_task = self.expand_hyde(query)
        orig_emb_task = self.embedder.embed_query(query)
        hyde_emb, orig_emb = await asyncio.gather(hyde_emb_task, orig_emb_task)

        # Average the two embeddings
        combined = (np.array(hyde_emb) + np.array(orig_emb)) / 2.0
        # Re-normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined.tolist()
