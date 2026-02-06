"""Document search tool for RAG-based retrieval."""

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from rag.retriever import HybridRetriever


class DocumentSearchTool(BaseTool):
    """Search documents in the knowledge base."""

    def __init__(self, retriever: HybridRetriever | None = None):
        self._retriever = retriever

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_documents",
            description="한국가스기술공사 문서 지식베이스에서 관련 문서를 검색합니다.",
            category="retrieval",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParamType.STRING,
                    description="검색 쿼리 (자연어)",
                ),
                ToolParameter(
                    name="top_k",
                    type=ToolParamType.INTEGER,
                    description="반환할 최대 문서 수",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="department",
                    type=ToolParamType.STRING,
                    description="부서 필터 (예: 안전팀, 기술연구소)",
                    required=False,
                ),
                ToolParameter(
                    name="category",
                    type=ToolParamType.STRING,
                    description="문서 유형 필터 (규정, 매뉴얼, 보고서 등)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="query parameter is required")

        top_k = kwargs.get("top_k", 5)
        filters = {}
        if kwargs.get("department"):
            filters["department"] = kwargs["department"]
        if kwargs.get("category"):
            filters["category"] = kwargs["category"]

        try:
            results = await self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters or None,
            )

            documents = [
                {
                    "id": r.id,
                    "content": r.content[:500],  # Truncate for tool output
                    "score": round(r.score, 3),
                    "metadata": r.metadata,
                }
                for r in results
            ]

            return ToolResult(
                success=True,
                data={"documents": documents, "total": len(documents)},
                metadata={"query": query, "filters": filters},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
