"""Database tool for managing knowledge base documents."""

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from core.vectorstore import BaseVectorStore, create_vectorstore


class KnowledgeBaseTool(BaseTool):
    """Manage knowledge base: count, list, get document info."""

    def __init__(self, vectorstore: BaseVectorStore | None = None):
        self._vectorstore = vectorstore

    @property
    def vectorstore(self) -> BaseVectorStore:
        if self._vectorstore is None:
            self._vectorstore = create_vectorstore()
        return self._vectorstore

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="knowledge_base",
            description="지식베이스 관리: 문서 수 확인, 문서 정보 조회",
            category="management",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ToolParamType.STRING,
                    description="수행할 작업",
                    enum=["count", "get", "info"],
                ),
                ToolParameter(
                    name="document_ids",
                    type=ToolParamType.ARRAY,
                    description="문서 ID 목록 (get 액션에 필요)",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "count")

        try:
            if action == "count":
                count = await self.vectorstore.count()
                return ToolResult(success=True, data={"count": count})

            elif action == "get":
                doc_ids = kwargs.get("document_ids", [])
                if not doc_ids:
                    return ToolResult(success=False, error="document_ids required for 'get' action")
                items = await self.vectorstore.get(ids=doc_ids)
                return ToolResult(success=True, data={"documents": items})

            elif action == "info":
                # Return collection metadata if available
                if hasattr(self.vectorstore, "get_collection_info"):
                    info = self.vectorstore.get_collection_info()
                    return ToolResult(success=True, data=info)
                count = await self.vectorstore.count()
                return ToolResult(success=True, data={"count": count})

            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")

        except Exception as e:
            return ToolResult(success=False, error=str(e))
