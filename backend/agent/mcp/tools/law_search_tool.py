"""법령 검색 도구 — 국가법령정보센터 OpenAPI."""

import logging

import httpx

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)

logger = logging.getLogger(__name__)

LAW_API_BASE = "https://www.law.go.kr/DRF/lawSearch.do"


class LawSearchTool(BaseTool):
    """법령 검색 도구 — 국가법령정보센터 OpenAPI"""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="law_search",
            description="국가법령정보센터에서 법령을 검색합니다.",
            category="search",
            help_text=(
                "국가법령정보센터에서 법령을 검색합니다. "
                "법률, 대통령령, 시행규칙 등을 조회할 수 있습니다.\n"
                "파라미터:\n"
                "  - query: 검색 키워드 (필수)\n"
                "  - law_type: 법령 유형 필터 (기본값: all)\n"
                "  - limit: 반환할 최대 결과 수 (기본값: 5)\n"
                "반환: 법령명, 공포일, 법령 ID, 법령 유형"
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParamType.STRING,
                    description="검색 키워드 (예: 가스안전관리법, 근로기준법)",
                ),
                ToolParameter(
                    name="law_type",
                    type=ToolParamType.STRING,
                    description="법령 유형 필터",
                    required=False,
                    default="all",
                    enum=["all", "법률", "대통령령", "총리령", "부령", "조례"],
                ),
                ToolParameter(
                    name="limit",
                    type=ToolParamType.INTEGER,
                    description="반환할 최대 결과 수",
                    required=False,
                    default=5,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="query 파라미터가 필요합니다.")

        law_type = kwargs.get("law_type", "all")
        limit = kwargs.get("limit", 5)

        try:
            params = {
                "OC": "test",
                "target": "law",
                "type": "JSON",
                "query": query,
                "display": str(limit),
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(LAW_API_BASE, params=params)
                response.raise_for_status()

            data = response.json()

            # Parse response — API returns laws under various structures
            laws_raw = []
            if isinstance(data, dict):
                # Common structures: {"LawSearch": {"law": [...]}} or {"법령": [...]}
                if "LawSearch" in data:
                    inner = data["LawSearch"]
                    if isinstance(inner, dict) and "law" in inner:
                        raw = inner["law"]
                        laws_raw = raw if isinstance(raw, list) else [raw]
                    elif isinstance(inner, list):
                        laws_raw = inner
                elif "law" in data:
                    raw = data["law"]
                    laws_raw = raw if isinstance(raw, list) else [raw]
                else:
                    # Try to find any list in the response
                    for v in data.values():
                        if isinstance(v, list):
                            laws_raw = v
                            break
                        if isinstance(v, dict):
                            for vv in v.values():
                                if isinstance(vv, list):
                                    laws_raw = vv
                                    break
            elif isinstance(data, list):
                laws_raw = data

            # Extract relevant fields from each law entry
            laws = []
            for item in laws_raw:
                if not isinstance(item, dict):
                    continue

                law_entry = {
                    "법령명": item.get("법령명한글") or item.get("법령명") or item.get("lawNameKorean") or item.get("법령명약칭") or "",
                    "공포일자": item.get("공포일자") or item.get("시행일자") or item.get("promulgationDate") or "",
                    "법령ID": item.get("법령일련번호") or item.get("법령ID") or item.get("lawId") or item.get("MST") or "",
                    "법령유형": item.get("법령구분명") or item.get("법령종류") or item.get("lawType") or "",
                }

                # Filter by law_type if specified
                if law_type != "all" and law_entry["법령유형"] and law_type not in law_entry["법령유형"]:
                    continue

                laws.append(law_entry)

            if not laws:
                return ToolResult(
                    success=True,
                    data={
                        "laws": [],
                        "total": 0,
                        "message": f"'{query}'에 대한 검색 결과가 없습니다.",
                    },
                    metadata={"query": query, "law_type": law_type},
                )

            return ToolResult(
                success=True,
                data={"laws": laws[:limit], "total": len(laws)},
                metadata={"query": query, "law_type": law_type},
            )

        except httpx.HTTPStatusError as e:
            logger.error("법령 검색 API HTTP 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"국가법령정보센터 API 오류 (HTTP {e.response.status_code}). 검색어: '{query}'",
            )
        except httpx.RequestError as e:
            logger.error("법령 검색 API 요청 실패: %s", e)
            return ToolResult(
                success=False,
                error=f"국가법령정보센터 API에 연결할 수 없습니다. 검색어: '{query}'. 네트워크 상태를 확인하세요.",
            )
        except Exception as e:
            logger.error("법령 검색 중 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"법령 검색 중 오류가 발생했습니다: {e}. 검색어: '{query}'",
            )
