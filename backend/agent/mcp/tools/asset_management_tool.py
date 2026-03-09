"""
설비/자재 관리 시스템 외부 연동 도구.

실제 외부 REST API(DummyJSON)를 호출하여 설비·자재 목록을 조회합니다.
시연용으로 DummyJSON의 products API를 "설비/자재 관리 시스템"으로 프레이밍합니다.

핵심 포인트: Mock이 아닌 **실제 HTTP API 호출** → 외부 시스템 연동 시연.
"""
import logging
import re
from datetime import datetime

import httpx

from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)

_API_BASE = "https://dummyjson.com"
_TIMEOUT = 10.0

# 월별로 다른 데이터셋을 보여주기 위한 오프셋 + 정렬 전략
# skip 값을 달리해서 월마다 다른 자재 목록이 나오게 함
_MONTH_CONFIG = {
    1:  {"skip": 0,  "sortBy": "price", "order": "asc",  "label": "2026년 1월"},
    2:  {"skip": 10, "sortBy": "price", "order": "desc", "label": "2026년 2월"},
    3:  {"skip": 20, "sortBy": "rating", "order": "desc", "label": "2026년 3월"},
    4:  {"skip": 30, "sortBy": "stock", "order": "desc",  "label": "2026년 4월"},
    5:  {"skip": 40, "sortBy": "title", "order": "asc",   "label": "2026년 5월"},
    6:  {"skip": 50, "sortBy": "price", "order": "asc",   "label": "2026년 6월"},
    7:  {"skip": 60, "sortBy": "rating", "order": "asc",  "label": "2026년 7월"},
    8:  {"skip": 70, "sortBy": "stock", "order": "asc",   "label": "2026년 8월"},
    9:  {"skip": 80, "sortBy": "price", "order": "desc",  "label": "2026년 9월"},
    10: {"skip": 90, "sortBy": "title", "order": "desc",  "label": "2026년 10월"},
    11: {"skip": 100, "sortBy": "rating", "order": "desc","label": "2026년 11월"},
    12: {"skip": 110, "sortBy": "stock", "order": "desc", "label": "2026년 12월"},
}

_MONTH_RE = re.compile(r"(\d{1,2})\s*월")


def _parse_month(period: str) -> int | None:
    """'3월', '2026년 3월' 등에서 월 숫자를 추출."""
    if not period:
        return None
    m = _MONTH_RE.search(period)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return month
    return None


def _translate_product(item: dict, period_label: str = "") -> dict:
    """DummyJSON product → 설비/자재 한글 형식으로 변환."""
    result = {
        "자재코드": item.get("sku", item.get("id", "")),
        "자재명": item.get("title", ""),
        "제조사": item.get("brand", ""),
        "분류": item.get("category", ""),
        "단가(USD)": item.get("price", 0),
        "재고수량": item.get("stock", 0),
        "평점": item.get("rating", 0),
        "최소주문수량": item.get("minimumOrderQuantity", 1),
        "보증기간": item.get("warrantyInformation", ""),
        "배송정보": item.get("shippingInformation", ""),
        "재고상태": item.get("availabilityStatus", ""),
    }
    if period_label:
        result["조회기간"] = period_label
    return result


class AssetManagementTool(BaseTool):
    """설비/자재 관리 시스템 연동 도구 — 실제 외부 API 호출."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="asset_management",
            description=(
                "설비/자재 관리 시스템에서 설비·자재 목록을 조회합니다. "
                "실제 외부 REST API를 호출하여 실시간 데이터를 반환합니다."
            ),
            category="integration",
            help_text=(
                "설비/자재 관리 시스템 외부 연동 도구입니다.\n\n"
                "action 종류:\n"
                "  - search : 키워드로 자재/설비 검색\n"
                "  - list   : 전체 자재 목록 조회 (상위 10건)\n"
                "  - detail : 특정 자재 상세 정보 조회 (ID 필요)\n\n"
                "keyword 예시:\n"
                "  search → '배관', '밸브', '센서', 'phone'\n"
                "  detail → '1', '5' (자재 ID)\n\n"
                "이 도구는 외부 설비/자재 관리 API에 실시간 연동됩니다."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type=ToolParamType.STRING,
                    description="조회 유형 (search: 키워드 검색, list: 목록 조회, detail: 상세 조회)",
                    required=True,
                    enum=["search", "list", "detail"],
                ),
                ToolParameter(
                    name="keyword",
                    type=ToolParamType.STRING,
                    description="검색 키워드 또는 자재 ID. search 시 검색어, detail 시 자재 ID",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="period",
                    type=ToolParamType.STRING,
                    description="조회 기간 (예: '1월', '3월', '2026년 2월'). 미입력 시 현재 월 기준",
                    required=False,
                    default="",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        action: str = kwargs.get("action", "list").strip().lower()
        keyword: str = kwargs.get("keyword", "").strip()
        period: str = kwargs.get("period", "").strip()

        if action not in ("search", "list", "detail"):
            return ToolResult(
                success=False,
                error="action은 search / list / detail 중 하나여야 합니다.",
            )

        # 월 파싱: 미입력 시 현재 월
        month = _parse_month(period)
        if month is None:
            month = datetime.now().month
        month_cfg = _MONTH_CONFIG[month]

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                if action == "search":
                    return await self._search(client, keyword, month_cfg)
                elif action == "detail":
                    return await self._detail(client, keyword, month_cfg)
                else:
                    return await self._list(client, month_cfg)

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="설비/자재 관리 시스템 연결 시간 초과 (10초). 네트워크를 확인해 주세요.",
            )
        except Exception as e:
            logger.error("AssetManagementTool 실행 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"설비/자재 관리 시스템 조회 중 오류: {e}",
            )

    async def _search(self, client: httpx.AsyncClient, keyword: str, month_cfg: dict) -> ToolResult:
        """키워드로 자재/설비 검색."""
        if not keyword:
            return await self._list(client, month_cfg)

        label = month_cfg["label"]
        params = {
            "q": keyword,
            "limit": 10,
            "sortBy": month_cfg["sortBy"],
            "order": month_cfg["order"],
        }
        resp = await client.get(f"{_API_BASE}/products/search", params=params)
        resp.raise_for_status()
        data = resp.json()

        products = [_translate_product(p, label) for p in data.get("products", [])]

        return ToolResult(
            success=True,
            data={
                "action": "search",
                "keyword": keyword,
                "조회기간": label,
                "total": data.get("total", 0),
                "반환건수": len(products),
                "result": products,
            },
            metadata={
                "source": "설비/자재 관리 시스템 (외부 API)",
                "api_endpoint": f"{_API_BASE}/products/search?q={keyword}",
                "period": label,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": resp.elapsed.total_seconds() * 1000,
            },
        )

    async def _list(self, client: httpx.AsyncClient, month_cfg: dict) -> ToolResult:
        """전체 자재 목록 조회 (상위 10건). 월별로 다른 데이터셋."""
        label = month_cfg["label"]
        params = {
            "limit": 10,
            "skip": month_cfg["skip"],
            "sortBy": month_cfg["sortBy"],
            "order": month_cfg["order"],
        }
        resp = await client.get(f"{_API_BASE}/products", params=params)
        resp.raise_for_status()
        data = resp.json()

        products = [_translate_product(p, label) for p in data.get("products", [])]

        return ToolResult(
            success=True,
            data={
                "action": "list",
                "조회기간": label,
                "total": data.get("total", 0),
                "반환건수": len(products),
                "result": products,
            },
            metadata={
                "source": "설비/자재 관리 시스템 (외부 API)",
                "api_endpoint": f"{_API_BASE}/products?limit=10&skip={month_cfg['skip']}",
                "period": label,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": resp.elapsed.total_seconds() * 1000,
            },
        )

    async def _detail(self, client: httpx.AsyncClient, item_id: str, month_cfg: dict) -> ToolResult:
        """자재 상세 정보 조회."""
        if not item_id:
            return ToolResult(
                success=False,
                error="상세 조회를 위해 자재 ID(keyword)를 입력해 주세요.",
            )

        label = month_cfg["label"]
        resp = await client.get(f"{_API_BASE}/products/{item_id}")
        resp.raise_for_status()
        product = resp.json()

        translated = _translate_product(product, label)
        translated["설명"] = product.get("description", "")
        translated["반환/교환정책"] = product.get("returnPolicy", "")
        translated["이미지수"] = len(product.get("images", []))

        return ToolResult(
            success=True,
            data={
                "action": "detail",
                "item_id": item_id,
                "조회기간": label,
                "result": translated,
            },
            metadata={
                "source": "설비/자재 관리 시스템 (외부 API)",
                "api_endpoint": f"{_API_BASE}/products/{item_id}",
                "period": label,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": resp.elapsed.total_seconds() * 1000,
            },
        )
