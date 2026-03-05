"""
ERP 시스템 연동 Mock 도구.

한국가스기술공사 ERP 시스템(예산/프로젝트/협력업체 정보)을 조회하는 모의(Mock) 도구.
실제 ERP 서버에 연결하지 않으며, 데모용 더미 데이터를 반환합니다.
"""
import logging
from datetime import datetime

from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Mock 데이터 상수
# ──────────────────────────────────────────────────────────────────────────────

_BUDGET_DATA = {
    "default": {
        "회계연도": "2026년",
        "총예산": "1,240억 원",
        "집행액": "687억 원",
        "잔액": "553억 원",
        "집행률": "55.4%",
        "부문별 현황": [
            {"부문": "시설유지보수", "예산": "380억 원", "집행": "210억 원", "집행률": "55.3%"},
            {"부문": "안전관리",     "예산": "290억 원", "집행": "175억 원", "집행률": "60.3%"},
            {"부문": "인건비",       "예산": "320억 원", "집행": "189억 원", "집행률": "59.1%"},
            {"부문": "R&D",          "예산": "140억 원", "집행":  "63억 원", "집행률": "45.0%"},
            {"부문": "기타운영비",   "예산": "110억 원", "집행":  "50억 원", "집행률": "45.5%"},
        ],
        "조회일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "비고": "2026년 1월 기준 잠정 집계 (확정치는 재무팀 확인 필요)",
    },
    "안전": {
        "회계연도": "2026년",
        "부문": "안전관리",
        "예산": "290억 원",
        "집행액": "175억 원",
        "잔액": "115억 원",
        "집행률": "60.3%",
        "세부항목": [
            {"항목": "정기검사 및 진단",     "예산": "85억 원",  "집행": "52억 원"},
            {"항목": "안전설비 교체",         "예산": "70억 원",  "집행": "43억 원"},
            {"항목": "안전교육 및 훈련",      "예산": "40억 원",  "집행": "26억 원"},
            {"항목": "비상대응체계 운영",     "예산": "55억 원",  "집행": "34억 원"},
            {"항목": "안전인증·외부감사",     "예산": "40억 원",  "집행": "20억 원"},
        ],
        "조회일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
    },
}

_PROJECT_DATA = {
    "default": [
        {
            "프로젝트코드": "PRJ-2026-001",
            "프로젝트명": "노후 배관 교체 1공구 (경기북부)",
            "발주처": "한국가스공사",
            "계약금액": "48억 원",
            "진행률": "62%",
            "착공일": "2026-01-15",
            "준공예정일": "2026-09-30",
            "담당PM": "김민준 부장",
            "상태": "진행중",
        },
        {
            "프로젝트코드": "PRJ-2026-002",
            "프로젝트명": "정압기 원격감시 시스템 고도화",
            "발주처": "한국가스기술공사 자체",
            "계약금액": "12억 원",
            "진행률": "35%",
            "착공일": "2026-02-01",
            "준공예정일": "2026-11-30",
            "담당PM": "이서연 차장",
            "상태": "진행중",
        },
        {
            "프로젝트코드": "PRJ-2025-047",
            "프로젝트명": "공급관리소 전기설비 정비",
            "발주처": "서울에너지공사",
            "계약금액": "6억 5천만 원",
            "진행률": "100%",
            "착공일": "2025-10-01",
            "준공예정일": "2026-01-31",
            "담당PM": "박지훈 과장",
            "상태": "완료",
        },
        {
            "프로젝트코드": "PRJ-2026-003",
            "프로젝트명": "LNG 저장탱크 안전진단 (인천기지)",
            "발주처": "한국가스공사 인천기지",
            "계약금액": "9억 원",
            "진행률": "10%",
            "착공일": "2026-03-01",
            "준공예정일": "2026-07-31",
            "담당PM": "최유나 대리",
            "상태": "착수준비",
        },
    ],
}

_VENDOR_DATA = {
    "default": [
        {
            "업체코드": "VND-001",
            "업체명": "한국강관(주)",
            "분류": "자재공급",
            "등록상태": "정상",
            "평가등급": "A",
            "최근계약일": "2026-01-10",
            "누적계약액": "340억 원",
            "담당자": "구매팀 이담당",
        },
        {
            "업체코드": "VND-002",
            "업체명": "(주)한진엔지니어링",
            "분류": "시공",
            "등록상태": "정상",
            "평가등급": "A+",
            "최근계약일": "2026-02-05",
            "누적계약액": "520억 원",
            "담당자": "구매팀 박담당",
        },
        {
            "업체코드": "VND-003",
            "업체명": "대화계측(주)",
            "분류": "계측·제어",
            "등록상태": "정상",
            "평가등급": "B+",
            "최근계약일": "2025-11-20",
            "누적계약액": "87억 원",
            "담당자": "구매팀 최담당",
        },
        {
            "업체코드": "VND-004",
            "업체명": "삼일안전기술(주)",
            "분류": "안전진단",
            "등록상태": "거래정지",
            "평가등급": "C",
            "최근계약일": "2024-08-01",
            "누적계약액": "22억 원",
            "담당자": "구매팀 김담당",
        },
    ],
}


def _filter_by_keyword(items: list, keyword: str) -> list:
    """keyword가 포함된 항목만 반환 (대소문자·공백 무시)."""
    if not keyword:
        return items
    kw = keyword.strip().lower()
    return [
        item for item in items
        if any(kw in str(v).lower() for v in item.values())
    ]


class ErpLookupTool(BaseTool):
    """ERP 시스템 Mock 조회 도구 — 예산·프로젝트·협력업체 정보 반환."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="erp_lookup",
            description="ERP 시스템에서 예산, 프로젝트, 협력업체 정보를 조회합니다 (데모용 Mock 데이터)",
            category="integration",
            help_text=(
                "한국가스기술공사 ERP 연동 조회 도구입니다 (데모 Mock).\n\n"
                "query_type 종류:\n"
                "  - budget  : 예산 집행 현황 (부문별 예산·집행액·집행률)\n"
                "  - project : 프로젝트 현황 (프로젝트코드·진행률·PM·상태)\n"
                "  - vendor  : 협력업체 정보 (업체명·분류·평가등급·계약이력)\n\n"
                "keyword 예시:\n"
                "  budget  → '안전', '시설'\n"
                "  project → '배관', '2026-001', '진행중'\n"
                "  vendor  → '한국강관', '안전진단', '거래정지'\n\n"
                "⚠️ 본 도구는 실제 ERP에 연결되지 않으며 데모용 Mock 데이터를 반환합니다."
            ),
            parameters=[
                ToolParameter(
                    name="query_type",
                    type=ToolParamType.STRING,
                    description="조회 유형 (budget: 예산현황, project: 프로젝트현황, vendor: 협력업체정보)",
                    required=True,
                    enum=["budget", "project", "vendor"],
                ),
                ToolParameter(
                    name="keyword",
                    type=ToolParamType.STRING,
                    description="검색 키워드 (예: '안전', '배관', '한국강관'). 미입력 시 전체 조회",
                    required=False,
                    default="",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        query_type: str = kwargs.get("query_type", "").strip().lower()
        keyword: str = kwargs.get("keyword", "").strip()

        if query_type not in ("budget", "project", "vendor"):
            return ToolResult(
                success=False,
                error="query_type은 budget / project / vendor 중 하나여야 합니다.",
            )

        try:
            if query_type == "budget":
                # keyword로 부문 매핑 (간단히 키워드 포함 여부 확인)
                key = "안전" if "안전" in keyword else "default"
                data = _BUDGET_DATA[key]
                return ToolResult(
                    success=True,
                    data={
                        "query_type": "budget",
                        "keyword": keyword or "(전체)",
                        "result": data,
                    },
                    metadata={"source": "ERP Mock", "timestamp": datetime.now().isoformat()},
                )

            elif query_type == "project":
                items = _PROJECT_DATA["default"]
                filtered = _filter_by_keyword(items, keyword)
                return ToolResult(
                    success=True,
                    data={
                        "query_type": "project",
                        "keyword": keyword or "(전체)",
                        "total": len(filtered),
                        "result": filtered,
                    },
                    metadata={"source": "ERP Mock", "timestamp": datetime.now().isoformat()},
                )

            else:  # vendor
                items = _VENDOR_DATA["default"]
                filtered = _filter_by_keyword(items, keyword)
                return ToolResult(
                    success=True,
                    data={
                        "query_type": "vendor",
                        "keyword": keyword or "(전체)",
                        "total": len(filtered),
                        "result": filtered,
                    },
                    metadata={"source": "ERP Mock", "timestamp": datetime.now().isoformat()},
                )

        except Exception as e:
            logger.error("ErpLookupTool 실행 오류: %s", e)
            return ToolResult(success=False, error=f"ERP 조회 중 오류가 발생했습니다: {e}")
