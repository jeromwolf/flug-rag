"""
EHSQ 시스템 연동 Mock 도구.

환경(Environment)·보건(Health)·안전(Safety)·품질(Quality) 시스템 데이터를 조회하는
모의(Mock) 도구입니다. 실제 EHSQ 서버에 연결하지 않으며 데모용 더미 데이터를 반환합니다.
"""
import logging
from datetime import datetime

from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult
from config.settings import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Mock 데이터 상수
# ──────────────────────────────────────────────────────────────────────────────

_INCIDENT_DATA = {
    "summary": {
        "기준일": datetime.now().strftime("%Y년 %m월 %d일"),
        "집계기간": "2026년 1월 1일 ~ 현재",
        "총 사고건수": 3,
        "중대재해": 0,
        "경미사고": 2,
        "아차사고": 1,
        "무재해 누적일수": 187,
        "전년 동기 대비": "-40% (개선)",
    },
    "incidents": [
        {
            "사고번호": "INC-2026-003",
            "발생일": "2026-02-18",
            "분류": "아차사고",
            "시설": "인천 LNG 기지",
            "내용": "배관 플랜지 작업 중 공구 낙하 (인명 피해 없음)",
            "조치상태": "원인분석 완료 / 재발방지 수립 중",
            "담당자": "안전팀 홍길동",
        },
        {
            "사고번호": "INC-2026-002",
            "발생일": "2026-01-29",
            "분류": "경미사고",
            "시설": "경기북부 공급관리소",
            "내용": "밸브 교체 작업 중 경미한 찰과상 (1명, 치료 후 복귀)",
            "조치상태": "조치 완료",
            "담당자": "안전팀 이안전",
        },
        {
            "사고번호": "INC-2026-001",
            "발생일": "2026-01-08",
            "분류": "경미사고",
            "시설": "부산 정압기실",
            "내용": "전기 작업 중 아크 화상 (2도 경미, 치료 완료)",
            "조치상태": "조치 완료 / 전기안전 특별교육 실시",
            "담당자": "안전팀 김전기",
        },
    ],
}

_SAFETY_STATUS_DATA = {
    "전체": {
        "기준일": datetime.now().strftime("%Y년 %m월 %d일"),
        "안전등급": "B+",
        "핵심지표": {
            "재해율": "0.08%",
            "도수율": "1.2",
            "강도율": "0.003",
            "위험성평가 완료율": "94%",
        },
        "시설별 상태": [
            {"시설명": "인천 LNG 기지",     "등급": "A",  "최근점검일": "2026-02-10", "지적사항": 0},
            {"시설명": "경기북부 공급관리소", "등급": "B+", "최근점검일": "2026-01-25", "지적사항": 2},
            {"시설명": "부산 정압기실",       "등급": "B",  "최근점검일": "2026-02-03", "지적사항": 3},
            {"시설명": "대전 가스공급소",     "등급": "A-", "최근점검일": "2026-01-15", "지적사항": 1},
            {"시설명": "광주 배관망 관리소",  "등급": "B+", "최근점검일": "2026-02-20", "지적사항": 1},
        ],
        "이달 예정 점검": [
            {"시설명": "울산 저장탱크",   "예정일": "2026-03-10", "점검유형": "정기안전점검"},
            {"시설명": "수원 정압기실",   "예정일": "2026-03-17", "점검유형": "특별점검"},
        ],
    },
}

_COMPLIANCE_DATA = {
    "전체": {
        "기준일": datetime.now().strftime("%Y년 %m월 %d일"),
        "전체 준수율": "97.3%",
        "법정의무 이행": [
            {"항목": "고압가스 안전관리법 정기검사",    "기한": "2026-04-30", "상태": "이행중"},
            {"항목": "산업안전보건법 안전보건관리계획",  "기한": "2026-12-31", "상태": "완료"},
            {"항목": "환경영향평가 이행보고",            "기한": "2026-06-30", "상태": "준비중"},
            {"항목": "품질경영시스템(ISO 9001) 심사",    "기한": "2026-05-20", "상태": "준비중"},
            {"항목": "환경경영시스템(ISO 14001) 심사",   "기한": "2026-09-10", "상태": "일정수립"},
        ],
        "미이행 항목": [
            {
                "항목": "작업환경측정 (일부 사업장)",
                "기한": "2026-03-15",
                "사유": "측정기관 일정 조율 중",
                "조치계획": "2026-03-10까지 완료 예정",
            },
        ],
        "비고": "미이행 1건은 조치계획 수립 완료, 기한 내 이행 예정",
    },
}


def _get_facility_filtered(base_data: dict, facility: str) -> dict:
    """시설명으로 필터링 (facility 미입력 시 전체 반환)."""
    if not facility:
        return base_data.get("전체", base_data)
    # 시설별 상태가 있는 경우 keyword 포함 항목만 추출
    result = dict(base_data.get("전체", base_data))
    for list_key in ("시설별 상태", "incidents"):
        if list_key in result:
            result[list_key] = [
                item for item in result[list_key]
                if facility in str(item)
            ]
    return result


class EhsqTool(BaseTool):
    """EHSQ 시스템 Mock 조회 도구 — 안전사고·안전현황·컴플라이언스 정보 반환."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ehsq_lookup",
            description="EHSQ 시스템에서 안전사고, 안전현황, 법정 컴플라이언스 정보를 조회합니다 (데모용 Mock 데이터)",
            category="integration",
            help_text=(
                f"{settings.platform_name} EHSQ(환경·보건·안전·품질) 시스템 조회 도구입니다 (데모 Mock).\n\n"
                "action 종류:\n"
                "  - incident_report   : 안전사고 현황 (발생건수·중대재해·무재해일수·사고목록)\n"
                "  - safety_status     : 시설별 안전등급 및 점검 현황\n"
                "  - compliance_check  : 법정 의무 준수 현황 (기한·이행상태·미이행 조치계획)\n\n"
                "facility 예시: '인천 LNG 기지', '부산 정압기실' (미입력 시 전체)\n\n"
                "⚠️ 본 도구는 실제 EHSQ 서버에 연결되지 않으며 데모용 Mock 데이터를 반환합니다."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type=ToolParamType.STRING,
                    description="조회 액션 (incident_report: 사고현황, safety_status: 안전현황, compliance_check: 컴플라이언스)",
                    required=True,
                    enum=["incident_report", "safety_status", "compliance_check"],
                ),
                ToolParameter(
                    name="facility",
                    type=ToolParamType.STRING,
                    description="특정 시설 필터 (예: '인천 LNG 기지'). 미입력 시 전체 조회",
                    required=False,
                    default="",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        action: str = kwargs.get("action", "").strip().lower()
        facility: str = kwargs.get("facility", "").strip()

        valid_actions = ("incident_report", "safety_status", "compliance_check")
        if action not in valid_actions:
            return ToolResult(
                success=False,
                error=f"action은 {' / '.join(valid_actions)} 중 하나여야 합니다.",
            )

        try:
            if action == "incident_report":
                data = dict(_INCIDENT_DATA)
                if facility:
                    data["incidents"] = [
                        inc for inc in _INCIDENT_DATA["incidents"]
                        if facility in inc.get("시설", "")
                    ]
                return ToolResult(
                    success=True,
                    data={"action": action, "facility": facility or "전체", "result": data},
                    metadata={"source": "EHSQ Mock", "timestamp": datetime.now().isoformat()},
                )

            elif action == "safety_status":
                data = _get_facility_filtered(_SAFETY_STATUS_DATA, facility)
                return ToolResult(
                    success=True,
                    data={"action": action, "facility": facility or "전체", "result": data},
                    metadata={"source": "EHSQ Mock", "timestamp": datetime.now().isoformat()},
                )

            else:  # compliance_check
                data = _get_facility_filtered(_COMPLIANCE_DATA, facility)
                return ToolResult(
                    success=True,
                    data={"action": action, "facility": facility or "전체", "result": data},
                    metadata={"source": "EHSQ Mock", "timestamp": datetime.now().isoformat()},
                )

        except Exception as e:
            logger.error("EhsqTool 실행 오류: %s", e)
            return ToolResult(success=False, error=f"EHSQ 조회 중 오류가 발생했습니다: {e}")
