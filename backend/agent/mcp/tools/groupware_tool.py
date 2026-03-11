"""
그룹웨어 연동 Mock 도구.

한국가스기술공사 그룹웨어(일정·결재·공지)를 조회하는 모의(Mock) 도구입니다.
실제 그룹웨어 서버에 연결하지 않으며 데모용 더미 데이터를 반환합니다.
"""
import logging
from datetime import datetime

from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult
from config.settings import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Mock 데이터 상수
# ──────────────────────────────────────────────────────────────────────────────

_SCHEDULE_DATA = [
    {
        "일정ID": "SCH-2026-0310",
        "제목": "2026년 1분기 안전점검 결과 보고 (전체 임원)",
        "일시": "2026-03-10 14:00",
        "장소": "본사 대회의실 (5층)",
        "주관": "안전관리본부",
        "참석": ["대표이사", "안전관리본부장", "각 사업부장", "기술지원팀장"],
        "비고": "PPT 자료 3/8까지 제출",
    },
    {
        "일정ID": "SCH-2026-0312",
        "제목": "ISO 9001 내부심사 (품질팀)",
        "일시": "2026-03-12 09:00 ~ 17:00",
        "장소": "본사 2층 회의실 A",
        "주관": "품질경영팀",
        "참석": ["품질경영팀 전원", "각 부서 심사 대상자"],
        "비고": "내부심사원 사전 준비 완료 요망",
    },
    {
        "일정ID": "SCH-2026-0317",
        "제목": "수원 정압기실 특별점검",
        "일시": "2026-03-17 10:00",
        "장소": "수원 정압기실 현장",
        "주관": "시설관리팀",
        "참석": ["시설관리팀장", "안전담당", "외부 점검기관"],
        "비고": "현장 접근 절차 사전 협의 완료",
    },
    {
        "일정ID": "SCH-2026-0325",
        "제목": "R&D 기술 세미나 — 수소 배관 신소재",
        "일시": "2026-03-25 13:30",
        "장소": "기술연구소 세미나실",
        "주관": "기술연구소",
        "참석": ["연구원 전원", "사업부 희망자"],
        "비고": "외부 발표자: KAIST 에너지공학과 교수",
    },
]

_APPROVAL_DATA = [
    {
        "결재번호": "APR-2026-00412",
        "문서명": "2026년 상반기 안전장비 구매 요청서",
        "기안자": "안전관리팀 이장비",
        "기안일": "2026-03-04",
        "결재금액": "4,800만 원",
        "현재단계": "팀장 결재 대기",
        "결재선": ["안전관리팀장 (대기)", "안전관리본부장", "구매팀 최종 확인"],
        "상태": "진행중",
    },
    {
        "결재번호": "APR-2026-00398",
        "문서명": "국외출장 결과보고서 (2026 가스기술 국제 컨퍼런스)",
        "기안자": "기술연구소 박연구",
        "기안일": "2026-02-28",
        "결재금액": "없음",
        "현재단계": "완료",
        "결재선": ["팀장 (완료)", "본부장 (완료)", "대표이사 (완료)"],
        "상태": "완료",
    },
    {
        "결재번호": "APR-2026-00401",
        "문서명": "협력업체 등록 갱신 신청 — 삼일안전기술(주)",
        "기안자": "구매팀 김구매",
        "기안일": "2026-03-01",
        "결재금액": "없음",
        "현재단계": "담당부서장 검토 반려",
        "결재선": ["구매팀장 (반려)", "법무팀 확인 요청"],
        "상태": "반려",
        "반려사유": "업체 평가등급 미달 (C등급) — 재평가 후 재신청 필요",
    },
    {
        "결재번호": "APR-2026-00405",
        "문서명": "직원 해외연수 계획서 (독일 가스기술박람회 E-world)",
        "기안자": "인사팀 정인사",
        "기안일": "2026-03-03",
        "결재금액": "1,200만 원",
        "현재단계": "본부장 결재 대기",
        "결재선": ["인사팀장 (완료)", "경영지원본부장 (대기)"],
        "상태": "진행중",
    },
]

_NOTICE_DATA = [
    {
        "공지번호": "NTC-2026-047",
        "제목": "[필독] 2026년 안전의식 제고 캠페인 — 전 직원 서약서 제출 안내",
        "작성부서": "안전관리본부",
        "게시일": "2026-03-04",
        "마감일": "2026-03-14",
        "내용요약": "1분기 안전의식 제고 캠페인 일환으로 전 직원 안전서약서 제출 필요. 사내 시스템에서 전자서명 가능.",
        "첨부": ["2026_안전서약서_양식.hwp"],
        "중요도": "긴급",
    },
    {
        "공지번호": "NTC-2026-045",
        "제목": "2026년 3월 정기 시스템 점검 안내 (3/15 02:00 ~ 06:00)",
        "작성부서": "IT운영팀",
        "게시일": "2026-03-01",
        "마감일": None,
        "내용요약": "3월 15일(토) 새벽 2시~6시 ERP·그룹웨어·EHSQ 시스템 정기점검 예정. 해당 시간대 접속 불가.",
        "첨부": [],
        "중요도": "일반",
    },
    {
        "공지번호": "NTC-2026-042",
        "제목": "2025년 우수직원 포상 명단 발표",
        "작성부서": "인사팀",
        "게시일": "2026-02-25",
        "마감일": None,
        "내용요약": "2025년 우수직원 15명 포상. 시상식: 3/4 14:00 본사 대강당.",
        "첨부": ["2025_우수직원_포상명단.pdf"],
        "중요도": "일반",
    },
    {
        "공지번호": "NTC-2026-038",
        "제목": "ISO 9001 내부심사 일정 및 체크리스트 배포",
        "작성부서": "품질경영팀",
        "게시일": "2026-02-20",
        "마감일": "2026-03-11",
        "내용요약": "3월 12일 내부심사 대비 체크리스트 확인 및 사전 서류 준비 요망.",
        "첨부": ["ISO9001_내부심사_체크리스트_2026.xlsx"],
        "중요도": "중요",
    },
]


def _keyword_filter(items: list, keyword: str) -> list:
    """keyword가 포함된 항목만 반환."""
    if not keyword:
        return items
    kw = keyword.strip().lower()
    return [item for item in items if any(kw in str(v).lower() for v in item.values())]


class GroupwareTool(BaseTool):
    """그룹웨어 Mock 조회 도구 — 일정·결재·공지 정보 반환."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="groupware_lookup",
            description="그룹웨어에서 일정, 결재 현황, 공지사항을 조회합니다 (데모용 Mock 데이터)",
            category="integration",
            help_text=(
                f"{settings.platform_name} 그룹웨어 조회 도구입니다 (데모 Mock).\n\n"
                "action 종류:\n"
                "  - schedule : 일정 조회 (회의·점검·세미나 등 전사 캘린더)\n"
                "  - approval : 결재 현황 (문서명·결재단계·상태·반려사유)\n"
                "  - notice   : 공지사항 (필독·일반·긴급 공지 목록)\n\n"
                "keyword 예시:\n"
                "  schedule → '안전', 'ISO', '3월'\n"
                "  approval → '안전장비', '출장', '반려'\n"
                "  notice   → '서약서', '점검', 'ISO'\n\n"
                "⚠️ 본 도구는 실제 그룹웨어에 연결되지 않으며 데모용 Mock 데이터를 반환합니다."
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type=ToolParamType.STRING,
                    description="조회 액션 (schedule: 일정, approval: 결재현황, notice: 공지사항)",
                    required=True,
                    enum=["schedule", "approval", "notice"],
                ),
                ToolParameter(
                    name="keyword",
                    type=ToolParamType.STRING,
                    description="검색 키워드 (예: '안전', '출장', 'ISO'). 미입력 시 전체 조회",
                    required=False,
                    default="",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        action: str = kwargs.get("action", "").strip().lower()
        keyword: str = kwargs.get("keyword", "").strip()

        valid_actions = ("schedule", "approval", "notice")
        if action not in valid_actions:
            return ToolResult(
                success=False,
                error=f"action은 {' / '.join(valid_actions)} 중 하나여야 합니다.",
            )

        try:
            if action == "schedule":
                items = _keyword_filter(_SCHEDULE_DATA, keyword)
                return ToolResult(
                    success=True,
                    data={
                        "action": action,
                        "keyword": keyword or "(전체)",
                        "total": len(items),
                        "result": items,
                    },
                    metadata={"source": "그룹웨어 Mock", "timestamp": datetime.now().isoformat()},
                )

            elif action == "approval":
                items = _keyword_filter(_APPROVAL_DATA, keyword)
                summary = {
                    "진행중": sum(1 for i in items if i.get("상태") == "진행중"),
                    "완료":   sum(1 for i in items if i.get("상태") == "완료"),
                    "반려":   sum(1 for i in items if i.get("상태") == "반려"),
                }
                return ToolResult(
                    success=True,
                    data={
                        "action": action,
                        "keyword": keyword or "(전체)",
                        "total": len(items),
                        "summary": summary,
                        "result": items,
                    },
                    metadata={"source": "그룹웨어 Mock", "timestamp": datetime.now().isoformat()},
                )

            else:  # notice
                items = _keyword_filter(_NOTICE_DATA, keyword)
                return ToolResult(
                    success=True,
                    data={
                        "action": action,
                        "keyword": keyword or "(전체)",
                        "total": len(items),
                        "result": items,
                    },
                    metadata={"source": "그룹웨어 Mock", "timestamp": datetime.now().isoformat()},
                )

        except Exception as e:
            logger.error("GroupwareTool 실행 오류: %s", e)
            return ToolResult(success=False, error=f"그룹웨어 조회 중 오류가 발생했습니다: {e}")
