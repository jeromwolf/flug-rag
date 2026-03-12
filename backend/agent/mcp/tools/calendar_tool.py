"""캘린더 도구 — 일정 조회/등록/관리."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class CalendarTool(BaseTool):
    """캘린더 도구 — 일정 조회/등록/관리"""

    def __init__(self):
        self._file_path = settings.data_dir / "calendar_events.json"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calendar_manager",
            description="일정을 조회, 등록, 검색합니다.",
            category="productivity",
            help_text=(
                "일정을 조회, 등록, 검색합니다. "
                "점검 일정, 회의, 교육 일정 관리에 사용합니다.\n"
                "파라미터:\n"
                "  - action: 수행할 작업 (list/create/search)\n"
                "  - date: 날짜 (YYYY-MM-DD, list/create에 사용)\n"
                "  - title: 일정 제목 (create에 필수)\n"
                "  - description: 일정 설명 (create에 사용)\n"
                "  - query: 검색어 (search에 사용)\n"
                "액션:\n"
                "  - list: 특정 날짜의 일정 조회 (날짜 미지정 시 오늘)\n"
                "  - create: 새 일정 등록\n"
                "  - search: 제목/설명에서 키워드 검색"
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type=ToolParamType.STRING,
                    description="수행할 작업",
                    enum=["list", "create", "search"],
                ),
                ToolParameter(
                    name="date",
                    type=ToolParamType.STRING,
                    description="날짜 (YYYY-MM-DD 형식, list/create에 사용)",
                    required=False,
                ),
                ToolParameter(
                    name="title",
                    type=ToolParamType.STRING,
                    description="일정 제목 (create에 필수)",
                    required=False,
                ),
                ToolParameter(
                    name="description",
                    type=ToolParamType.STRING,
                    description="일정 설명 (create에 사용)",
                    required=False,
                ),
                ToolParameter(
                    name="query",
                    type=ToolParamType.STRING,
                    description="검색 키워드 (search에 사용)",
                    required=False,
                ),
            ],
        )

    def _load_events(self) -> list[dict]:
        """Load events from JSON file."""
        if not self._file_path.exists():
            return []
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
            return data.get("events", [])
        except (json.JSONDecodeError, KeyError):
            logger.warning("캘린더 파일 파싱 실패, 빈 목록 반환")
            return []

    def _save_events(self, events: list[dict]) -> None:
        """Save events to JSON file."""
        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(
            json.dumps({"events": events}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "")
        if not action:
            return ToolResult(success=False, error="action 파라미터가 필요합니다.")

        try:
            if action == "list":
                return await self._list_events(kwargs)
            elif action == "create":
                return await self._create_event(kwargs)
            elif action == "search":
                return await self._search_events(kwargs)
            else:
                return ToolResult(success=False, error=f"알 수 없는 액션: {action}")
        except Exception as e:
            logger.error("캘린더 작업 중 오류: %s", e)
            return ToolResult(success=False, error=f"캘린더 작업 중 오류가 발생했습니다: {e}")

    async def _list_events(self, kwargs: dict) -> ToolResult:
        """List events for a given date."""
        date_str = kwargs.get("date", "")
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return ToolResult(
                success=False,
                error=f"잘못된 날짜 형식입니다: {date_str}. YYYY-MM-DD 형식을 사용하세요.",
            )

        events = self._load_events()
        day_events = [e for e in events if e.get("date") == date_str]

        # Sort by time if available
        day_events.sort(key=lambda e: e.get("time", "00:00"))

        return ToolResult(
            success=True,
            data={
                "date": date_str,
                "events": day_events,
                "count": len(day_events),
            },
            metadata={"action": "list"},
        )

    async def _create_event(self, kwargs: dict) -> ToolResult:
        """Create a new event."""
        title = kwargs.get("title", "")
        if not title:
            return ToolResult(success=False, error="일정 제목(title)이 필요합니다.")

        date_str = kwargs.get("date", "")
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return ToolResult(
                success=False,
                error=f"잘못된 날짜 형식입니다: {date_str}. YYYY-MM-DD 형식을 사용하세요.",
            )

        description = kwargs.get("description", "")
        now = datetime.now()

        new_event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "date": date_str,
            "time": now.strftime("%H:%M"),
            "created_at": now.isoformat(),
        }

        events = self._load_events()
        events.append(new_event)
        self._save_events(events)

        logger.info("캘린더 일정 생성: %s (%s)", title, date_str)

        return ToolResult(
            success=True,
            data={
                "message": "일정이 등록되었습니다.",
                "event": new_event,
            },
            metadata={"action": "create"},
        )

    async def _search_events(self, kwargs: dict) -> ToolResult:
        """Search events by keyword."""
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="검색어(query)가 필요합니다.")

        events = self._load_events()
        query_lower = query.lower()

        matched = [
            e for e in events
            if query_lower in (e.get("title", "") + " " + e.get("description", "")).lower()
        ]

        # Sort by date descending (most recent first)
        matched.sort(key=lambda e: e.get("date", ""), reverse=True)

        return ToolResult(
            success=True,
            data={
                "query": query,
                "events": matched,
                "count": len(matched),
            },
            metadata={"action": "search"},
        )
