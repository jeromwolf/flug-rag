"""
시스템 DB 조회 도구 — 실제 SQLite 데이터베이스에서 시스템 현황을 조회합니다.

사용자 현황, 질의 통계, 문서 현황, 감사 로그 등 실시간 운영 데이터를 반환합니다.
실제 DB(memory.db, audit.db)를 읽기 전용으로 조회합니다.
"""
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from .base import BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult

logger = logging.getLogger(__name__)

# DB 경로 (backend/data/ 기준)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_MEMORY_DB = _DATA_DIR / "memory.db"
_AUDIT_DB = _DATA_DIR / "audit.db"


def _query(db_path: Path, sql: str, params: tuple = ()) -> list[dict]:
    """읽기 전용 SQLite 쿼리 실행."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


class SystemDbQueryTool(BaseTool):
    """시스템 DB 실시간 조회 도구 — 운영 현황 데이터를 실제 DB에서 조회합니다."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system_db_query",
            description="시스템 데이터베이스에서 사용자 현황, 질의 통계, 문서 현황, 감사 로그 등을 실시간으로 조회합니다",
            category="integration",
            help_text=(
                "시스템 운영 DB를 실시간으로 조회하는 도구입니다.\n\n"
                "query_type 종류:\n"
                "  - user_stats    : 사용자 현황 (역할별 인원수, 최근 가입자)\n"
                "  - query_stats   : 질의 통계 (일별 질의수, 총 세션수, 평균 대화 길이)\n"
                "  - document_stats: 문서 현황 (등록 문서수, 최근 업로드, 유형별 분류)\n"
                "  - audit_log     : 감사 로그 (최근 접근 기록, 로그인 이력)\n"
                "  - system_summary: 전체 시스템 요약 (모든 항목 종합)\n\n"
                "이 도구는 실제 운영 데이터베이스를 읽기 전용으로 조회합니다."
            ),
            parameters=[
                ToolParameter(
                    name="query_type",
                    type=ToolParamType.STRING,
                    description="조회 유형",
                    required=True,
                    enum=["user_stats", "query_stats", "document_stats", "audit_log", "system_summary"],
                ),
                ToolParameter(
                    name="days",
                    type=ToolParamType.INTEGER,
                    description="최근 N일 기준 조회 (기본: 7일)",
                    required=False,
                    default=7,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        query_type: str = kwargs.get("query_type", "system_summary")
        days: int = int(kwargs.get("days", 7))
        since = (datetime.now() - timedelta(days=days)).isoformat()

        try:
            if query_type == "user_stats":
                data = self._get_user_stats()
            elif query_type == "query_stats":
                data = self._get_query_stats(since, days)
            elif query_type == "document_stats":
                data = self._get_document_stats(since)
            elif query_type == "audit_log":
                data = self._get_audit_log(since, days)
            elif query_type == "system_summary":
                data = self._get_system_summary(since, days)
            else:
                return ToolResult(success=False, error=f"지원하지 않는 조회 유형: {query_type}")

            return ToolResult(
                success=True,
                data=data,
                metadata={
                    "source": "시스템 DB (실시간)",
                    "query_type": query_type,
                    "period": f"최근 {days}일",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        except Exception as e:
            logger.error("SystemDbQueryTool 실행 오류: %s", e, exc_info=True)
            return ToolResult(success=False, error=f"DB 조회 중 오류: {e}")

    def _get_user_stats(self) -> dict:
        """사용자 현황 조회."""
        # sessions 테이블에서 고유 user_id 추출
        users = _query(_MEMORY_DB, """
            SELECT user_id, COUNT(*) as session_count,
                   MIN(created_at) as first_session,
                   MAX(created_at) as last_session
            FROM sessions
            WHERE user_id IS NOT NULL AND user_id != ''
            GROUP BY user_id
            ORDER BY last_session DESC
        """)

        total_users = len(users)
        active_users = [u for u in users if u["session_count"] >= 2]

        return {
            "총 사용자 수": total_users,
            "활성 사용자 수": len(active_users),
            "사용자 목록": [
                {
                    "사용자ID": u["user_id"],
                    "세션 수": u["session_count"],
                    "최근 활동": u["last_session"],
                }
                for u in users[:10]
            ],
        }

    def _get_query_stats(self, since: str, days: int) -> dict:
        """질의 통계 조회."""
        # 전체 통계
        total = _query(_MEMORY_DB, "SELECT COUNT(*) as cnt FROM sessions")
        total_sessions = total[0]["cnt"] if total else 0

        total_msgs = _query(_MEMORY_DB, "SELECT COUNT(*) as cnt FROM messages")
        total_messages = total_msgs[0]["cnt"] if total_msgs else 0

        # 최근 N일 세션 수
        recent = _query(_MEMORY_DB, """
            SELECT COUNT(*) as cnt FROM sessions WHERE created_at >= ?
        """, (since,))
        recent_sessions = recent[0]["cnt"] if recent else 0

        # 최근 메시지 수
        recent_msgs = _query(_MEMORY_DB, """
            SELECT COUNT(*) as cnt FROM messages WHERE created_at >= ?
        """, (since,))
        recent_messages = recent_msgs[0]["cnt"] if recent_msgs else 0

        # 일별 질의 추이
        daily = _query(_MEMORY_DB, """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM messages
            WHERE role = 'user' AND created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 7
        """, (since,))

        # 평균 대화 길이 (세션당 메시지 수)
        avg_len = _query(_MEMORY_DB, """
            SELECT AVG(msg_count) as avg_messages FROM (
                SELECT session_id, COUNT(*) as msg_count
                FROM messages
                GROUP BY session_id
            )
        """)
        avg_messages = round(avg_len[0]["avg_messages"], 1) if avg_len and avg_len[0]["avg_messages"] else 0

        return {
            "전체 세션 수": total_sessions,
            "전체 메시지 수": total_messages,
            f"최근 {days}일 세션 수": recent_sessions,
            f"최근 {days}일 메시지 수": recent_messages,
            "세션당 평균 메시지 수": avg_messages,
            "일별 질의 추이": [{"날짜": d["date"], "질의 수": d["count"]} for d in daily],
        }

    def _get_document_stats(self, since: str) -> dict:
        """문서 현황 조회 — Milvus 벡터DB 기반."""
        try:
            # Milvus에서 컬렉션 통계 조회
            from core.vectorstore import create_vectorstore
            import asyncio

            vs = create_vectorstore()
            # count()는 async이므로 sync context에서 호출
            loop = asyncio.new_event_loop()
            try:
                count = loop.run_until_complete(vs.count())
            finally:
                loop.close()

            return {
                "벡터DB 청크 수": count,
                "벡터DB 유형": "Milvus Lite",
                "임베딩 모델": "BAAI/bge-m3 (1024차원)",
                "데이터 소스": "RAG평가용 문서 목록 (92파일)",
                "조회 시간": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        except Exception as e:
            logger.debug("Milvus count failed: %s", e)
            return {
                "벡터DB 청크 수": "조회 불가 (서버 전용 기능)",
                "데이터 소스": "RAG평가용 문서 목록",
            }

    def _get_audit_log(self, since: str, days: int) -> dict:
        """감사 로그 조회."""
        # 전체 통계
        total = _query(_AUDIT_DB, "SELECT COUNT(*) as cnt FROM audit_log")
        total_logs = total[0]["cnt"] if total else 0

        # 액션별 통계
        by_action = _query(_AUDIT_DB, """
            SELECT action, COUNT(*) as count
            FROM audit_log
            GROUP BY action
            ORDER BY count DESC
        """)

        # 최근 로그
        recent = _query(_AUDIT_DB, """
            SELECT timestamp, username, action, resource, ip_address
            FROM audit_log
            ORDER BY timestamp DESC
            LIMIT 10
        """)

        # 최근 N일 사용자별 활동
        user_activity = _query(_AUDIT_DB, """
            SELECT username, COUNT(*) as count, MAX(timestamp) as last_activity
            FROM audit_log
            WHERE timestamp >= ?
            GROUP BY username
            ORDER BY count DESC
        """, (since,))

        return {
            "전체 감사 로그 수": total_logs,
            "액션별 통계": [
                {"액션": a["action"], "건수": a["count"]}
                for a in by_action
            ],
            f"최근 {days}일 사용자 활동": [
                {"사용자": u["username"], "활동 수": u["count"], "최근 활동": u["last_activity"]}
                for u in user_activity
            ],
            "최근 로그 (10건)": [
                {
                    "시간": r["timestamp"],
                    "사용자": r["username"],
                    "액션": r["action"],
                    "대상": r["resource"],
                    "IP": r["ip_address"],
                }
                for r in recent
            ],
        }

    def _get_system_summary(self, since: str, days: int) -> dict:
        """전체 시스템 요약."""
        user_stats = self._get_user_stats()
        query_stats = self._get_query_stats(since, days)
        audit_stats = self._get_audit_log(since, days)

        return {
            "시스템 현황 요약": {
                "사용자": f"{user_stats['총 사용자 수']}명 (활성: {user_stats['활성 사용자 수']}명)",
                "세션": f"총 {query_stats['전체 세션 수']}건",
                "메시지": f"총 {query_stats['전체 메시지 수']}건",
                "세션당 평균 대화": f"{query_stats['세션당 평균 메시지 수']}건",
                "감사 로그": f"총 {audit_stats['전체 감사 로그 수']}건",
            },
            "일별 질의 추이": query_stats.get("일별 질의 추이", []),
            f"최근 {days}일 사용자 활동": audit_stats.get(f"최근 {days}일 사용자 활동", []),
            "조회 시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
