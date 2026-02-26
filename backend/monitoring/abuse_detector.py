"""
오용 감지 모듈
SFR-010: 비정상 사용 패턴 탐지 + IP 관리
- Blacklist/whitelist/events: SQLite 영속화 (서버 재시작 시 유지)
- Rate limiting: 인메모리 (성능 우선, 재시작 시 리셋)
"""
import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from core.db.base import AsyncSQLiteManager

logger = logging.getLogger(__name__)


@dataclass
class AbuseEvent:
    """오용 이벤트"""
    timestamp: str
    user_id: str
    ip_address: str
    event_type: str  # rate_limit, blacklisted_ip, suspicious_pattern
    details: str


class AbuseDetector(AsyncSQLiteManager):
    """비정상 사용 패턴 탐지기 (SQLite 영속화)"""

    def __init__(
        self,
        rate_limit_window: int = 60,
        rate_limit_max: int = 30,
        db_path: Path | None = None,
    ):
        super().__init__(db_path or Path("data/abuse.db"))
        self.rate_limit_window = rate_limit_window
        self.rate_limit_max = rate_limit_max

        # Rate limiting stays in-memory for speed
        self._request_log: dict[str, list[float]] = defaultdict(list)

        # In-memory caches (loaded from DB on init)
        self._ip_blacklist: set[str] = set()
        self._ip_whitelist: set[str] = {"127.0.0.1", "::1"}
        self._lists_loaded = False

    async def _create_tables(self, db: aiosqlite.Connection) -> None:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ip_lists (
                ip TEXT PRIMARY KEY,
                list_type TEXT NOT NULL,
                added_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS abuse_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                ip_address TEXT,
                event_type TEXT NOT NULL,
                details TEXT
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_abuse_events_ts ON abuse_events(timestamp)"
        )
        await db.commit()

        # Seed default whitelist if table is empty
        async with db.execute("SELECT COUNT(*) FROM ip_lists") as cursor:
            row = await cursor.fetchone()
            if row and row[0] == 0:
                for ip in ("127.0.0.1", "::1"):
                    await db.execute(
                        "INSERT OR IGNORE INTO ip_lists (ip, list_type, added_at) VALUES (?, ?, ?)",
                        (ip, "whitelist", datetime.now(timezone.utc).isoformat()),
                    )
                await db.commit()

    async def _load_lists(self):
        """Load blacklist/whitelist from DB into memory cache."""
        if self._lists_loaded:
            return
        async with self.get_connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT ip, list_type FROM ip_lists") as cursor:
                rows = await cursor.fetchall()
            self._ip_blacklist = {r["ip"] for r in rows if r["list_type"] == "blacklist"}
            self._ip_whitelist = {r["ip"] for r in rows if r["list_type"] == "whitelist"}
        self._lists_loaded = True

    async def add_blacklist(self, ip: str):
        await self._load_lists()
        self._ip_blacklist.add(ip)
        self._ip_whitelist.discard(ip)
        async with self.get_connection() as db:
            await db.execute("DELETE FROM ip_lists WHERE ip = ?", (ip,))
            await db.execute(
                "INSERT INTO ip_lists (ip, list_type, added_at) VALUES (?, ?, ?)",
                (ip, "blacklist", datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()

    async def remove_blacklist(self, ip: str):
        await self._load_lists()
        self._ip_blacklist.discard(ip)
        async with self.get_connection() as db:
            await db.execute("DELETE FROM ip_lists WHERE ip = ? AND list_type = 'blacklist'", (ip,))
            await db.commit()

    async def add_whitelist(self, ip: str):
        await self._load_lists()
        self._ip_whitelist.add(ip)
        self._ip_blacklist.discard(ip)
        async with self.get_connection() as db:
            await db.execute("DELETE FROM ip_lists WHERE ip = ?", (ip,))
            await db.execute(
                "INSERT INTO ip_lists (ip, list_type, added_at) VALUES (?, ?, ?)",
                (ip, "whitelist", datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()

    def is_blacklisted(self, ip: str) -> bool:
        return ip in self._ip_blacklist

    def is_whitelisted(self, ip: str) -> bool:
        return ip in self._ip_whitelist

    async def check_request(self, ip: str, user_id: str = "") -> AbuseEvent | None:
        """요청 검사 - 오용 감지 시 AbuseEvent 반환."""
        await self._load_lists()
        now = time.time()

        # Blacklist check
        if self.is_blacklisted(ip):
            event = AbuseEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_id=user_id, ip_address=ip,
                event_type="blacklisted_ip",
                details=f"Blocked request from blacklisted IP: {ip}",
            )
            await self._record_event(event)
            return event

        # Whitelist bypass
        if self.is_whitelisted(ip):
            return None

        # Rate limit check (in-memory for speed)
        window_start = now - self.rate_limit_window
        self._request_log[ip] = [t for t in self._request_log[ip] if t > window_start]
        self._request_log[ip].append(now)

        if len(self._request_log[ip]) > self.rate_limit_max:
            event = AbuseEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_id=user_id, ip_address=ip,
                event_type="rate_limit",
                details=f"Rate limit exceeded: {len(self._request_log[ip])} requests in {self.rate_limit_window}s (max: {self.rate_limit_max})",
            )
            await self._record_event(event)
            return event

        return None

    async def _record_event(self, event: AbuseEvent):
        """Persist abuse event to SQLite."""
        try:
            async with self.get_connection() as db:
                await db.execute(
                    "INSERT INTO abuse_events (timestamp, user_id, ip_address, event_type, details) VALUES (?, ?, ?, ?, ?)",
                    (event.timestamp, event.user_id, event.ip_address, event.event_type, event.details),
                )
                await db.commit()
        except Exception:
            logger.exception("Failed to persist abuse event")
        logger.warning("Abuse detected: %s - %s", event.event_type, event.details)

    async def get_events(self, limit: int = 50) -> list[AbuseEvent]:
        """Get recent abuse events from SQLite."""
        try:
            async with self.get_connection() as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT timestamp, user_id, ip_address, event_type, details FROM abuse_events ORDER BY id DESC LIMIT ?",
                    (limit,),
                ) as cursor:
                    rows = await cursor.fetchall()
            return [
                AbuseEvent(
                    timestamp=r["timestamp"],
                    user_id=r["user_id"] or "",
                    ip_address=r["ip_address"] or "",
                    event_type=r["event_type"],
                    details=r["details"] or "",
                )
                for r in rows
            ]
        except Exception:
            logger.exception("Failed to read abuse events")
            return []

    async def get_stats(self) -> dict:
        await self._load_lists()
        event_count = 0
        try:
            async with self.get_connection() as db:
                async with db.execute("SELECT COUNT(*) FROM abuse_events") as cursor:
                    row = await cursor.fetchone()
                    event_count = row[0] if row else 0
        except Exception:
            pass
        return {
            "blacklist_count": len(self._ip_blacklist),
            "whitelist_count": len(self._ip_whitelist),
            "tracked_ips": len(self._request_log),
            "total_events": event_count,
        }

    async def get_blacklist(self) -> list[str]:
        await self._load_lists()
        return sorted(self._ip_blacklist)

    async def get_whitelist(self) -> list[str]:
        await self._load_lists()
        return sorted(self._ip_whitelist)


# Singleton
_detector: AbuseDetector | None = None
_detector_lock = asyncio.Lock()


async def get_abuse_detector() -> AbuseDetector:
    global _detector
    if _detector is None:
        async with _detector_lock:
            if _detector is None:
                _detector = AbuseDetector()
                await _detector._ensure_initialized()
    return _detector
