"""
오용 감지 모듈
SFR-010: 비정상 사용 패턴 탐지 + IP 관리
"""
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class AbuseEvent:
    """오용 이벤트"""
    timestamp: str
    user_id: str
    ip_address: str
    event_type: str  # rate_limit, blacklisted_ip, suspicious_pattern
    details: str


class AbuseDetector:
    """비정상 사용 패턴 탐지기"""

    def __init__(
        self,
        rate_limit_window: int = 60,  # 윈도우 (초)
        rate_limit_max: int = 30,  # 윈도우당 최대 요청
    ):
        self.rate_limit_window = rate_limit_window
        self.rate_limit_max = rate_limit_max

        self._ip_blacklist: set[str] = set()
        self._ip_whitelist: set[str] = {"127.0.0.1", "::1"}

        # {ip: [timestamp, timestamp, ...]}
        self._request_log: dict[str, list[float]] = defaultdict(list)
        self._events: list[AbuseEvent] = []
        self._max_events = 1000

    def add_blacklist(self, ip: str):
        self._ip_blacklist.add(ip)
        self._ip_whitelist.discard(ip)

    def remove_blacklist(self, ip: str):
        self._ip_blacklist.discard(ip)

    def add_whitelist(self, ip: str):
        self._ip_whitelist.add(ip)
        self._ip_blacklist.discard(ip)

    def is_blacklisted(self, ip: str) -> bool:
        return ip in self._ip_blacklist

    def is_whitelisted(self, ip: str) -> bool:
        return ip in self._ip_whitelist

    def check_request(self, ip: str, user_id: str = "") -> AbuseEvent | None:
        """요청 검사 - 오용 감지 시 AbuseEvent 반환."""
        now = time.time()

        # Blacklist check
        if self.is_blacklisted(ip):
            event = AbuseEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_id=user_id, ip_address=ip,
                event_type="blacklisted_ip",
                details=f"Blocked request from blacklisted IP: {ip}",
            )
            self._record_event(event)
            return event

        # Whitelist bypass
        if self.is_whitelisted(ip):
            return None

        # Rate limit check
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
            self._record_event(event)
            return event

        return None

    def _record_event(self, event: AbuseEvent):
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        logger.warning("Abuse detected: %s - %s", event.event_type, event.details)

    def get_events(self, limit: int = 50) -> list[AbuseEvent]:
        return list(reversed(self._events[-limit:]))

    def get_stats(self) -> dict:
        return {
            "blacklist_count": len(self._ip_blacklist),
            "whitelist_count": len(self._ip_whitelist),
            "tracked_ips": len(self._request_log),
            "total_events": len(self._events),
        }

    def get_blacklist(self) -> list[str]:
        return sorted(self._ip_blacklist)

    def get_whitelist(self) -> list[str]:
        return sorted(self._ip_whitelist)


# Singleton
_detector: AbuseDetector | None = None


def get_abuse_detector() -> AbuseDetector:
    global _detector
    if _detector is None:
        _detector = AbuseDetector()
    return _detector
