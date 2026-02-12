"""
알림 시스템
SFR-010: 리소스 임계치 모니터링 + 알림
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """알림 규칙"""
    name: str
    metric: str  # cpu, memory, disk, gpu, error_rate
    threshold: float
    operator: str = ">"  # >, <, >=, <=
    cooldown_seconds: int = 300  # 중복 알림 방지 쿨다운
    last_triggered: float = 0.0


@dataclass
class Alert:
    """발생한 알림"""
    rule_name: str
    metric: str
    current_value: float
    threshold: float
    timestamp: str
    message: str


class AlertManager:
    """알림 관리자"""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.history: list[Alert] = []
        self._max_history = 500
        self._webhooks: list[str] = []
        self._pending_tasks: list = []

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def remove_rule(self, name: str):
        self.rules = [r for r in self.rules if r.name != name]

    def add_webhook(self, url: str):
        if url not in self._webhooks:
            self._webhooks.append(url)

    def check_threshold(self, rule: AlertRule, value: float) -> bool:
        ops = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
               ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b}
        op = ops.get(rule.operator, ops[">"])
        return op(value, rule.threshold)

    async def evaluate(self, metrics: dict[str, float]) -> list[Alert]:
        """메트릭 평가 및 알림 발생."""
        now = time.time()
        alerts = []

        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                continue

            if not self.check_threshold(rule, value):
                continue

            if now - rule.last_triggered < rule.cooldown_seconds:
                continue

            rule.last_triggered = now
            alert = Alert(
                rule_name=rule.name,
                metric=rule.metric,
                current_value=value,
                threshold=rule.threshold,
                timestamp=datetime.now(timezone.utc).isoformat(),
                message=f"[{rule.name}] {rule.metric} = {value:.2f} (임계치: {rule.operator} {rule.threshold})",
            )
            alerts.append(alert)
            self.history.append(alert)

            if len(self.history) > self._max_history:
                self.history = self.history[-self._max_history:]

        # Dispatch webhooks
        if alerts and self._webhooks:
            task = asyncio.create_task(self._dispatch_webhooks(alerts))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
            self._pending_tasks = [t for t in self._pending_tasks if not t.done()]
            self._pending_tasks.append(task)

        return alerts

    async def _dispatch_webhooks(self, alerts: list[Alert]):
        """웹훅 알림 전송."""
        payload = {
            "alerts": [
                {"rule": a.rule_name, "metric": a.metric,
                 "value": a.current_value, "threshold": a.threshold,
                 "message": a.message, "timestamp": a.timestamp}
                for a in alerts
            ]
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in self._webhooks:
                try:
                    await client.post(url, json=payload)
                except Exception as e:
                    logger.warning("Webhook dispatch failed (%s): %s", url, e)

    async def collect_system_metrics(self) -> dict[str, float]:
        """시스템 메트릭 수집."""
        import shutil
        metrics = {}
        try:
            import psutil
            metrics["cpu"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics["memory"] = mem.percent
        except ImportError:
            pass

        try:
            disk = shutil.disk_usage("/")
            metrics["disk"] = (disk.used / disk.total) * 100
        except Exception:
            pass

        return metrics

    def get_history(self, limit: int = 50) -> list[Alert]:
        return list(reversed(self.history[-limit:]))

    def get_rules(self) -> list[dict]:
        return [{"name": r.name, "metric": r.metric, "threshold": r.threshold,
                 "operator": r.operator, "cooldown": r.cooldown_seconds} for r in self.rules]


# Singleton
_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
        # Default rules
        _alert_manager.add_rule(AlertRule(name="CPU 높음", metric="cpu", threshold=90.0))
        _alert_manager.add_rule(AlertRule(name="메모리 높음", metric="memory", threshold=85.0))
        _alert_manager.add_rule(AlertRule(name="디스크 높음", metric="disk", threshold=90.0))
    return _alert_manager
