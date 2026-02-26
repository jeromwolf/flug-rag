"""
Guardrails 모듈
SFR-003: 입력/출력 안전장치
- 프롬프트 인젝션 탐지
- 키워드 필터링 (관리자 설정)
- 출력 민감정보 마스킹
- 필터링 이력 로깅
"""
import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class GuardRule:
    """Guardrail 규칙"""
    id: str
    name: str
    rule_type: str  # "input_keyword", "input_pattern", "output_keyword", "output_pattern"
    pattern: str  # 정규식 또는 키워드
    action: str = "block"  # "block", "warn", "mask"
    message: str = ""  # 사용자 안내 메시지
    is_active: bool = True
    priority: int = 0
    created_at: str = ""
    created_by: str = ""


@dataclass
class GuardResult:
    """Guardrail 검사 결과"""
    passed: bool
    triggered_rules: list[str] = field(default_factory=list)
    action: str = "allow"  # "allow", "block", "warn", "mask"
    message: str = ""
    modified_text: str | None = None


@dataclass
class GuardLog:
    """Guardrail 필터링 이력"""
    id: str
    timestamp: str
    direction: str  # "input" or "output"
    rule_id: str
    rule_name: str
    action: str
    user_id: str = ""
    snippet: str = ""  # 매칭된 텍스트 일부 (최대 100자)


# Built-in prompt injection patterns (Korean + English)
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)ignore\s+the\s+above",
    r"(?i)disregard\s+(all\s+)?prior",
    r"(?i)you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"(?i)system\s*:\s*",
    r"(?i)forget\s+(everything|all)",
    r"(?i)new\s+instructions?\s*:",
    r"(?i)override\s+(?:system|prompt)",
    r"(?i)act\s+as\s+(?:a|an)\s+",
    r"이전\s*(?:지시|명령|설정).*무시",
    r"시스템\s*프롬프트.*변경",
    r"역할.*변경.*해줘",
    r"(?i)jailbreak",
    r"(?i)DAN\s+mode",
]

# Maximum input length (characters)
MAX_INPUT_LENGTH = 10000


class GuardrailsManager:
    """Guardrails 관리 + 검사 엔진"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/guardrails.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._lock = asyncio.Lock()
        self._rules_cache: list[GuardRule] | None = None

    async def _ensure_initialized(self):
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self._init_db()
                    self._initialized = True

    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS guard_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    action TEXT DEFAULT 'block',
                    message TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    priority INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    created_by TEXT DEFAULT ''
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS guard_logs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    rule_id TEXT,
                    rule_name TEXT,
                    action TEXT,
                    user_id TEXT DEFAULT '',
                    snippet TEXT DEFAULT ''
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_guard_logs_ts ON guard_logs(timestamp)"
            )
            await db.commit()

    # ==================== Rule CRUD ====================

    async def add_rule(self, name: str, rule_type: str, pattern: str,
                       action: str = "block", message: str = "",
                       priority: int = 0, created_by: str = "") -> GuardRule:
        await self._ensure_initialized()
        rule = GuardRule(
            id=str(uuid.uuid4()),
            name=name, rule_type=rule_type, pattern=pattern,
            action=action, message=message, priority=priority,
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by=created_by,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO guard_rules
                   (id, name, rule_type, pattern, action, message, is_active, priority, created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (rule.id, rule.name, rule.rule_type, rule.pattern,
                 rule.action, rule.message, rule.priority,
                 rule.created_at, rule.created_by),
            )
            await db.commit()
        self._rules_cache = None
        return rule

    async def update_rule(self, rule_id: str, **kwargs) -> GuardRule:
        await self._ensure_initialized()
        _SAFE_COLUMN = re.compile(r'^[a-z_]+$')
        _ALLOWED_COLUMNS = frozenset({"name", "rule_type", "pattern", "action", "message", "is_active", "priority"})
        updates = {k: v for k, v in kwargs.items() if k in _ALLOWED_COLUMNS and _SAFE_COLUMN.match(k)}
        if not updates:
            raise ValueError("No valid fields to update")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [rule_id]

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(f"UPDATE guard_rules SET {set_clause} WHERE id = ?", values)
            if cursor.rowcount == 0:
                raise ValueError(f"Rule not found: {rule_id}")
            await db.commit()

            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM guard_rules WHERE id = ?", (rule_id,)) as cur:
                row = await cur.fetchone()
                self._rules_cache = None
                return GuardRule(**{**dict(row), "is_active": bool(row["is_active"])})

    async def delete_rule(self, rule_id: str):
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM guard_rules WHERE id = ?", (rule_id,))
            if cursor.rowcount == 0:
                raise ValueError(f"Rule not found: {rule_id}")
            await db.commit()
        self._rules_cache = None

    async def list_rules(self) -> list[GuardRule]:
        await self._ensure_initialized()
        if self._rules_cache is not None:
            return self._rules_cache

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM guard_rules WHERE is_active = 1 ORDER BY priority DESC, name"
            ) as cur:
                rows = await cur.fetchall()
                rules = [GuardRule(**{**dict(r), "is_active": bool(r["is_active"])}) for r in rows]
                self._rules_cache = rules
                return rules

    async def list_all_rules(self) -> list[GuardRule]:
        """모든 규칙 (비활성 포함)"""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM guard_rules ORDER BY priority DESC, name") as cur:
                rows = await cur.fetchall()
                return [GuardRule(**{**dict(r), "is_active": bool(r["is_active"])}) for r in rows]

    # ==================== Guard Checks ====================

    async def check_input(self, text: str, user_id: str = "") -> GuardResult:
        """입력 텍스트 검사."""
        # 1. Length check
        if len(text) > MAX_INPUT_LENGTH:
            await self._log("input", "system", "length_limit", "block", user_id,
                            f"입력 길이 초과: {len(text)} chars")
            return GuardResult(
                passed=False,
                triggered_rules=["length_limit"],
                action="block",
                message=f"입력이 너무 깁니다. 최대 {MAX_INPUT_LENGTH}자까지 허용됩니다.",
            )

        # 2. Built-in prompt injection patterns
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text):
                await self._log("input", "system", "injection_detect", "block", user_id,
                                text[:100])
                return GuardResult(
                    passed=False,
                    triggered_rules=["injection_detect"],
                    action="block",
                    message="입력에 허용되지 않는 패턴이 감지되었습니다. 질문을 수정해주세요.",
                )

        # 3. Custom rules
        rules = await self.list_rules()
        input_rules = [r for r in rules if r.rule_type.startswith("input_")]

        triggered = []
        worst_action = "allow"
        worst_message = ""
        action_priority = {"allow": 0, "warn": 1, "mask": 2, "block": 3}

        for rule in input_rules:
            try:
                if rule.rule_type == "input_keyword":
                    matched = rule.pattern.lower() in text.lower()
                else:
                    matched = bool(re.search(rule.pattern, text, re.IGNORECASE))

                if matched:
                    triggered.append(rule.name)
                    await self._log("input", rule.id, rule.name, rule.action, user_id, text[:100])
                    if action_priority.get(rule.action, 0) > action_priority.get(worst_action, 0):
                        worst_action = rule.action
                        worst_message = rule.message
            except re.error:
                logger.warning("Invalid regex in rule %s: %s", rule.name, rule.pattern)

        if worst_action == "block":
            return GuardResult(
                passed=False, triggered_rules=triggered, action="block",
                message=worst_message or "입력이 필터링되었습니다.",
            )

        if worst_action == "warn":
            return GuardResult(
                passed=True, triggered_rules=triggered, action="warn",
                message=worst_message or "입력에 주의가 필요한 내용이 포함되어 있습니다.",
            )

        return GuardResult(passed=True)

    async def check_output(self, text: str, user_id: str = "") -> GuardResult:
        """출력 텍스트 검사 + 마스킹."""
        rules = await self.list_rules()
        output_rules = [r for r in rules if r.rule_type.startswith("output_")]

        modified = text
        triggered = []

        for rule in output_rules:
            try:
                if rule.rule_type == "output_keyword":
                    if rule.pattern.lower() in modified.lower():
                        triggered.append(rule.name)
                        if rule.action == "mask":
                            modified = re.sub(re.escape(rule.pattern), "[마스킹됨]", modified, flags=re.IGNORECASE)
                        elif rule.action == "block":
                            await self._log("output", rule.id, rule.name, "block", user_id, text[:100])
                            return GuardResult(
                                passed=False, triggered_rules=triggered, action="block",
                                message=rule.message or "응답에 부적절한 내용이 포함되어 필터링되었습니다.",
                            )
                else:
                    if re.search(rule.pattern, modified, re.IGNORECASE):
                        triggered.append(rule.name)
                        if rule.action == "mask":
                            modified = re.sub(rule.pattern, "[마스킹됨]", modified, flags=re.IGNORECASE)
                        elif rule.action == "block":
                            await self._log("output", rule.id, rule.name, "block", user_id, text[:100])
                            return GuardResult(
                                passed=False, triggered_rules=triggered, action="block",
                                message=rule.message or "응답에 부적절한 내용이 포함되어 필터링되었습니다.",
                            )
            except re.error:
                logger.warning("Invalid regex in output rule %s: %s", rule.name, rule.pattern)

        # PII masking on output (reuse existing detector)
        try:
            from pipeline.pii_detector import get_pii_detector
            pii_result = get_pii_detector().scan(modified[:50000])
            if pii_result.has_pii:
                modified = get_pii_detector().mask_text(modified).masked_text
                triggered.append("pii_auto_mask")
        except Exception:
            pass

        if triggered:
            await self._log("output", "system", ",".join(triggered), "mask", user_id, "")
            return GuardResult(
                passed=True, triggered_rules=triggered, action="mask",
                modified_text=modified,
            )

        return GuardResult(passed=True)

    # ==================== Logging ====================

    async def _log(self, direction: str, rule_id: str, rule_name: str,
                   action: str, user_id: str, snippet: str):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO guard_logs (id, timestamp, direction, rule_id, rule_name, action, user_id, snippet)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(),
                     direction, rule_id, rule_name, action, user_id, snippet[:100]),
                )
                await db.commit()
        except Exception as e:
            logger.warning("Failed to log guard event: %s", e)

    async def get_logs(self, limit: int = 50, direction: str | None = None) -> list[GuardLog]:
        await self._ensure_initialized()
        conditions = []
        params: list = []
        if direction:
            conditions.append("direction = ?")
            params.append(direction)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM guard_logs {where} ORDER BY timestamp DESC LIMIT ?",
                params + [limit],
            ) as cur:
                rows = await cur.fetchall()
                return [GuardLog(**dict(r)) for r in rows]


# Singleton
_manager: GuardrailsManager | None = None
_manager_lock = asyncio.Lock()


async def get_guardrails_manager() -> GuardrailsManager:
    global _manager
    if _manager is not None:
        return _manager
    async with _manager_lock:
        if _manager is None:
            _manager = GuardrailsManager()
            await _manager._ensure_initialized()
    return _manager
