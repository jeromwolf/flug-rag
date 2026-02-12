"""
PII (개인정보) 탐지 모듈

SFR-005: 문서 업로드 시 개인정보 자동 탐지 및 마스킹
- 한국 PII 패턴: 주민등록번호, 전화번호, 이메일 등
- 탐지 시 경고 생성 + 선택적 마스킹
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PIIMatch:
    """PII 탐지 결과"""
    pii_type: str          # "resident_id", "phone", "email", etc.
    value: str             # 마스킹된 값
    original_length: int   # 원본 길이
    position: tuple[int, int]  # (start, end) in text
    context: str           # 주변 텍스트 (앞뒤 20자)


@dataclass
class PIIScanResult:
    """PII 스캔 결과"""
    has_pii: bool = False
    match_count: int = 0
    matches: list[PIIMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    masked_text: str | None = None  # 마스킹된 텍스트 (옵션)


# PII patterns with labels
PII_PATTERNS: dict[str, re.Pattern] = {
    "resident_id": re.compile(r'\d{6}-[1-4]\d{6}'),
    "phone": re.compile(r'01[016789]-?\d{3,4}-?\d{4}'),
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    "bank_account": re.compile(r'\d{3,4}-\d{4,6}-\d{4,6}(?:-\d{1,3})?'),
    "passport": re.compile(r'[A-Z]{1,2}\d{7,8}'),
    "driver_license": re.compile(r'\d{2}-\d{2}-\d{6}-\d{2}'),
    "business_registration": re.compile(r'\d{3}-\d{2}-\d{5}'),
}

# Korean display names
PII_TYPE_LABELS: dict[str, str] = {
    "resident_id": "주민등록번호",
    "phone": "전화번호",
    "email": "이메일",
    "bank_account": "계좌번호",
    "passport": "여권번호",
    "driver_license": "운전면허번호",
    "business_registration": "사업자등록번호",
}


class PIIDetector:
    """PII 탐지기"""

    def __init__(self, patterns: dict[str, re.Pattern] | None = None):
        self.patterns = patterns or PII_PATTERNS

    def scan(self, text: str) -> PIIScanResult:
        """텍스트에서 PII 패턴 탐지.

        Args:
            text: 스캔할 텍스트

        Returns:
            PIIScanResult with matches and warnings
        """
        matches: list[PIIMatch] = []

        for pii_type, pattern in self.patterns.items():
            for m in pattern.finditer(text):
                masked_value = self._mask_value(pii_type, m.group())
                start, end = m.start(), m.end()

                # 주변 컨텍스트 (앞뒤 20자, PII 부분은 마스킹)
                ctx_start = max(0, start - 20)
                ctx_end = min(len(text), end + 20)
                context = text[ctx_start:start] + masked_value + text[end:ctx_end]

                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=masked_value,
                    original_length=len(m.group()),
                    position=(start, end),
                    context=context.strip(),
                ))

        # Deduplicate overlapping matches (keep the more specific type)
        if len(matches) > 1:
            matches.sort(key=lambda m: (m.position[0], -m.original_length))
            deduped = []
            last_end = -1
            for match in matches:
                if match.position[0] >= last_end:
                    deduped.append(match)
                    last_end = match.position[1]
            matches = deduped

        # Build warnings
        warnings = []
        if matches:
            # Count by type
            type_counts: dict[str, int] = {}
            for match in matches:
                label = PII_TYPE_LABELS.get(match.pii_type, match.pii_type)
                type_counts[label] = type_counts.get(label, 0) + 1

            for label, count in type_counts.items():
                warnings.append(f"PII 탐지: {label} {count}건")

        return PIIScanResult(
            has_pii=bool(matches),
            match_count=len(matches),
            matches=matches,
            warnings=warnings,
        )

    def mask_text(self, text: str) -> PIIScanResult:
        """텍스트에서 PII를 마스킹하여 반환.

        Args:
            text: 마스킹할 텍스트

        Returns:
            PIIScanResult with masked_text populated
        """
        result = self.scan(text)

        if not result.has_pii:
            result.masked_text = text
            return result

        # Sort matches by position in reverse order to replace from end
        sorted_matches = sorted(result.matches, key=lambda m: m.position[0], reverse=True)

        masked = text
        for match in sorted_matches:
            start, end = match.position
            masked = masked[:start] + match.value + masked[end:]

        result.masked_text = masked
        return result

    @staticmethod
    def _mask_value(pii_type: str, value: str) -> str:
        """PII 유형별 마스킹 규칙."""
        if pii_type == "resident_id":
            # 앞 6자리 유지, 뒤 마스킹: 920101-*******
            return value[:6] + "-*******"
        elif pii_type == "phone":
            # 뒤 4자리만 마스킹: 010-****-****
            clean = value.replace("-", "")
            return clean[:3] + "-****-" + "****"
        elif pii_type == "email":
            # 앞 2자리만 유지: ab***@domain.com
            parts = value.split("@")
            if len(parts) == 2:
                local = parts[0][:2] + "***"
                return f"{local}@{parts[1]}"
            return "***@***"
        elif pii_type == "bank_account":
            # 뒤 절반 마스킹
            half = len(value) // 2
            return value[:half] + "*" * (len(value) - half)
        elif pii_type == "passport":
            # 앞 1자리만 유지
            return value[0] + "*" * (len(value) - 1)
        elif pii_type == "driver_license":
            # 뒤 절반 마스킹
            half = len(value) // 2
            return value[:half] + "*" * (len(value) - half)
        elif pii_type == "business_registration":
            # 앞 3자리만 유지
            return value[:3] + "-**-*****"
        else:
            return "*" * len(value)


# Module-level singleton
_detector: PIIDetector | None = None


def get_pii_detector() -> PIIDetector:
    """PIIDetector 싱글톤 인스턴스 반환."""
    global _detector
    if _detector is None:
        _detector = PIIDetector()
    return _detector
