"""
쿼리 교정 모듈
SFR-006: 유의어/오타 교정 + 도메인 전문용어 사전
"""
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """교정 결과"""
    original: str
    corrected: str
    corrections: list[dict]  # [{"original": "가스공사법", "corrected": "한국가스공사법", "type": "synonym"}, ...]
    was_corrected: bool = False


# 가스 분야 도메인 유의어 사전
DOMAIN_SYNONYMS: dict[str, str] = {
    # 조직명 약어/오타
    "가스기술공사": "한국가스기술공사",
    "가공": "한국가스공사",
    "가스공사": "한국가스공사",
    "KOGAS": "한국가스공사",
    "kogas": "한국가스공사",

    # 법률명
    "가스공사법": "한국가스공사법",
    "도시가스법": "도시가스사업법",
    "고압가스법": "고압가스 안전관리법",
    "LP가스법": "액화석유가스의 안전관리 및 사업법",
    "LPG법": "액화석유가스의 안전관리 및 사업법",

    # 기술 용어
    "LNG": "액화천연가스(LNG)",
    "LPG": "액화석유가스(LPG)",
    "CNG": "압축천연가스(CNG)",
    "배관망": "가스배관망",
    "정압기": "가스정압기",
    "안전밸브": "안전차단밸브",

    # 일반 약어
    "안관법": "안전관리법",
    "산안법": "산업안전보건법",
    "환경영향평가": "환경영향평가법",
}

# 한국어 일반 오타 교정 사전
TYPO_CORRECTIONS: dict[str, str] = {
    "가스관련": "가스 관련",
    "안전관리기준": "안전관리 기준",
    "검사기준": "검사 기준",
    "설비기준": "설비 기준",
    "운영기준": "운영 기준",
}


class QueryCorrector:
    """쿼리 교정기"""

    def __init__(
        self,
        synonyms: dict[str, str] | None = None,
        typos: dict[str, str] | None = None,
    ):
        self._synonyms = synonyms or DOMAIN_SYNONYMS
        self._typos = typos or TYPO_CORRECTIONS
        # Build pattern cache sorted by length (longest first for greedy match)
        self._all_patterns: list[tuple[str, str, str]] = []
        for orig, corrected in sorted(self._synonyms.items(), key=lambda x: -len(x[0])):
            self._all_patterns.append((orig, corrected, "synonym"))
        for orig, corrected in sorted(self._typos.items(), key=lambda x: -len(x[0])):
            self._all_patterns.append((orig, corrected, "typo"))

    def correct(self, query: str) -> CorrectionResult:
        """쿼리 교정 실행."""
        if not query or not query.strip():
            return CorrectionResult(original=query, corrected=query, corrections=[])

        corrected = query
        corrections = []
        placeholder_map = {}
        placeholder_counter = 0

        # Apply patterns in order (longest first), using placeholders to prevent overlap
        for original_pattern, replacement, correction_type in self._all_patterns:
            # Case-insensitive for English, exact for Korean
            matches = list(re.finditer(re.escape(original_pattern), corrected, re.IGNORECASE))
            if matches:
                applied = False
                # Replace matches with placeholders to prevent re-matching
                for match in reversed(matches):  # Reverse to preserve positions
                    # Skip if the match is already part of the replacement text
                    # e.g. don't replace "가스공사" in "한국가스공사" with "한국가스공사"
                    start = max(0, match.start() - len(replacement))
                    end = min(len(corrected), match.end() + len(replacement))
                    surrounding = corrected[start:end]
                    if replacement in surrounding:
                        continue

                    placeholder = f"__PLACEHOLDER_{placeholder_counter}__"
                    placeholder_map[placeholder] = replacement
                    placeholder_counter += 1
                    corrected = corrected[:match.start()] + placeholder + corrected[match.end():]
                    applied = True

                if applied:
                    corrections.append({
                        "original": original_pattern,
                        "corrected": replacement,
                        "type": correction_type,
                    })

        # Replace all placeholders with actual corrections
        for placeholder, replacement in placeholder_map.items():
            corrected = corrected.replace(placeholder, replacement)

        # Normalize whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()

        was_corrected = corrected != query
        if was_corrected:
            logger.debug("Query corrected: '%s' -> '%s'", query, corrected)

        return CorrectionResult(
            original=query,
            corrected=corrected,
            corrections=corrections,
            was_corrected=was_corrected,
        )

    def add_synonym(self, original: str, replacement: str):
        """동적으로 유의어 추가."""
        self._synonyms[original] = replacement
        self._rebuild_patterns()

    def add_typo(self, original: str, replacement: str):
        """동적으로 오타 교정 추가."""
        self._typos[original] = replacement
        self._rebuild_patterns()

    def _rebuild_patterns(self):
        self._all_patterns = []
        for orig, corrected in sorted(self._synonyms.items(), key=lambda x: -len(x[0])):
            self._all_patterns.append((orig, corrected, "synonym"))
        for orig, corrected in sorted(self._typos.items(), key=lambda x: -len(x[0])):
            self._all_patterns.append((orig, corrected, "typo"))

    def get_synonyms(self) -> dict[str, str]:
        return dict(self._synonyms)

    def get_typos(self) -> dict[str, str]:
        return dict(self._typos)


# Singleton
_corrector: QueryCorrector | None = None


def get_query_corrector() -> QueryCorrector:
    global _corrector
    if _corrector is None:
        _corrector = QueryCorrector()
    return _corrector
