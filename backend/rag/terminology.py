"""
가스 기술용어 사전 서비스
SFR: 기술용어 사전 연동 - 동의어/유의어 확장 검색
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TermEntry:
    """용어 사전 엔트리"""
    term: str
    synonyms: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    english: str = ""
    category: str = ""
    definition: str = ""


@dataclass
class ExpansionResult:
    """쿼리 확장 결과"""
    original_query: str
    expanded_query: str
    matched_terms: list[str]
    expansions: list[dict]  # [{"term": "...", "synonyms_added": [...]}]
    was_expanded: bool = False


class TerminologyService:
    """가스 기술용어 사전 서비스 - 쿼리 확장 및 용어 조회"""

    def __init__(self, glossary_path: str | Path | None = None):
        self._entries: list[TermEntry] = []
        self._term_map: dict[str, TermEntry] = {}  # term/synonym -> entry (lowercase)
        self._loaded = False
        self._glossary_path = glossary_path or (
            settings.data_dir / "glossary" / "gas_terminology.json"
        )
        self._load()

    def _load(self):
        """Load glossary from JSON file."""
        path = Path(self._glossary_path)
        if not path.exists():
            logger.warning("Glossary file not found: %s", path)
            self._loaded = False
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data if isinstance(data, list) else data.get("terms", [])

            for item in entries:
                entry = TermEntry(
                    term=item.get("term", ""),
                    synonyms=item.get("synonyms", []),
                    related=item.get("related", []),
                    english=item.get("english", ""),
                    category=item.get("category", ""),
                    definition=item.get("definition", ""),
                )
                self._entries.append(entry)

                # Index by term and all synonyms (case-insensitive for English)
                for key in [entry.term] + entry.synonyms:
                    key_lower = key.lower().strip()
                    if key_lower:
                        self._term_map[key_lower] = entry

            self._loaded = True
            logger.info("Loaded %d terminology entries from %s", len(self._entries), path)
        except Exception as e:
            logger.warning("Failed to load glossary: %s", e)
            self._loaded = False

    def lookup(self, term: str) -> TermEntry | None:
        """용어 조회."""
        if not self._loaded:
            return None
        return self._term_map.get(term.lower().strip())

    def expand_query(self, query: str) -> ExpansionResult:
        """쿼리에 포함된 전문용어의 동의어로 확장.

        Example:
            "정압기 설치기준" -> "정압기 가스정압기 정압장치 레귤레이터 설치기준"
        """
        if not self._loaded or not query.strip():
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                matched_terms=[],
                expansions=[],
            )

        matched_terms = []
        expansions = []
        added_words = set()
        query_lower = query.lower()

        # Check each term/synonym against query
        # Sort by length (longest first) to match specific terms first
        checked_entries = set()
        for key in sorted(self._term_map.keys(), key=len, reverse=True):
            if key in query_lower:
                entry = self._term_map[key]
                if id(entry) in checked_entries:
                    continue
                checked_entries.add(id(entry))

                matched_terms.append(entry.term)
                synonyms_to_add = []

                # Add synonyms not already in query
                for syn in [entry.term] + entry.synonyms:
                    syn_lower = syn.lower().strip()
                    if syn_lower and syn_lower not in query_lower and syn_lower not in added_words:
                        synonyms_to_add.append(syn)
                        added_words.add(syn_lower)

                if synonyms_to_add:
                    expansions.append({
                        "term": entry.term,
                        "synonyms_added": synonyms_to_add,
                    })

        # Build expanded query: original + added synonyms
        if expansions:
            extra_terms = []
            for exp in expansions:
                extra_terms.extend(exp["synonyms_added"])
            expanded_query = f"{query} {' '.join(extra_terms)}"
        else:
            expanded_query = query

        was_expanded = expanded_query != query
        if was_expanded:
            logger.debug("Query expanded: '%s' -> '%s'", query[:50], expanded_query[:80])

        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            matched_terms=matched_terms,
            expansions=expansions,
            was_expanded=was_expanded,
        )

    def get_all_terms(self) -> list[TermEntry]:
        """전체 용어 목록 반환."""
        return list(self._entries)

    def get_categories(self) -> list[str]:
        """카테고리 목록 반환."""
        return sorted(set(e.category for e in self._entries if e.category))

    def search_terms(self, keyword: str) -> list[TermEntry]:
        """키워드로 용어 검색."""
        keyword = keyword.lower()
        results = []
        for entry in self._entries:
            if (keyword in entry.term.lower()
                or keyword in entry.english.lower()
                or keyword in entry.definition.lower()
                or any(keyword in s.lower() for s in entry.synonyms)):
                results.append(entry)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# Singleton
_service: TerminologyService | None = None


def get_terminology_service() -> TerminologyService:
    global _service
    if _service is None:
        _service = TerminologyService()
    return _service
