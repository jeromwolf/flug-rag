"""Automatic metadata extraction from documents."""

import re
from datetime import datetime
from pathlib import Path


class MetadataExtractor:
    """Extract metadata from document text and filename."""

    # Korean department name patterns
    DEPARTMENT_PATTERNS = [
        r'(인사부|인사팀|총무부|총무팀)',
        r'(재무부|재무팀|회계팀|경리팀)',
        r'(기획부|기획팀|전략기획)',
        r'(기술부|기술팀|기술연구소|연구개발)',
        r'(안전부|안전팀|안전관리|가스안전)',
        r'(시설부|시설팀|시설관리|설비팀)',
        r'(영업부|영업팀|사업팀|사업부)',
        r'(품질부|품질팀|품질관리)',
        r'(정보보안|보안팀|IT팀|정보시스템)',
        r'(교육팀|교육훈련|인재개발)',
    ]

    # Document category patterns
    CATEGORY_PATTERNS = {
        "규정": r'(규정|규칙|내규|조례|법규|지침)',
        "매뉴얼": r'(매뉴얼|사용자?\s*가이드|안내서|지침서|절차서)',
        "보고서": r'(보고서|리포트|결과서|분석서|평가서)',
        "계획서": r'(계획서|계획안|사업계획|추진계획)',
        "공문": r'(공문|통보|통지|안내문|회신)',
        "교육": r'(교육|훈련|연수|세미나|워크숍)',
        "점검": r'(점검|검사|진단|감사|심사)',
    }

    # Date patterns
    DATE_PATTERNS = [
        r'(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})[일]?',
        r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        r'(\d{4})[-./](\d{1,2})',  # YYYY-MM
    ]

    def extract(self, text: str, filename: str = "", file_path: str = "") -> dict:
        """Extract metadata from document text and filename.

        Returns:
            dict with keys: department, category, dates, tags, title
        """
        metadata = {}

        # Extract from filename
        if filename:
            metadata.update(self._extract_from_filename(filename))

        # Extract from text content
        if text:
            text_meta = self._extract_from_text(text[:5000])  # First 5000 chars
            # Merge (text-based fills gaps)
            for key, value in text_meta.items():
                if key not in metadata or not metadata[key]:
                    metadata[key] = value

        return metadata

    def _extract_from_filename(self, filename: str) -> dict:
        """Extract metadata from filename."""
        name = Path(filename).stem
        metadata = {}

        # Department from filename
        for pattern in self.DEPARTMENT_PATTERNS:
            match = re.search(pattern, name)
            if match:
                metadata["department"] = match.group(1)
                break

        # Category from filename
        for category, pattern in self.CATEGORY_PATTERNS.items():
            if re.search(pattern, name):
                metadata["category"] = category
                break

        # Date from filename
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, name)
            if match:
                groups = match.groups()
                try:
                    if len(groups) == 3:
                        metadata["document_date"] = f"{groups[0]}-{int(groups[1]):02d}-{int(groups[2]):02d}"
                    elif len(groups) == 2:
                        metadata["document_date"] = f"{groups[0]}-{int(groups[1]):02d}"
                except (ValueError, IndexError):
                    pass
                break

        # Title guess: clean filename
        title = re.sub(r'[\d_\-]+', ' ', name).strip()
        if title:
            metadata["title"] = title

        return metadata

    def _extract_from_text(self, text: str) -> dict:
        """Extract metadata from document text content."""
        metadata = {}

        # Department from text
        for pattern in self.DEPARTMENT_PATTERNS:
            match = re.search(pattern, text)
            if match:
                metadata["department"] = match.group(1)
                break

        # Category from text
        for category, pattern in self.CATEGORY_PATTERNS.items():
            if re.search(pattern, text):
                metadata["category"] = category
                break

        # Dates from text
        dates = set()
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text):
                groups = match.groups()
                try:
                    if len(groups) == 3:
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        if 1990 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                            dates.add(f"{year}-{month:02d}-{day:02d}")
                    elif len(groups) == 2:
                        year, month = int(groups[0]), int(groups[1])
                        if 1990 <= year <= 2030 and 1 <= month <= 12:
                            dates.add(f"{year}-{month:02d}")
                except (ValueError, IndexError):
                    continue

        if dates:
            sorted_dates = sorted(dates)
            metadata["document_date"] = sorted_dates[-1]  # Most recent
            # ChromaDB doesn't support list values, join as comma-separated string
            metadata["dates_found"] = ",".join(sorted_dates)

        # Tags: extract key terms (simple approach)
        tags = set()
        tag_patterns = [
            r'가스', r'안전', r'배관', r'점검', r'설비',
            r'비상', r'대응', r'교육', r'훈련', r'예산',
            r'계획', r'보고', r'규정', r'관리', r'운영',
        ]
        for pattern in tag_patterns:
            if re.search(pattern, text):
                tags.add(pattern)

        if tags:
            # ChromaDB doesn't support list values, join as comma-separated string
            metadata["tags"] = ",".join(sorted(tags))

        return metadata
