"""ISO/KS standard document loader with structure detection."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import fitz  # PyMuPDF

from .base import BaseLoader, LoadedDocument


class ISOLoader(BaseLoader):
    """Load ISO/KS standard documents (PDF) with structure detection.

    Detects ISO-specific patterns:
    - Standard numbers: ISO 12345:2020, KS B 6211:2019
    - Section numbering: 4.1.2 Requirements for...
    - Standard sections: Scope, Normative references, Terms and definitions
    - Annexes: Annex A (normative)
    """

    supported_extensions = [".pdf"]

    # ISO/KS standard number patterns
    ISO_NUMBER_RE = re.compile(
        r'(?:ISO|IEC|KS\s*[A-Z](?:\s*[A-Z])?(?:\s*IEC)?)\s*[\d\-]+(?::[\d]{4})?',
        re.IGNORECASE
    )

    # Section heading patterns
    SECTION_RE = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
    ANNEX_RE = re.compile(
        r'^(?:Annex|부속서|별표|附属書)\s+([A-Z])\s*(?:\(.*?\))?\s*(.*)$',
        re.IGNORECASE
    )

    # Standard ISO section keywords
    STANDARD_SECTIONS = {
        "scope": ["scope", "범위", "適用範囲"],
        "normative_references": ["normative references", "인용표준", "引用規格"],
        "terms": ["terms and definitions", "용어", "용어 및 정의", "用語及び定義"],
        "requirements": ["requirements", "요구사항", "要求事項"],
        "test_methods": ["test methods", "시험방법", "試験方法"],
        "annex": ["annex", "부속서", "별표", "附属書"],
    }

    async def load(self, file_path: str | Path) -> LoadedDocument:
        path = self._validate_file(file_path)
        return await asyncio.to_thread(self._load_sync, path)

    def _load_sync(self, path: Path) -> LoadedDocument:
        """Load PDF and detect ISO structure."""
        doc = fitz.open(str(path))
        pages = []
        all_text_parts = []
        total_pages = len(doc)

        # Extract text from all pages
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            if text.strip():
                pages.append({
                    "page_num": page_num + 1,
                    "content": text.strip(),
                })
                all_text_parts.append(text.strip())

        doc.close()

        full_text = "\n\n".join(all_text_parts)

        # Try to detect ISO structure
        iso_metadata = self._detect_iso_structure(full_text, path.name)

        if iso_metadata["is_iso_standard"]:
            # Parse sections for structured pages
            structured_pages = self._parse_iso_sections(full_text)
            if structured_pages:
                pages = structured_pages

        # Merge base PDF metadata with ISO metadata
        metadata = {
            "filename": path.name,
            "file_type": "pdf",
            "page_count": total_pages,
            **iso_metadata,
        }

        return LoadedDocument(
            content=full_text,
            metadata=metadata,
            pages=pages,
        )

    def _detect_iso_structure(self, text: str, filename: str) -> dict:
        """Detect if document is an ISO/KS standard and extract metadata."""
        metadata = {
            "is_iso_standard": False,
            "standard_number": None,
            "standard_title": None,
            "standard_year": None,
        }

        # Search in first 3000 characters (title page)
        search_text = text[:3000]

        # Find standard number
        match = self.ISO_NUMBER_RE.search(search_text)
        if not match:
            # Also check filename
            match = self.ISO_NUMBER_RE.search(filename)

        if match:
            metadata["is_iso_standard"] = True
            standard_number = match.group(0)
            metadata["standard_number"] = standard_number

            # Extract year from standard number
            year_match = re.search(r':(\d{4})', standard_number)
            if year_match:
                metadata["standard_year"] = year_match.group(1)

            # Try to find title (usually the line after standard number)
            lines = search_text.split('\n')
            for i, line in enumerate(lines):
                if standard_number in line and i + 1 < len(lines):
                    potential_title = lines[i + 1].strip()
                    # Filter out empty lines and section numbers
                    if potential_title and not self.SECTION_RE.match(potential_title):
                        metadata["standard_title"] = potential_title[:200]
                    break

        # Check if document has ISO-style section structure
        if not metadata["is_iso_standard"]:
            # Look for numbered sections (e.g., "4.1.2 ...")
            section_matches = self.SECTION_RE.findall(text[:5000], re.MULTILINE)
            if len(section_matches) >= 3:
                metadata["is_iso_standard"] = True

        return metadata

    def _parse_iso_sections(self, text: str) -> list[dict]:
        """Parse ISO document into structured sections."""
        lines = text.split('\n')
        sections = []
        current_section = None
        buffer = []

        for line in lines:
            # Check for section heading
            section_match = self.SECTION_RE.match(line.strip())
            annex_match = self.ANNEX_RE.match(line.strip())

            if section_match or annex_match:
                # Save previous section
                if current_section is not None and buffer:
                    current_section["content"] = "\n".join(buffer).strip()
                    if current_section["content"]:
                        sections.append(current_section)

                # Start new section
                if section_match:
                    section_num, section_title = section_match.groups()
                    current_section = {
                        "page_num": len(sections) + 1,
                        "content": "",
                        "section_number": section_num,
                        "section_title": section_title.strip(),
                        "section_type": self._classify_section(section_title),
                    }
                else:  # annex_match
                    annex_id, annex_title = annex_match.groups()
                    current_section = {
                        "page_num": len(sections) + 1,
                        "content": "",
                        "section_number": f"Annex {annex_id}",
                        "section_title": annex_title.strip() if annex_title else f"Annex {annex_id}",
                        "section_type": "annex",
                    }

                buffer = [line]
            else:
                buffer.append(line)

        # Save last section
        if current_section is not None and buffer:
            current_section["content"] = "\n".join(buffer).strip()
            if current_section["content"]:
                sections.append(current_section)

        # If no sections detected, return empty list (will use original pages)
        if len(sections) < 2:
            return []

        return sections

    def _classify_section(self, title: str) -> str:
        """Classify section type based on title keywords."""
        title_lower = title.lower()

        for section_type, keywords in self.STANDARD_SECTIONS.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return section_type

        return "general"
