"""PDF document loader using PyMuPDF."""

import asyncio
import re
from pathlib import Path

import fitz  # PyMuPDF

from .base import BaseLoader, LoadedDocument


class PDFLoader(BaseLoader):
    """Load PDF files using PyMuPDF (fitz)."""

    supported_extensions = [".pdf"]

    async def load(self, file_path: str | Path) -> LoadedDocument:
        path = self._validate_file(file_path)
        return await asyncio.to_thread(self._load_sync, path)

    def _load_sync(self, path: Path) -> LoadedDocument:
        doc = fitz.open(str(path))
        pages = []
        all_text_parts = []
        total_pages = len(doc)

        image_count = 0
        has_tables = False
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            images = page.get_images()

            image_count += len(images)

            # Detect tables: look for pipe characters or structured number patterns
            if not has_tables and text:
                if "|" in text or re.search(r"\d+\s+\d+\s+\d+", text):
                    has_tables = True

            if text.strip():
                pages.append({
                    "page_num": page_num + 1,
                    "content": text.strip(),
                })
                all_text_parts.append(text.strip())

        doc.close()

        # Compute text quality score
        full_text = "\n\n".join(all_text_parts)
        text_quality_score = self._compute_text_quality(full_text)

        # Improved needs_ocr logic
        needs_ocr = (len(all_text_parts) == 0 and image_count > 0) or text_quality_score < 0.3

        return LoadedDocument(
            content=full_text,
            metadata={
                "filename": path.name,
                "file_type": "pdf",
                "page_count": total_pages,
                "has_images": image_count > 0,
                "image_count": image_count,
                "has_tables": has_tables,
                "text_quality_score": text_quality_score,
                "needs_ocr": needs_ocr,
            },
            pages=pages,
        )

    def _compute_text_quality(self, text: str) -> float:
        """Compute text quality score based on ratio of Korean/English printable chars."""
        if not text:
            return 0.0

        total_chars = len(text)
        if total_chars == 0:
            return 0.0

        # Count Korean (Hangul), English letters, and common punctuation
        printable = 0
        for char in text:
            if (
                ("\uac00" <= char <= "\ud7a3")  # Korean Hangul
                or char.isalpha()  # English and other letters
                or char.isdigit()  # Numbers
                or char in " .,!?;:()\n\t-"  # Common punctuation/whitespace
            ):
                printable += 1

        return printable / total_chars
