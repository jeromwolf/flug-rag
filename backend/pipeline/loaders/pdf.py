"""PDF document loader using PyMuPDF."""

import asyncio
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

        has_images = False
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            images = page.get_images()

            if images:
                has_images = True

            if text.strip():
                pages.append({
                    "page_num": page_num + 1,
                    "content": text.strip(),
                })
                all_text_parts.append(text.strip())

        doc.close()

        needs_ocr = len(all_text_parts) == 0 and has_images

        return LoadedDocument(
            content="\n\n".join(all_text_parts),
            metadata={
                "filename": path.name,
                "file_type": "pdf",
                "page_count": total_pages,
                "has_images": has_images,
                "needs_ocr": needs_ocr,
            },
            pages=pages,
        )
