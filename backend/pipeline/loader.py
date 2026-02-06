"""Unified document loading orchestrator."""

from pathlib import Path

from .loaders import get_loader, LoadedDocument
from .ocr import UpstageOCR


class DocumentLoader:
    """Orchestrates document loading across all formats."""

    def __init__(self):
        self._ocr = None

    async def load(
        self,
        file_path: str | Path,
        apply_ocr: bool = False,
    ) -> LoadedDocument:
        """Load a document from any supported format.

        Args:
            file_path: Path to the file.
            apply_ocr: Force OCR processing (for scanned PDFs).

        Returns:
            LoadedDocument with extracted text and metadata.
        """
        path = Path(file_path)
        loader = get_loader(path.suffix)
        doc = await loader.load(path)

        # Auto-detect OCR need for PDFs
        if doc.metadata.get("needs_ocr") or apply_ocr:
            ocr_doc = await self._process_ocr(path)
            if ocr_doc:
                doc.content = ocr_doc["text"]
                doc.metadata["ocr_applied"] = True
                doc.metadata["ocr_confidence"] = ocr_doc["confidence"]

        return doc

    async def _process_ocr(self, file_path: Path) -> dict | None:
        """Process file with OCR if API key is available."""
        from config.settings import settings
        if not settings.upstage_api_key:
            return None

        if self._ocr is None:
            self._ocr = UpstageOCR()

        try:
            return await self._ocr.process(file_path)
        except Exception:
            return None
