"""Unified document loading orchestrator."""

import logging
from pathlib import Path

from .loaders import get_loader, LoadedDocument
from .ocr import BaseOCR, create_ocr

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Orchestrates document loading across all formats."""

    def __init__(self):
        self._ocr: BaseOCR | None = None

    async def load(
        self,
        file_path: str | Path,
        apply_ocr: bool = False,
        enhanced_ocr: bool = False,
        dp_mode: str = "auto",
    ) -> LoadedDocument:
        """Load a document from any supported format.

        Args:
            file_path: Path to the file.
            apply_ocr: Force OCR processing (backward compatibility, acts like dp_mode="force_dp").
            enhanced_ocr: Enhanced Mode (복잡한 표/차트/다단).
            dp_mode: Document Parse mode - "auto", "force_dp", or "local_only".

        Returns:
            LoadedDocument with extracted text and metadata.
        """
        path = Path(file_path)
        loader = get_loader(path.suffix)
        doc = await loader.load(path)

        # Backward compatibility: apply_ocr=True acts like force_dp
        if apply_ocr:
            dp_mode = "force_dp"

        # Determine DP usage strategy
        dp_decision = self._should_use_dp(doc, dp_mode)
        logger.debug("DP decision for %s: %s", path.name, dp_decision)

        if dp_decision != "none":
            dp_result = await self._process_dp(path, enhanced=enhanced_ocr)
            if dp_result:
                if dp_decision == "full":
                    # Replace entire content with DP result
                    doc.content = dp_result.text
                    doc.metadata["dp_applied"] = True
                    doc.metadata["dp_mode"] = "full"
                elif dp_decision == "supplement":
                    # Keep local text, append table data if available
                    if dp_result.tables:
                        table_parts = []
                        for i, table in enumerate(dp_result.tables, 1):
                            if isinstance(table, dict):
                                content = table.get("content", {})
                                if isinstance(content, dict):
                                    text = content.get("markdown", content.get("text", ""))
                                else:
                                    text = str(content)
                            else:
                                text = str(table)
                            if text:
                                table_parts.append(f"[Table {i}]\n{text}")
                        if table_parts:
                            doc.content += "\n\n" + "\n\n".join(table_parts)
                    doc.metadata["dp_applied"] = True
                    doc.metadata["dp_mode"] = "supplement"

                doc.metadata["dp_confidence"] = dp_result.confidence
                doc.metadata["dp_provider"] = dp_result.metadata.get("provider", "")
                if dp_result.tables:
                    doc.metadata["dp_tables"] = len(dp_result.tables)

        return doc

    def _should_use_dp(self, doc: LoadedDocument, dp_mode: str) -> str:
        """Determine if and how to use Upstage Document Parse.

        Args:
            doc: Loaded document with metadata.
            dp_mode: User-specified mode - "auto", "force_dp", or "local_only".

        Returns:
            "none" - don't use DP
            "full" - use DP for full processing (replace local text)
            "supplement" - use DP to supplement (tables/images in readable PDFs)
        """
        if dp_mode == "local_only":
            return "none"

        if dp_mode == "force_dp":
            return "full"

        # Auto mode logic
        if doc.metadata.get("needs_ocr"):
            return "full"

        has_tables = doc.metadata.get("has_tables", False)
        image_count = doc.metadata.get("image_count", 0)

        if has_tables and image_count > 0:
            return "supplement"

        return "none"

    async def _process_dp(self, file_path: Path, enhanced: bool = False):
        """Process file with Upstage Document Parse."""
        from config.settings import settings
        if not settings.upstage_api_key and settings.ocr_provider == "cloud":
            return None

        if self._ocr is None:
            self._ocr = create_ocr()

        try:
            result = await self._ocr.process(file_path, enhanced=enhanced)

            # 학습 데이터 수집 (non-blocking)
            try:
                from config.settings import settings
                if settings.ocr_training_enabled:
                    from pipeline.ocr.training_collector import get_training_collector
                    collector = await get_training_collector()
                    await collector.collect(
                        file_path=file_path,
                        ocr_result=result,
                        filename=file_path.name,
                        enhanced=enhanced,
                    )
            except Exception as e:
                logger.debug("OCR training data collection skipped: %s", e)

            return result
        except Exception as e:
            logger.warning("Document Parse processing failed for %s: %s", file_path.name, e)
            return None
