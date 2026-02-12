"""Upstage Document Parse 클라우드 API OCR 프로바이더.

기존 pipeline/ocr.py 코드를 새 구조로 이전.
"""

import logging
from pathlib import Path

import httpx

from config.settings import settings
from .base import BaseOCR, OCRResult

logger = logging.getLogger(__name__)


class UpstageCloudOCR(BaseOCR):
    """Upstage Document Parse 클라우드 API 클라이언트."""

    def __init__(self, api_key: str | None = None, api_url: str | None = None):
        self.api_key = api_key or settings.upstage_api_key
        self.api_url = api_url or settings.upstage_api_url
        self._client = httpx.AsyncClient(timeout=120.0)

    async def process(
        self,
        file_path: str | Path,
        enhanced: bool = False,
    ) -> OCRResult:
        """Upstage 클라우드 API로 문서 OCR 처리."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        output_formats = '["text", "html"]' if enhanced else '["text"]'

        with open(path, "rb") as f:
            response = await self._client.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"document": (path.name, f, "application/pdf")},
                data={"output_formats": output_formats},
            )

        response.raise_for_status()
        result = response.json()

        # 텍스트 추출: API v1 신규 구조 (content.text) + 레거시 (text) 지원
        text = result.get("text", "")
        if not text and isinstance(result.get("content"), dict):
            text = result["content"].get("text", "")

        # elements에서 텍스트 조합 (위 방법들이 모두 비어있을 때)
        elements = result.get("elements", [])
        tables = []
        if not text and elements:
            text_parts = []
            for elem in elements:
                category = elem.get("category", "")
                content = elem.get("content", {})
                if isinstance(content, dict):
                    elem_text = content.get("text", content.get("markdown", ""))
                else:
                    elem_text = str(content)
                if elem_text:
                    text_parts.append(elem_text)
                if category == "table":
                    tables.append(elem)
            text = "\n\n".join(text_parts)
        elif elements:
            tables = [e for e in elements if e.get("category") == "table"]

        # confidence: API v1 → ocr.confidence, 레거시 → confidence
        confidence = result.get("confidence", 0.0)
        if not confidence and isinstance(result.get("ocr"), dict):
            confidence = result["ocr"].get("confidence", 0.0)

        return OCRResult(
            text=text,
            confidence=confidence,
            pages=result.get("pages", []),
            tables=tables,
            layout=result.get("layout", {}),
            metadata={
                "provider": "upstage_cloud",
                "enhanced": enhanced,
                "api_url": self.api_url,
                "elements_count": len(elements),
            },
        )

    async def close(self):
        await self._client.aclose()
