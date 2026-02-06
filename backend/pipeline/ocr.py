"""OCR processing using Upstage Document Parse API."""

import asyncio
from pathlib import Path

import httpx

from config.settings import settings


class UpstageOCR:
    """Upstage Document Parse API client for OCR."""

    def __init__(self, api_key: str | None = None, api_url: str | None = None):
        self.api_key = api_key or settings.upstage_api_key
        self.api_url = api_url or settings.upstage_api_url
        self._client = httpx.AsyncClient(timeout=120.0)

    async def process(self, file_path: str | Path) -> dict:
        """Process a document with Upstage OCR.

        Returns:
            dict with keys: text, confidence, pages
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "rb") as f:
            response = await self._client.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"document": (path.name, f, "application/pdf")},
                data={"output_formats": '["text"]'},
            )

        response.raise_for_status()
        result = response.json()

        return {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "pages": result.get("pages", []),
            "raw": result,
        }

    async def close(self):
        await self._client.aclose()
