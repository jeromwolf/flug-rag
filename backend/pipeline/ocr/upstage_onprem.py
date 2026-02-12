"""Upstage Document Parse 온프레미스 OCR 프로바이더.

SFR-015: 로컬 Docker 컨테이너로 실행되는 Upstage DP 서버 연동.
- Enhanced Mode 지원 (복잡한 표, 차트, 다단 레이아웃)
- 네트워크 격리 환경(폐쇄망)에서 운영 가능
"""

import logging
from pathlib import Path

import httpx

from config.settings import settings
from .base import BaseOCR, OCRResult

logger = logging.getLogger(__name__)


class UpstageOnpremOCR(BaseOCR):
    """Upstage Document Parse 온프레미스 클라이언트.

    Docker 컨테이너로 실행되는 Upstage DP API를 호출합니다.
    기본 URL: http://localhost:8501
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.ocr_onprem_url).rstrip("/")
        self._client = httpx.AsyncClient(timeout=180.0)

    async def process(
        self,
        file_path: str | Path,
        enhanced: bool = False,
    ) -> OCRResult:
        """온프레미스 Upstage DP로 문서 OCR 처리.

        Args:
            file_path: 처리할 파일 경로.
            enhanced: Enhanced Mode - 복잡한 표/차트/다단 처리.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # 온프레미스 API 엔드포인트
        endpoint = f"{self.base_url}/v1/document-ai/document-parse"

        # Enhanced Mode 설정
        output_formats = '["text", "html"]' if enhanced else '["text"]'
        data = {
            "output_formats": output_formats,
        }
        if enhanced:
            data["use_ocr"] = "true"
            data["coordinates"] = "true"
            data["model"] = "document-parse-240910"

        with open(path, "rb") as f:
            response = await self._client.post(
                endpoint,
                files={"document": (path.name, f, "application/pdf")},
                data=data,
            )

        response.raise_for_status()
        result = response.json()

        # 테이블 추출
        tables = []
        if "elements" in result:
            tables = [
                {
                    "category": elem.get("category"),
                    "content": elem.get("content", {}).get("html", ""),
                    "coordinates": elem.get("coordinates"),
                    "page": elem.get("page"),
                }
                for elem in result["elements"]
                if elem.get("category") == "table"
            ]

        # 레이아웃 정보 (다단 감지)
        layout = {}
        if "elements" in result:
            categories = [elem.get("category") for elem in result["elements"]]
            layout = {
                "total_elements": len(result["elements"]),
                "tables": categories.count("table"),
                "figures": categories.count("figure"),
                "headers": categories.count("header"),
                "paragraphs": categories.count("paragraph"),
            }

        return OCRResult(
            text=result.get("text", ""),
            confidence=result.get("confidence", 0.0),
            pages=result.get("pages", []),
            tables=tables,
            layout=layout,
            metadata={
                "provider": "upstage_onprem",
                "enhanced": enhanced,
                "base_url": self.base_url,
                "model": data.get("model", "default"),
            },
        )

    async def health_check(self) -> bool:
        """온프레미스 서버 상태 확인."""
        try:
            response = await self._client.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()
