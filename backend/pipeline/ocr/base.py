"""OCR 프로바이더 추상화 인터페이스.

SFR-015: OCR 온프레미스 전환을 위한 공통 인터페이스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRResult:
    """OCR 처리 결과."""
    text: str = ""
    confidence: float = 0.0
    pages: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    layout: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class BaseOCR(ABC):
    """OCR 프로바이더 공통 인터페이스."""

    @abstractmethod
    async def process(
        self,
        file_path: str | Path,
        enhanced: bool = False,
    ) -> OCRResult:
        """문서 OCR 처리.

        Args:
            file_path: 처리할 파일 경로.
            enhanced: Enhanced Mode (복잡한 표/차트/다단 처리).

        Returns:
            OCRResult with extracted text and metadata.
        """

    async def close(self):
        """리소스 정리."""
