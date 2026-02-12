"""OCR 프로바이더 패키지.

SFR-015: Strategy + Factory 패턴으로 OCR 프로바이더 관리
- upstage_cloud: Upstage 클라우드 API (기존)
- upstage_onprem: Upstage 온프레미스 Docker (신규)
"""

from .base import BaseOCR, OCRResult
from .upstage_cloud import UpstageCloudOCR
from .upstage_onprem import UpstageOnpremOCR

# 하위 호환성: 기존 `from pipeline.ocr import UpstageOCR`
UpstageOCR = UpstageCloudOCR

__all__ = [
    "BaseOCR",
    "OCRResult",
    "UpstageCloudOCR",
    "UpstageOnpremOCR",
    "UpstageOCR",
    "create_ocr",
]


def create_ocr(provider: str | None = None) -> BaseOCR:
    """OCR 프로바이더 팩토리.

    Args:
        provider: "cloud" 또는 "onprem". None이면 설정값 사용.

    Returns:
        BaseOCR 구현체 인스턴스.
    """
    from config.settings import settings

    provider = provider or settings.ocr_provider

    if provider == "onprem":
        return UpstageOnpremOCR()
    elif provider == "cloud":
        return UpstageCloudOCR()
    else:
        raise ValueError(f"Unknown OCR provider: {provider}. Use 'cloud' or 'onprem'.")
