"""이미지 분석 도구 — 이미지 내용을 분석하고 설명."""

import base64
import logging
from pathlib import Path

import httpx

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from config.settings import settings

logger = logging.getLogger(__name__)

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class VisionAnalyzerTool(BaseTool):
    """이미지 분석 도구 — 이미지 내용을 분석하고 설명"""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vision_analyzer",
            description="이미지 파일을 분석하여 내용을 설명합니다.",
            category="analysis",
            help_text=(
                "이미지 파일을 Vision LLM으로 분석하여 내용을 설명합니다.\n"
                "Upstage Solar Pro의 비전 기능을 사용합니다.\n"
                "파라미터:\n"
                "  - image_path: 이미지 파일 경로 (data 디렉토리 기준 상대경로)\n"
                "  - question: 이미지에 대한 질문 (기본: 내용 설명)\n"
                "  - language: 응답 언어 (ko/en)\n"
                "지원 형식: PNG, JPG, JPEG, GIF, WEBP"
            ),
            parameters=[
                ToolParameter(
                    name="image_path",
                    type=ToolParamType.STRING,
                    description="이미지 파일 경로 (data 디렉토리 기준 상대경로 또는 절대경로)",
                ),
                ToolParameter(
                    name="question",
                    type=ToolParamType.STRING,
                    description="이미지에 대한 질문",
                    required=False,
                    default="이 이미지의 내용을 상세히 설명하세요",
                ),
                ToolParameter(
                    name="language",
                    type=ToolParamType.STRING,
                    description="응답 언어",
                    required=False,
                    default="ko",
                    enum=["ko", "en"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        image_path_str = kwargs.get("image_path", "")
        if not image_path_str:
            return ToolResult(success=False, error="image_path 파라미터가 필요합니다.")

        question = kwargs.get("question", "이 이미지의 내용을 상세히 설명하세요")
        language = kwargs.get("language", "ko")

        # Resolve path
        image_path = Path(image_path_str)
        if not image_path.is_absolute():
            image_path = settings.data_dir / image_path_str

        if not image_path.exists():
            return ToolResult(
                success=False,
                error=f"이미지 파일을 찾을 수 없습니다: {image_path}",
            )

        # Check file extension
        ext = image_path.suffix.lower()
        mime = MIME_MAP.get(ext)
        if not mime:
            return ToolResult(
                success=False,
                error=f"지원하지 않는 이미지 형식입니다: {ext}. 지원 형식: {', '.join(MIME_MAP.keys())}",
            )

        # Check Upstage API key
        if not settings.upstage_api_key:
            return ToolResult(
                success=False,
                error=(
                    "Upstage API 키가 설정되지 않았습니다. "
                    "환경변수 UPSTAGE_API_KEY를 설정하세요."
                ),
            )

        try:
            # Read and encode image
            image_bytes = image_path.read_bytes()
            b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Add language instruction to question
            if language == "ko":
                full_question = f"한국어로 답변하세요. {question}"
            else:
                full_question = f"Answer in English. {question}"

            # Call Upstage Solar Pro vision
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.upstage.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.upstage_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "solar-pro",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{b64}",
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": full_question,
                                    },
                                ],
                            }
                        ],
                    },
                )
                response.raise_for_status()

            result = response.json()

            # Extract answer from response
            answer = ""
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice:
                    answer = choice["message"].get("content", "")
                elif "text" in choice:
                    answer = choice["text"]

            if not answer:
                return ToolResult(
                    success=False,
                    error="이미지 분석 결과를 받지 못했습니다.",
                )

            logger.info("이미지 분석 완료: %s", image_path.name)

            return ToolResult(
                success=True,
                data={
                    "analysis": answer,
                    "image_file": image_path.name,
                    "question": question,
                    "language": language,
                },
                metadata={
                    "image_path": str(image_path),
                    "mime_type": mime,
                    "file_size_kb": round(len(image_bytes) / 1024, 1),
                },
            )

        except httpx.HTTPStatusError as e:
            logger.error("Upstage Vision API HTTP 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"Upstage Vision API 오류 (HTTP {e.response.status_code}). API 키를 확인하세요.",
            )
        except httpx.RequestError as e:
            logger.error("Upstage Vision API 요청 실패: %s", e)
            return ToolResult(
                success=False,
                error=f"Upstage Vision API에 연결할 수 없습니다: {e}",
            )
        except Exception as e:
            logger.error("이미지 분석 중 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"이미지 분석 중 오류가 발생했습니다: {e}",
            )
