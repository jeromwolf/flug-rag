"""Upstage Universal Extraction tool — extract structured data from documents/images."""

import base64
import json
import mimetypes
from pathlib import Path

import httpx

from agent.mcp.tools.base import (
    BaseTool,
    ToolDefinition,
    ToolParameter,
    ToolParamType,
    ToolResult,
)
from config.settings import settings


# MIME type map for common document/image extensions
_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".hwp": "application/x-hwp",
    ".txt": "text/plain",
}

_UPSTAGE_CHAT_URL = "https://api.upstage.ai/v1/chat/completions"
_DEFAULT_MIME = "image/png"


def _detect_mime(path: str) -> str:
    """Detect MIME type from file extension."""
    ext = Path(path).suffix.lower()
    if ext in _MIME_MAP:
        return _MIME_MAP[ext]
    guessed, _ = mimetypes.guess_type(path)
    return guessed or _DEFAULT_MIME


class UpstageExtractTool(BaseTool):
    """Upstage Universal Extraction — 문서/이미지에서 구조화 데이터 추출."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="upstage_extract",
            description="문서 또는 이미지에서 구조화된 데이터를 추출합니다 (Upstage Universal Extraction)",
            category="extraction",
            help_text=(
                "Upstage Universal Extraction API를 사용하여 문서나 이미지에서\n"
                "원하는 필드를 JSON Schema 형식으로 정의하고 구조화된 데이터를 추출합니다.\n\n"
                "사용 예시:\n"
                "  source: 'data/invoice.png'\n"
                "  schema: {\"type\": \"object\", \"properties\": {\"invoice_no\": {\"type\": \"string\"}, "
                "\"total\": {\"type\": \"number\"}}}\n"
                "  source_type: 'file'  (기본값)\n\n"
                "source_type 옵션:\n"
                "  - file   : data/ 디렉토리의 파일 경로 (자동 base64 인코딩)\n"
                "  - base64 : 이미 base64로 인코딩된 이미지 데이터\n\n"
                "지원 파일 형식: PNG, JPG, GIF, WebP, BMP, TIFF, PDF, DOCX, XLSX, PPTX, HWP\n"
                "UPSTAGE_API_KEY 환경 변수가 설정되어 있어야 합니다."
            ),
            parameters=[
                ToolParameter(
                    name="source",
                    type=ToolParamType.STRING,
                    description="추출할 파일 경로 (data/ 디렉토리) 또는 base64 인코딩 이미지",
                    required=True,
                ),
                ToolParameter(
                    name="schema",
                    type=ToolParamType.OBJECT,
                    description="추출할 데이터 구조 (JSON Schema 형식)",
                    required=True,
                ),
                ToolParameter(
                    name="source_type",
                    type=ToolParamType.STRING,
                    description="소스 유형: 'file' (파일 경로) 또는 'base64' (base64 데이터)",
                    required=False,
                    default="file",
                    enum=["file", "base64"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        # --- validate API key ---
        if not settings.upstage_api_key:
            return ToolResult(
                success=False,
                error="Upstage API key가 설정되지 않았습니다. UPSTAGE_API_KEY 환경 변수를 설정하세요.",
            )

        source: str = kwargs.get("source", "").strip()
        schema: dict = kwargs.get("schema", {})
        source_type: str = kwargs.get("source_type", "file")

        if not source:
            return ToolResult(success=False, error="source 파라미터가 필요합니다.")
        if not schema:
            return ToolResult(success=False, error="schema 파라미터가 필요합니다.")

        # --- prepare base64 data and MIME type ---
        try:
            if source_type == "file":
                b64_data, mime_type = self._load_file(source)
            else:
                b64_data = source
                mime_type = _DEFAULT_MIME
        except FileNotFoundError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"파일 로드 오류: {exc}")

        # --- call Upstage Universal Extraction API ---
        try:
            raw_content = await self._call_upstage(b64_data, mime_type, schema)
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Upstage API 요청 시간 초과 (60초)")
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                error=f"Upstage API 오류 (HTTP {exc.response.status_code}): {exc.response.text[:500]}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Upstage API 호출 실패: {exc}")

        # --- parse JSON response ---
        extracted = self._parse_json(raw_content)

        formatted = self._format_extraction_result(extracted, schema)

        return ToolResult(
            success=True,
            data={
                "extracted": extracted,
                "formatted": formatted,
                "source": source,
                "source_type": source_type,
            },
            metadata={
                "mime_type": mime_type,
                "schema_fields": list(schema.get("properties", {}).keys()),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, source: str) -> tuple[str, str]:
        """Read file from disk and return (base64_string, mime_type)."""
        path = Path(source)

        # Resolve relative paths against the data directory
        if not path.is_absolute():
            candidate = settings.data_dir / path
            if candidate.exists():
                path = candidate

        if not path.exists():
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {source} "
                f"(data/ 디렉토리 기준 경로를 사용하세요)"
            )

        mime_type = _detect_mime(str(path))
        raw_bytes = path.read_bytes()
        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        return b64_data, mime_type

    async def _call_upstage(
        self,
        b64_data: str,
        mime_type: str,
        schema: dict,
    ) -> str:
        """POST to Upstage chat completions endpoint and return raw content string."""
        schema_str = json.dumps(schema, ensure_ascii=False)
        payload = {
            "model": "information-extract",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract the following structured information from this "
                                "document/image according to this JSON schema:\n"
                                f"{schema_str}\n\n"
                                "Return ONLY valid JSON matching the schema."
                            ),
                        },
                    ],
                }
            ],
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                _UPSTAGE_CHAT_URL,
                headers={"Authorization": f"Bearer {settings.upstage_api_key}"},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Upstage 응답 형식이 올바르지 않습니다: {result}") from exc

    @staticmethod
    def _parse_json(raw: str) -> dict | list:
        """Try to parse raw string as JSON; fall back to wrapped dict."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_content": raw}

    @staticmethod
    def _format_extraction_result(data: dict | list, schema: dict) -> str:
        """Format extracted data as a markdown table or list."""
        if not data:
            return "_추출된 데이터가 없습니다._"

        lines: list[str] = ["## 추출 결과\n"]

        if isinstance(data, list):
            if not data:
                return "_추출된 데이터가 없습니다._"
            # Render each item as a sub-section
            for idx, item in enumerate(data, 1):
                lines.append(f"### 항목 {idx}\n")
                if isinstance(item, dict):
                    lines.append("| 필드 | 값 |")
                    lines.append("|------|-----|")
                    for k, v in item.items():
                        lines.append(f"| {k} | {v} |")
                    lines.append("")
                else:
                    lines.append(f"- {item}\n")
            return "\n".join(lines)

        if isinstance(data, dict):
            raw = data.get("raw_content")
            if raw is not None:
                # Parsing failed — show raw output
                lines.append("> JSON 파싱 실패. 원본 응답:\n")
                lines.append(f"```\n{raw}\n```")
                return "\n".join(lines)

            # Build markdown table from schema property order when available
            props = schema.get("properties", {})
            keys = list(props.keys()) if props else list(data.keys())
            # Include any extra keys returned by the API
            for k in data:
                if k not in keys:
                    keys.append(k)

            lines.append("| 필드 | 값 |")
            lines.append("|------|-----|")
            for key in keys:
                value = data.get(key, "—")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                lines.append(f"| {key} | {value} |")
            lines.append("")
            return "\n".join(lines)

        # Fallback: JSON dump
        lines.append(f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```")
        return "\n".join(lines)
