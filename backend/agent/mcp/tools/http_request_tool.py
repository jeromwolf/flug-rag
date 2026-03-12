"""Generic HTTP request tool for external REST API calls."""

import json
import re
from urllib.parse import urlparse

import httpx

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)

# SSRF prevention: block internal network ranges
_BLOCKED_HOSTS = re.compile(
    r"^("
    r"localhost|"
    r"127\.\d+\.\d+\.\d+|"
    r"10\.\d+\.\d+\.\d+|"
    r"192\.168\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|"
    r"0\.0\.0\.0|"
    r"\[::1\]|"
    r"\[::\]"
    r")$",
    re.IGNORECASE,
)

MAX_RESPONSE_LENGTH = 5000


class HttpRequestTool(BaseTool):
    """범용 HTTP 요청 도구 -- 외부 REST API 호출"""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="http_request",
            description="외부 REST API를 호출합니다.",
            category="integration",
            help_text=(
                "외부 REST API를 호출합니다. "
                "공공데이터포털, 기상청 API 등 외부 서비스 연동에 사용합니다."
            ),
            parameters=[
                ToolParameter(
                    name="url",
                    type=ToolParamType.STRING,
                    description="요청할 URL (https://... )",
                ),
                ToolParameter(
                    name="method",
                    type=ToolParamType.STRING,
                    description="HTTP 메서드",
                    required=False,
                    default="GET",
                    enum=["GET", "POST", "PUT", "DELETE"],
                ),
                ToolParameter(
                    name="headers",
                    type=ToolParamType.OBJECT,
                    description="요청 헤더 (예: {\"Authorization\": \"Bearer ...\"})",
                    required=False,
                ),
                ToolParameter(
                    name="body",
                    type=ToolParamType.OBJECT,
                    description="요청 본문 (POST/PUT 시 JSON 객체)",
                    required=False,
                ),
                ToolParameter(
                    name="timeout",
                    type=ToolParamType.INTEGER,
                    description="타임아웃 (초, 기본값: 30)",
                    required=False,
                    default=30,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        method = kwargs.get("method", "GET").upper()
        headers = kwargs.get("headers") or {}
        body = kwargs.get("body")
        timeout = kwargs.get("timeout", 30)

        if not url:
            return ToolResult(success=False, error="url parameter is required")

        # Validate URL
        try:
            parsed = urlparse(url)
        except Exception:
            return ToolResult(success=False, error=f"Invalid URL: {url}")

        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                success=False,
                error=f"Unsupported scheme: {parsed.scheme}. Only http/https allowed.",
            )

        # SSRF prevention
        hostname = parsed.hostname or ""
        if _BLOCKED_HOSTS.match(hostname):
            return ToolResult(
                success=False,
                error="Blocked: requests to internal/private network addresses are not allowed.",
            )

        # Clamp timeout
        timeout = max(1, min(timeout, 60))

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                max_redirects=5,
            ) as client:
                request_kwargs: dict = {
                    "method": method,
                    "url": url,
                    "headers": headers,
                }
                if body is not None and method in ("POST", "PUT"):
                    request_kwargs["json"] = body

                response = await client.request(**request_kwargs)

            # Parse response
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")

            # Try JSON first
            response_data = None
            response_text = None
            if "json" in content_type or "javascript" in content_type:
                try:
                    response_data = response.json()
                except (json.JSONDecodeError, ValueError):
                    response_text = response.text[:MAX_RESPONSE_LENGTH]
            else:
                response_text = response.text[:MAX_RESPONSE_LENGTH]

            result_data = {
                "status_code": status_code,
                "content_type": content_type,
            }
            if response_data is not None:
                result_data["data"] = response_data
            else:
                result_data["text"] = response_text
                if len(response.text) > MAX_RESPONSE_LENGTH:
                    result_data["truncated"] = True
                    result_data["original_length"] = len(response.text)

            is_success = 200 <= status_code < 400

            return ToolResult(
                success=is_success,
                data=result_data,
                error=None if is_success else f"HTTP {status_code}",
                metadata={"method": method, "url": url},
            )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error=f"Request timed out after {timeout}s",
            )
        except httpx.ConnectError as e:
            return ToolResult(
                success=False,
                error=f"Connection error: {e}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"HTTP request failed: {e}",
            )
