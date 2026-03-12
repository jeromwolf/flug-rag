"""Microsoft Teams 알림 도구 — Incoming Webhook으로 메시지 발송."""

import logging

import httpx

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class TeamsNotifyTool(BaseTool):
    """Microsoft Teams 알림 도구 — Incoming Webhook으로 메시지 발송"""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="teams_notify",
            description="Microsoft Teams 채널에 알림 메시지를 발송합니다.",
            category="communication",
            help_text=(
                "Microsoft Teams Incoming Webhook을 통해 알림 메시지를 발송합니다.\n"
                "Adaptive Card 형식으로 제목과 본문을 포함한 메시지를 보냅니다.\n"
                "파라미터:\n"
                "  - message: 메시지 본문 (필수)\n"
                "  - title: 메시지 제목 (기본값: 'AI 에이전트 알림')\n"
                "  - webhook_url: Teams Webhook URL (미지정 시 시스템 설정 사용)\n"
                "  - color: 메시지 색상 (파랑/초록/빨강/노랑)"
            ),
            parameters=[
                ToolParameter(
                    name="message",
                    type=ToolParamType.STRING,
                    description="발송할 메시지 본문",
                ),
                ToolParameter(
                    name="title",
                    type=ToolParamType.STRING,
                    description="메시지 제목",
                    required=False,
                    default="AI 에이전트 알림",
                ),
                ToolParameter(
                    name="webhook_url",
                    type=ToolParamType.STRING,
                    description="Teams Incoming Webhook URL (미지정 시 시스템 설정 사용)",
                    required=False,
                ),
                ToolParameter(
                    name="color",
                    type=ToolParamType.STRING,
                    description="메시지 테마 색상 (파랑/초록/빨강/노랑)",
                    required=False,
                    default="0078D4",
                    enum=["0078D4", "28A745", "DC3545", "FFC107"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        message = kwargs.get("message", "")
        if not message:
            return ToolResult(success=False, error="message 파라미터가 필요합니다.")

        title = kwargs.get("title", "AI 에이전트 알림")
        color = kwargs.get("color", "0078D4")

        # Resolve webhook URL
        webhook_url = kwargs.get("webhook_url", "")
        if not webhook_url:
            webhook_url = getattr(settings, "teams_webhook_url", "")
        if not webhook_url:
            return ToolResult(
                success=False,
                error=(
                    "Teams Webhook URL이 설정되지 않았습니다. "
                    "webhook_url 파라미터를 직접 지정하거나, "
                    "환경변수 TEAMS_WEBHOOK_URL을 설정하세요."
                ),
            )

        # Color label mapping for metadata
        color_labels = {
            "0078D4": "파랑 (정보)",
            "28A745": "초록 (성공)",
            "DC3545": "빨강 (오류)",
            "FFC107": "노랑 (경고)",
        }

        try:
            payload = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": [
                                {
                                    "type": "TextBlock",
                                    "text": title,
                                    "weight": "Bolder",
                                    "size": "Medium",
                                    "color": "Accent",
                                },
                                {
                                    "type": "TextBlock",
                                    "text": message,
                                    "wrap": True,
                                },
                            ],
                        },
                    }
                ],
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            logger.info("Teams 알림 발송 완료: %s", title)

            return ToolResult(
                success=True,
                data={
                    "message": "Teams 알림이 성공적으로 발송되었습니다.",
                    "title": title,
                    "color": color_labels.get(color, color),
                },
                metadata={"title": title, "message_length": len(message)},
            )

        except httpx.HTTPStatusError as e:
            logger.error("Teams 알림 HTTP 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"Teams Webhook 전송 실패 (HTTP {e.response.status_code}). Webhook URL을 확인하세요.",
            )
        except httpx.RequestError as e:
            logger.error("Teams 알림 요청 실패: %s", e)
            return ToolResult(
                success=False,
                error=f"Teams Webhook에 연결할 수 없습니다: {e}",
            )
        except Exception as e:
            logger.error("Teams 알림 발송 중 오류: %s", e)
            return ToolResult(
                success=False,
                error=f"Teams 알림 발송 중 오류가 발생했습니다: {e}",
            )
