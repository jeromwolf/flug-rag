"""CRUD endpoints for report template management + image-to-template auto-generation."""

import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from auth.dependencies import require_role
from auth.models import Role, User
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ReportTemplate(BaseModel):
    id: str
    name: str
    description: str = ""
    sections: list[dict] = []  # [{title: str, description: str, required: bool}]
    jinja_template: str = ""   # Jinja2 HTML template for PDF rendering
    header_html: str = ""
    footer_html: str = ""
    created_at: str = ""
    updated_at: str = ""


class ReportTemplateCreate(BaseModel):
    name: str
    description: str = ""
    sections: list[dict] = []
    jinja_template: str = ""
    header_html: str = ""
    footer_html: str = ""


class ReportTemplateUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    sections: list[dict] | None = None
    jinja_template: str | None = None
    header_html: str | None = None
    footer_html: str | None = None


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _templates_dir() -> Path:
    """Return the directory where template JSON files are stored, creating it if needed."""
    d = settings.data_dir / "report_templates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_template(template_id: str) -> ReportTemplate:
    """Load a single template by id. Raises HTTPException 404 if not found."""
    path = _templates_dir() / f"{template_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ReportTemplate(**data)
    except Exception as exc:
        logger.error("Failed to load template %s: %s", template_id, exc)
        raise HTTPException(status_code=500, detail="Failed to read template file")


def _save_template(template: ReportTemplate) -> None:
    """Persist a template to disk."""
    path = _templates_dir() / f"{template.id}.json"
    path.write_text(template.model_dump_json(indent=2), encoding="utf-8")


def _list_templates() -> list[ReportTemplate]:
    """Return all templates sorted by updated_at descending."""
    templates: list[ReportTemplate] = []
    for path in _templates_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            templates.append(ReportTemplate(**data))
        except Exception as exc:
            logger.warning("Skipping malformed template file %s: %s", path.name, exc)
    templates.sort(key=lambda t: t.updated_at, reverse=True)
    return templates


# ---------------------------------------------------------------------------
# Helper: build a minimal Jinja2 HTML template from sections
# ---------------------------------------------------------------------------


def _build_jinja_template(sections: list[dict]) -> str:
    """Generate a simple Jinja2 HTML template scaffold from section definitions."""
    rows = ""
    for sec in sections:
        title = sec.get("title", "")
        desc = sec.get("description", "")
        required_mark = " *" if sec.get("required", False) else ""
        rows += (
            f"  <section>\n"
            f"    <h2>{title}{required_mark}</h2>\n"
            f"    <p class=\"hint\">{desc}</p>\n"
            f"    <div class=\"content\">{{{{ {title.lower().replace(' ', '_')} | default('') }}}}</div>\n"
            f"  </section>\n"
        )

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"ko\">\n"
        "<head><meta charset=\"UTF-8\"><title>{{ title | default('보고서') }}</title>\n"
        "<style>\n"
        "  body { font-family: 'Malgun Gothic', sans-serif; margin: 2cm; }\n"
        "  h1 { text-align: center; }\n"
        "  section { margin-bottom: 1.5em; }\n"
        "  h2 { border-bottom: 1px solid #333; padding-bottom: 4px; }\n"
        "  .hint { color: #666; font-size: 0.85em; margin: 0 0 0.5em; }\n"
        "  .content { min-height: 3em; border: 1px solid #ccc; padding: 0.5em; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>{{ title | default('보고서') }}</h1>\n"
        "<p>작성일: {{ date | default('') }} | 작성자: {{ author | default('') }}</p>\n"
        f"{rows}"
        "</body>\n"
        "</html>\n"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/report-templates", response_model=list[ReportTemplate])
async def list_templates(
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Return all report templates sorted by updated_at descending."""
    return _list_templates()


@router.post("/report-templates", response_model=ReportTemplate, status_code=201)
async def create_template(
    body: ReportTemplateCreate,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Create a new report template."""
    now = datetime.now(timezone.utc).isoformat()
    template = ReportTemplate(
        id=str(uuid.uuid4()),
        name=body.name,
        description=body.description,
        sections=body.sections,
        jinja_template=body.jinja_template or _build_jinja_template(body.sections),
        header_html=body.header_html,
        footer_html=body.footer_html,
        created_at=now,
        updated_at=now,
    )
    _save_template(template)
    logger.info("Created report template '%s' (id=%s) by %s", template.name, template.id, current_user.username)
    return template


@router.get("/report-templates/{template_id}", response_model=ReportTemplate)
async def get_template(
    template_id: str,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Get a single report template by id."""
    return _load_template(template_id)


@router.put("/report-templates/{template_id}", response_model=ReportTemplate)
async def update_template(
    template_id: str,
    body: ReportTemplateUpdate,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Update an existing report template (merge non-None fields)."""
    template = _load_template(template_id)

    if body.name is not None:
        template.name = body.name
    if body.description is not None:
        template.description = body.description
    if body.sections is not None:
        template.sections = body.sections
    if body.jinja_template is not None:
        template.jinja_template = body.jinja_template
    if body.header_html is not None:
        template.header_html = body.header_html
    if body.footer_html is not None:
        template.footer_html = body.footer_html

    template.updated_at = datetime.now(timezone.utc).isoformat()
    _save_template(template)
    logger.info("Updated report template '%s' (id=%s) by %s", template.name, template.id, current_user.username)
    return template


@router.delete("/report-templates/{template_id}", status_code=204)
async def delete_template(
    template_id: str,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Delete a report template."""
    path = _templates_dir() / f"{template_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    path.unlink()
    logger.info("Deleted report template id=%s by %s", template_id, current_user.username)


@router.post("/report-templates/from-image", response_model=ReportTemplate, status_code=201)
async def create_template_from_image(
    file: UploadFile = File(...),
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Auto-generate a report template from an uploaded image using Upstage information-extract API."""
    if not settings.upstage_api_key:
        raise HTTPException(
            status_code=400,
            detail="UPSTAGE_API_KEY is not configured. Set the environment variable to use this feature.",
        )

    # Validate that we received an image
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be an image (got '{content_type}')",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    b64 = base64.b64encode(file_bytes).decode("utf-8")

    # Call Upstage information-extract API
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.upstage.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.upstage_api_key}"},
                json={
                    "model": "information-extract",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{content_type};base64,{b64}"},
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "이 보고서/양식 이미지의 구조를 분석하세요. "
                                        "각 섹션의 제목과 설명을 JSON 배열로 반환하세요. "
                                        '형식: [{"title": "섹션 제목", "description": "설명", "required": true/false}]'
                                    ),
                                },
                            ],
                        }
                    ],
                },
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="Upstage API request timed out")
    except httpx.RequestError as exc:
        logger.error("Upstage API request failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Upstage API request failed: {exc}")

    if resp.status_code != 200:
        logger.error("Upstage API returned %s: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=500,
            detail=f"Upstage API error (HTTP {resp.status_code}): {resp.text[:300]}",
        )

    # Parse response
    sections: list[dict] = []
    try:
        result = resp.json()
        raw_content: str = result["choices"][0]["message"]["content"]

        # Extract the JSON array from the response text
        # The model may wrap it in markdown fences or prose
        start = raw_content.find("[")
        end = raw_content.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_str = raw_content[start : end + 1]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                sections = [
                    {
                        "title": str(item.get("title", "")),
                        "description": str(item.get("description", "")),
                        "required": bool(item.get("required", False)),
                    }
                    for item in parsed
                    if isinstance(item, dict)
                ]
    except Exception as exc:
        logger.warning("Could not parse sections from Upstage response: %s", exc)
        # Fall back to an empty section list — still create the template

    # Build and save template
    original_name = Path(file.filename or "uploaded_image").stem
    now = datetime.now(timezone.utc).isoformat()
    template = ReportTemplate(
        id=str(uuid.uuid4()),
        name=f"{original_name} (자동 생성)",
        description=f"이미지 '{file.filename}'에서 자동 생성된 보고서 템플릿",
        sections=sections,
        jinja_template=_build_jinja_template(sections),
        header_html="",
        footer_html="",
        created_at=now,
        updated_at=now,
    )
    _save_template(template)
    logger.info(
        "Auto-generated report template '%s' (id=%s, %d sections) from image by %s",
        template.name,
        template.id,
        len(sections),
        current_user.username,
    )
    return template
