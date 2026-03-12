"""
보고서 내보내기 API
Markdown 콘텐츠를 PDF 또는 DOCX 파일로 변환하여 다운로드.
"""
import io
import logging
import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.models import User
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reports"])


class ReportExportRequest(BaseModel):
    content: str          # Markdown content
    format: str           # "pdf" or "docx"
    title: str = "보고서"  # Report title


def _build_html(content: str, title: str) -> str:
    """Markdown을 HTML로 변환하고 스타일이 적용된 전체 HTML 문서를 반환."""
    import markdown

    body_html = markdown.markdown(
        content,
        extensions=["tables", "fenced_code"],
    )

    now_str = datetime.now().strftime("%Y년 %m월 %d일")
    platform = settings.platform_name

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');

    @page {{
      size: A4;
      margin: 2cm;
      @top-center {{
        content: "{title}  |  {platform}";
        font-family: 'Noto Sans KR', sans-serif;
        font-size: 9pt;
        color: #666;
      }}
      @bottom-center {{
        content: "생성일: {now_str}    페이지 " counter(page) " / " counter(pages);
        font-family: 'Noto Sans KR', sans-serif;
        font-size: 9pt;
        color: #666;
      }}
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      font-family: 'Noto Sans KR', sans-serif;
      font-size: 11pt;
      line-height: 1.7;
      color: #1a1a1a;
      background: #ffffff;
    }}

    .report-header {{
      text-align: center;
      border-bottom: 2px solid #003366;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}

    .report-header h1 {{
      font-size: 20pt;
      font-weight: 700;
      color: #003366;
      margin: 0 0 4px 0;
    }}

    .report-header .meta {{
      font-size: 9pt;
      color: #555;
    }}

    h1, h2, h3, h4, h5, h6 {{
      font-family: 'Noto Sans KR', sans-serif;
      font-weight: 700;
      color: #003366;
      margin-top: 1.4em;
      margin-bottom: 0.4em;
    }}

    h1 {{ font-size: 16pt; border-bottom: 1px solid #ccd6e0; padding-bottom: 4px; }}
    h2 {{ font-size: 13pt; }}
    h3 {{ font-size: 11pt; color: #004080; }}

    p {{
      margin: 0.5em 0 1em 0;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1em 0;
      font-size: 10pt;
    }}

    th, td {{
      border: 1px solid #b0c4d8;
      padding: 6px 10px;
      text-align: left;
    }}

    th {{
      background-color: #003366;
      color: #ffffff;
      font-weight: 700;
    }}

    tr:nth-child(even) td {{
      background-color: #f0f4f8;
    }}

    ul, ol {{
      padding-left: 1.5em;
      margin: 0.5em 0 1em 0;
    }}

    li {{
      margin-bottom: 0.3em;
    }}

    code {{
      font-family: 'Courier New', Courier, monospace;
      font-size: 9pt;
      background: #f4f4f4;
      padding: 1px 4px;
      border-radius: 3px;
      color: #c0392b;
    }}

    pre {{
      background: #f4f4f4;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 12px;
      overflow-x: auto;
      font-size: 9pt;
    }}

    pre code {{
      background: none;
      padding: 0;
      color: #1a1a1a;
    }}

    blockquote {{
      border-left: 4px solid #003366;
      margin: 1em 0;
      padding: 0.5em 1em;
      color: #555;
      background: #f8fafc;
    }}

    a {{
      color: #003366;
    }}
  </style>
</head>
<body>
  <div class="report-header">
    <h1>{title}</h1>
    <div class="meta">{platform} &nbsp;|&nbsp; 생성일: {now_str}</div>
  </div>
  <div class="report-body">
    {body_html}
  </div>
</body>
</html>"""
    return html


def _convert_to_pdf(content: str, title: str) -> bytes:
    """Markdown → HTML → PDF 바이트 반환."""
    from weasyprint import HTML

    html_str = _build_html(content, title)
    pdf_bytes = HTML(string=html_str).write_pdf()
    return pdf_bytes


def _convert_to_docx(content: str, title: str) -> bytes:
    """Markdown → DOCX 바이트 반환."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()

    # 페이지 여백 설정 (2cm)
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # 기본 폰트 설정 (Malgun Gothic — 한국어 지원)
    style = doc.styles["Normal"]
    style.font.name = "맑은 고딕"
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = Pt(18)  # ~1.5 line spacing
    # East Asian font fallback
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")

    # 제목
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(title)
    title_run.bold = True
    title_run.font.size = Pt(18)
    title_run.font.name = "맑은 고딕"
    title_run.font.color.rgb = RGBColor(0, 51, 102)  # #003366

    # 부제목 (날짜 + 플랫폼명)
    now_str = datetime.now().strftime("%Y년 %m월 %d일")
    subtitle_para = doc.add_paragraph()
    subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle_para.add_run(f"{now_str}  |  {settings.platform_name}")
    subtitle_run.font.size = Pt(10)
    subtitle_run.font.name = "맑은 고딕"
    subtitle_run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph()  # 빈 줄

    # Markdown 라인별 파싱
    numbered_list_re = re.compile(r"^(\d+)\.\s+(.*)")

    for line in content.splitlines():
        stripped = line.rstrip()

        if stripped.startswith("### "):
            text = stripped[4:]
            para = doc.add_heading(text, level=3)
            _set_heading_font(para, Pt(11), RGBColor(0, 64, 128))

        elif stripped.startswith("## "):
            text = stripped[3:]
            para = doc.add_heading(text, level=2)
            _set_heading_font(para, Pt(13), RGBColor(0, 51, 102))

        elif stripped.startswith("# "):
            text = stripped[2:]
            para = doc.add_heading(text, level=1)
            _set_heading_font(para, Pt(16), RGBColor(0, 51, 102))

        elif stripped.startswith("- ") or stripped.startswith("* "):
            text = stripped[2:]
            para = doc.add_paragraph(style="List Bullet")
            _add_run(para, text)

        elif numbered_list_re.match(stripped):
            m = numbered_list_re.match(stripped)
            text = m.group(2)  # type: ignore[union-attr]
            para = doc.add_paragraph(style="List Number")
            _add_run(para, text)

        elif stripped == "":
            # 빈 줄은 건너뜀
            continue

        else:
            para = doc.add_paragraph()
            _add_run(para, stripped)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


def _set_heading_font(para, size, color: "RGBColor") -> None:
    """Heading 단락의 폰트를 한국어 호환으로 설정."""
    from docx.oxml.ns import qn

    for run in para.runs:
        run.font.name = "맑은 고딕"
        run.font.size = size
        run.font.color.rgb = color
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")


def _add_run(para, text: str) -> None:
    """단락에 런을 추가하고 한국어 폰트를 설정."""
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run = para.add_run(text)
    run.font.name = "맑은 고딕"
    run.font.size = Pt(11)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")


@router.post("/export")
async def export_report(
    request: ReportExportRequest,
    current_user: User | None = Depends(get_current_user),
) -> StreamingResponse:
    """Markdown 콘텐츠를 PDF 또는 DOCX 파일로 변환하여 반환."""

    if request.format not in ("pdf", "docx"):
        raise HTTPException(
            status_code=400,
            detail="format은 'pdf' 또는 'docx'여야 합니다.",
        )

    safe_title = re.sub(r'[\\/*?:"<>|]', "_", request.title)

    if request.format == "pdf":
        try:
            pdf_bytes = _convert_to_pdf(request.content, request.title)
        except Exception as exc:
            logger.error("PDF 변환 실패: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"PDF 변환 중 오류가 발생했습니다: {exc}",
            ) from exc

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_title}.pdf"',
            },
        )

    else:  # docx
        try:
            docx_bytes = _convert_to_docx(request.content, request.title)
        except Exception as exc:
            logger.error("DOCX 변환 실패: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"DOCX 변환 중 오류가 발생했습니다: {exc}",
            ) from exc

        return StreamingResponse(
            io.BytesIO(docx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_title}.docx"',
            },
        )
