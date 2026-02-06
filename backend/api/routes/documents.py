"""Document upload and management endpoints."""

import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import DocumentListResponse, DocumentUploadResponse
from config.settings import settings
from pipeline import IngestPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

_DOCUMENT_LIST_CACHE_KEY = "documents:list"
_DOCUMENT_LIST_TTL = 30  # seconds

_ingest = None


def _get_ingest():
    global _ingest
    if _ingest is None:
        _ingest = IngestPipeline()
    return _ingest


async def _invalidate_document_cache() -> None:
    """Invalidate document list cache after upload/delete."""
    try:
        from core.cache import get_cache

        cache = await get_cache()
        await cache.delete(_DOCUMENT_LIST_CACHE_KEY)
        await cache.clear_pattern("documents:*")
    except Exception:
        pass


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document."""
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}",
        )

    # Validate file size
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            400,
            f"File too large. Max: {settings.max_file_size // (1024 * 1024)}MB",
        )

    # Save file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    file_path = upload_dir / f"{doc_id}_{file.filename}"
    file_path.write_bytes(content)

    # Ingest
    ingest = _get_ingest()
    result = await ingest.ingest(str(file_path), document_id=doc_id)

    # Invalidate document list cache
    await _invalidate_document_cache()

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename or "",
        chunk_count=result.chunk_count,
        status=result.status,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents (cached for 30s)."""
    # Check cache first
    try:
        from core.cache import get_cache

        cache = await get_cache()
        cached = await cache.get(_DOCUMENT_LIST_CACHE_KEY)
        if cached is not None:
            return DocumentListResponse(**cached)
    except Exception:
        cache = None

    # Get upload directory files
    upload_dir = Path(settings.upload_dir)
    documents = []
    if upload_dir.exists():
        for f in sorted(
            upload_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True
        ):
            if f.is_file():
                parts = f.stem.split("_", 1)
                doc_id = parts[0] if len(parts) > 1 else f.stem
                filename = parts[1] + f.suffix if len(parts) > 1 else f.name
                documents.append(
                    {
                        "id": doc_id,
                        "filename": filename,
                        "file_type": f.suffix,
                        "chunk_count": 0,  # Simplified; actual counts in vectorstore
                        "uploaded_at": datetime.fromtimestamp(
                            f.stat().st_mtime
                        ).isoformat(),
                        "metadata": {},
                    }
                )

    result = DocumentListResponse(documents=documents, total=len(documents))

    # Store in cache
    if cache is not None:
        try:
            await cache.set(_DOCUMENT_LIST_CACHE_KEY, result.model_dump(), ttl=_DOCUMENT_LIST_TTL)
        except Exception:
            pass

    return result


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document."""
    upload_dir = Path(settings.upload_dir)
    deleted = False
    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.name.startswith(document_id):
                f.unlink()
                deleted = True

    if not deleted:
        raise HTTPException(404, f"Document not found: {document_id}")

    # Invalidate document list cache
    await _invalidate_document_cache()

    return {"status": "deleted", "id": document_id}
