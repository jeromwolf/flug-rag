"""Document upload and management endpoints."""

import logging
import re
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.schemas import DocumentListResponse, DocumentUploadResponse
from auth.dependencies import get_current_user, require_role
from auth.models import Role, User
from config.settings import settings
from pipeline import IngestPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def _sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent path traversal."""
    # Extract basename only (strip directory components)
    name = PurePosixPath(filename).name
    # Remove any remaining path separators or null bytes
    name = re.sub(r'[\x00/\\]', '', name)
    return name or "unnamed"

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
async def upload_document(
    file: UploadFile = File(...),
    current_user: User | None = Depends(get_current_user),
):
    """Upload and ingest a document."""
    # Validate file extension
    safe_name = _sanitize_filename(file.filename or "unnamed")
    ext = Path(safe_name).suffix.lower()
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
    file_path = upload_dir / f"{doc_id}_{safe_name}"

    # Verify resolved path stays within upload_dir
    if not file_path.resolve().is_relative_to(upload_dir.resolve()):
        raise HTTPException(400, "Invalid filename")

    file_path.write_bytes(content)

    # Store original in MinIO object storage (best-effort)
    try:
        from core.storage import get_storage
        storage = get_storage()
        if storage:
            object_name = f"{doc_id}_{safe_name}"
            await storage.upload_file(file_path, object_name=object_name)
    except Exception as e:
        logger.debug("MinIO upload skipped: %s", e)

    # Ingest
    ingest = _get_ingest()
    result = await ingest.ingest(str(file_path), document_id=doc_id)

    # Invalidate document list cache
    await _invalidate_document_cache()

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=safe_name,
        chunk_count=result.chunk_count,
        status=result.status,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(current_user: User | None = Depends(get_current_user)):
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
async def delete_document(
    document_id: str,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER])),
):
    """Delete a document and its vector chunks."""
    upload_dir = Path(settings.upload_dir)
    deleted = False
    deleted_filename = None
    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.name.startswith(document_id):
                deleted_filename = f.name
                f.unlink()
                deleted = True

    if not deleted:
        raise HTTPException(404, f"Document not found: {document_id}")

    # Delete from MinIO (best-effort)
    if deleted_filename:
        try:
            from core.storage import get_storage
            storage = get_storage()
            if storage:
                await storage.delete_file(deleted_filename)
        except Exception as e:
            logger.debug("MinIO delete skipped: %s", e)

    # Delete vector chunks associated with this document
    chunks_deleted = 0
    try:
        from core.vectorstore import create_vectorstore
        vectorstore = create_vectorstore()
        all_docs = await vectorstore.get_all_documents()
        chunk_ids = [
            doc["id"] for doc in all_docs
            if doc.get("metadata", {}).get("document_id") == document_id
        ]
        if chunk_ids:
            await vectorstore.delete(chunk_ids)
            chunks_deleted = len(chunk_ids)
            logger.info("Deleted %d vector chunks for document %s", chunks_deleted, document_id)
    except Exception as e:
        logger.warning("Failed to delete vector chunks for %s: %s", document_id, e)

    # Invalidate caches
    await _invalidate_document_cache()

    return {"status": "deleted", "id": document_id, "chunks_deleted": chunks_deleted}


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    current_user: User | None = Depends(get_current_user),
):
    """원본 문서 파일 다운로드."""
    logger.info("Document download: doc=%s user=%s", document_id, current_user.username if current_user else "anonymous")
    # Try MinIO first
    try:
        from core.storage import get_storage
        storage = get_storage()
        if storage:
            # Find object by prefix
            object_name = None
            upload_dir = Path(settings.upload_dir)
            if upload_dir.exists():
                matching = [f for f in upload_dir.iterdir() if f.is_file() and f.name.startswith(document_id)]
                if matching:
                    object_name = matching[0].name

            if object_name and await storage.file_exists(object_name):
                data = await storage.download_file(object_name)
                if data:
                    from starlette.responses import Response
                    return Response(
                        content=data,
                        media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{object_name}"'},
                    )
    except Exception as e:
        logger.debug("MinIO download fallback to filesystem: %s", e)

    # Fallback to filesystem
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Upload directory not found")

    matching = [f for f in upload_dir.iterdir() if f.is_file() and f.name.startswith(document_id)]

    if not matching:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = matching[0]
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )
