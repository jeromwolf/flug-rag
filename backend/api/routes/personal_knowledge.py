"""
개인 지식 공간 API 엔드포인트

SFR-005: 사용자별 개인 문서 관리
- 개인 문서 업로드 (user_id 메타데이터 태깅)
- 내 문서 목록 조회
- 개인 문서 삭제
- PII 스캔 결과 포함
"""

import logging
import re as _re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from auth.dependencies import get_current_user
from auth.models import User
from config.settings import settings
from pipeline.ingest import IngestPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/my-knowledge", tags=["personal-knowledge"])

_UUID_RE = _re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', _re.IGNORECASE)

def _validate_document_id(document_id: str) -> None:
    """document_id가 유효한 UUID인지 검증."""
    if not _UUID_RE.match(document_id):
        raise HTTPException(status_code=400, detail="Invalid document_id format (must be UUID)")

def _sanitize_filename(filename: str) -> str:
    """Remove dangerous characters from filename."""
    name = _re.sub(r'[^\w\s\-.]', '', filename)
    name = _re.sub(r'\.\.+', '.', name)
    return name.strip() or "unnamed"

_ingest = None


def _get_ingest():
    global _ingest
    if _ingest is None:
        _ingest = IngestPipeline()
    return _ingest


class PersonalDocResponse(BaseModel):
    """개인 문서 응답"""
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: str
    pii_warnings: list[str] = []


class PersonalDocListResponse(BaseModel):
    """개인 문서 목록 응답"""
    documents: list[PersonalDocResponse]
    total: int


@router.post("/upload", response_model=PersonalDocResponse)
async def upload_personal_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    개인 문서 업로드

    - 파일 업로드 후 자동 인제스트
    - 사용자 ID 메타데이터 자동 태깅 (검색 시 본인만 접근 가능)
    - PII 자동 스캔 및 경고
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}"
        )

    # Validate file size
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {settings.max_file_size // (1024*1024)}MB"
        )

    # Save file to personal directory
    personal_dir = Path(settings.upload_dir) / "personal" / current_user.id
    personal_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_name = _sanitize_filename(file.filename)
    file_path = personal_dir / f"{doc_id}_{safe_name}"
    # Verify path doesn't escape personal_dir
    if not file_path.resolve().is_relative_to(personal_dir.resolve()):
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다.")
    file_path.write_bytes(content)

    try:
        # Run ingestion with personal metadata
        ingest = _get_ingest()

        # Add user_id to metadata for access control
        result = await ingest.ingest(
            str(file_path),
            document_id=doc_id,
            extra_metadata={
                "user_id": current_user.id,
                "owner_type": "personal",
                "department": current_user.department or "",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # PII warnings from ingest pipeline (scans extracted text, not raw bytes)
        pii_warnings = result.pii_warnings or []

        return PersonalDocResponse(
            document_id=doc_id,
            filename=file.filename,
            file_type=ext,
            chunk_count=result.chunk_count,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            pii_warnings=pii_warnings,
        )

    except Exception as e:
        # Clean up on failure
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Personal document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("", response_model=PersonalDocListResponse)
async def list_personal_documents(
    current_user: User = Depends(get_current_user),
):
    """
    내 개인 문서 목록 조회

    - 현재 사용자의 개인 문서만 반환
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # List files from personal directory
        personal_dir = Path(settings.upload_dir) / "personal" / current_user.id
        documents = []

        if personal_dir.exists():
            for file_path in sorted(
                personal_dir.iterdir(),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            ):
                if file_path.is_file():
                    # Parse: {doc_id}_{filename}
                    parts = file_path.stem.split("_", 1)
                    doc_id = parts[0] if len(parts) > 1 else file_path.stem
                    filename = parts[1] + file_path.suffix if len(parts) > 1 else file_path.name

                    # Get chunk count from vectorstore (best effort)
                    chunk_count = 0
                    try:
                        from core.vectorstore import create_vectorstore
                        vectorstore = create_vectorstore()

                        # Search with user_id and document_id filters
                        # Note: This is a workaround since we don't have get_documents_by_metadata
                        # We'll need to do a query instead
                        # For now, we'll skip chunk counting for performance
                        chunk_count = 0
                    except Exception:
                        pass

                    documents.append(PersonalDocResponse(
                        document_id=doc_id,
                        filename=filename,
                        file_type=file_path.suffix,
                        chunk_count=chunk_count,
                        uploaded_at=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        pii_warnings=[],
                    ))

        return PersonalDocListResponse(
            documents=documents,
            total=len(documents),
        )

    except Exception as e:
        logger.error(f"Failed to list personal documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_personal_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    개인 문서 삭제

    - 본인 문서만 삭제 가능
    - 파일시스템 및 ChromaDB에서 관련 청크 모두 제거
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    _validate_document_id(document_id)

    try:
        # Delete file from personal directory
        personal_dir = Path(settings.upload_dir) / "personal" / current_user.id
        deleted_file = False
        file_path = None

        if personal_dir.exists():
            for fp in personal_dir.iterdir():
                if fp.name.startswith(document_id):
                    fp.unlink()
                    deleted_file = True
                    file_path = fp
                    break

        if not deleted_file:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # Delete chunks from vectorstore
        # Note: ChromaDB doesn't have delete_by_metadata, so we need to query first
        chunks_removed = 0
        try:
            from core.vectorstore import create_vectorstore
            vectorstore = create_vectorstore()

            # Get all chunks for this document
            # Since we can't query by metadata directly, we'll use the collection's get method
            # with where filter (ChromaDB supports this in the underlying API)
            import asyncio

            def _delete_chunks():
                collection = vectorstore._collection
                # Query for chunks with this document_id and user_id
                results = collection.get(
                    where={
                        "$and": [
                            {"document_id": {"$eq": document_id}},
                            {"user_id": {"$eq": current_user.id}},
                        ]
                    },
                    include=["metadatas"],
                )

                if results["ids"]:
                    collection.delete(ids=results["ids"])
                    return len(results["ids"])
                return 0

            chunks_removed = await asyncio.to_thread(_delete_chunks)

        except Exception as e:
            logger.warning(f"Failed to delete chunks from vectorstore: {e}")

        return {
            "status": "success",
            "message": f"Document {document_id} deleted",
            "chunks_removed": chunks_removed,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete personal document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/pii-scan")
async def scan_document_pii(
    document_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    문서 PII 재스캔

    - 지정 문서의 청크 텍스트를 PII 스캔하여 결과 반환
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    _validate_document_id(document_id)

    try:
        from core.vectorstore import create_vectorstore
        from pipeline.pii_detector import get_pii_detector

        vectorstore = create_vectorstore()
        pii_detector = get_pii_detector()

        # Get document chunks
        import asyncio

        def _get_chunks():
            collection = vectorstore._collection
            results = collection.get(
                where={
                    "$and": [
                        {"document_id": {"$eq": document_id}},
                        {"user_id": {"$eq": current_user.id}},
                    ]
                },
                include=["documents"],
            )
            return results

        results = await asyncio.to_thread(_get_chunks)

        if not results["ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found or not owned by you: {document_id}"
            )

        # Scan each chunk
        all_warnings = []
        total_matches = 0
        for text in results.get("documents", []):
            if text:
                scan_result = pii_detector.scan(text)
                total_matches += scan_result.match_count
                all_warnings.extend(scan_result.warnings)

        return {
            "document_id": document_id,
            "has_pii": total_matches > 0,
            "total_matches": total_matches,
            "warnings": list(set(all_warnings)),  # Deduplicate
            "chunks_scanned": len(results["ids"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PII scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
