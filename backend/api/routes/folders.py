"""
폴더 관리 API 엔드포인트

SFR-005: 지식 폴더별 접근 권한 관리 API
"""

import logging
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user, require_role
from auth.models import Role, User
from rag.access_control import get_access_manager, KnowledgeFolder, FolderPermission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/folders", tags=["folders"])


# ===== Pydantic Models =====

class FolderCreateRequest(BaseModel):
    """폴더 생성 요청"""
    name: str = Field(..., description="폴더 이름")
    parent_id: Optional[str] = Field(None, description="상위 폴더 ID (최상위는 None)")
    access_level: Literal["public", "department", "private", "admin"] = Field("private", description="접근 수준")
    department: Optional[str] = Field(None, description="부서 (access_level='department'일 때)")
    owner_id: Optional[str] = Field(None, description="소유자 ID")


class FolderUpdateRequest(BaseModel):
    """폴더 수정 요청"""
    name: Optional[str] = Field(None, description="폴더 이름")
    access_level: Literal["public", "department", "private", "admin"] | None = Field(None, description="접근 수준")
    department: Optional[str] = Field(None, description="부서")


class FolderResponse(BaseModel):
    """폴더 응답"""
    id: str
    name: str
    parent_id: Optional[str]
    owner_id: Optional[str]
    access_level: str
    department: Optional[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_folder(cls, folder: KnowledgeFolder):
        return cls(
            id=folder.id,
            name=folder.name,
            parent_id=folder.parent_id,
            owner_id=folder.owner_id,
            access_level=folder.access_level,
            department=folder.department,
            created_at=folder.created_at,
            updated_at=folder.updated_at
        )


class PermissionSetRequest(BaseModel):
    """권한 설정 요청"""
    user_id: Optional[str] = Field(None, description="특정 사용자 ID")
    department: Optional[str] = Field(None, description="부서 전체 권한")
    permission_type: Literal["read", "write", "admin"] = Field("read", description="권한 유형")


class PermissionResponse(BaseModel):
    """권한 응답"""
    id: str
    folder_id: str
    user_id: Optional[str]
    department: Optional[str]
    permission_type: str

    @classmethod
    def from_permission(cls, permission: FolderPermission):
        return cls(
            id=permission.id,
            folder_id=permission.folder_id,
            user_id=permission.user_id,
            department=permission.department,
            permission_type=permission.permission_type
        )


# ===== API Endpoints =====

@router.get("", response_model=list[FolderResponse])
async def list_folders(
    parent_id: Optional[str] = Query(None, description="상위 폴더 ID (None이면 최상위)"),
    current_user: User | None = Depends(get_current_user)
):
    """
    폴더 목록 조회

    - parent_id가 None이면 최상위 폴더 목록 반환
    - parent_id가 주어지면 해당 폴더의 하위 폴더 목록 반환
    """
    try:
        manager = await get_access_manager()
        folders = await manager.list_folders(parent_id=parent_id)
        return [FolderResponse.from_folder(folder) for folder in folders]
    except Exception as e:
        logger.error(f"Failed to list folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=FolderResponse)
async def create_folder(
    request: FolderCreateRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER]))
):
    """
    폴더 생성

    - 이름, 접근 수준, 부서 등을 지정하여 새 폴더 생성
    """
    try:
        manager = await get_access_manager()
        folder = await manager.create_folder(
            name=request.name,
            parent_id=request.parent_id,
            owner_id=request.owner_id,
            access_level=request.access_level,
            department=request.department
        )
        return FolderResponse.from_folder(folder)
    except Exception as e:
        logger.error(f"Failed to create folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: str,
    current_user: User | None = Depends(get_current_user)
):
    """
    폴더 상세 조회

    - 특정 폴더의 상세 정보 조회
    """
    try:
        manager = await get_access_manager()
        folder = await manager.get_folder(folder_id)
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_id}")
        return FolderResponse.from_folder(folder)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: str,
    request: FolderUpdateRequest,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER]))
):
    """
    폴더 수정

    - 폴더 이름, 접근 수준, 부서 등을 수정
    """
    try:
        manager = await get_access_manager()

        # 요청에서 None이 아닌 필드만 추출
        updates = {
            k: v for k, v in request.model_dump().items()
            if v is not None
        }

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        folder = await manager.update_folder(folder_id, **updates)
        return FolderResponse.from_folder(folder)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: str,
    current_user: User = Depends(require_role([Role.ADMIN]))
):
    """
    폴더 삭제

    - 폴더와 관련된 모든 권한도 함께 삭제
    """
    try:
        manager = await get_access_manager()
        await manager.delete_folder(folder_id)
        return {"status": "success", "message": f"Folder {folder_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{folder_id}/permissions", response_model=list[PermissionResponse])
async def get_folder_permissions(
    folder_id: str,
    current_user: User = Depends(require_role([Role.ADMIN, Role.MANAGER]))
):
    """
    폴더 권한 목록 조회

    - 특정 폴더에 설정된 모든 권한 목록 반환
    """
    try:
        manager = await get_access_manager()
        permissions = await manager.get_folder_permissions(folder_id)
        return [PermissionResponse.from_permission(perm) for perm in permissions]
    except Exception as e:
        logger.error(f"Failed to get folder permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{folder_id}/permissions", response_model=dict)
async def set_folder_permission(
    folder_id: str,
    request: PermissionSetRequest,
    current_user: User = Depends(require_role([Role.ADMIN]))
):
    """
    폴더 권한 설정

    - 특정 사용자 또는 부서에 대한 권한 설정
    - user_id 또는 department 중 하나는 반드시 제공되어야 함
    """
    try:
        if not request.user_id and not request.department:
            raise HTTPException(
                status_code=400,
                detail="Either user_id or department must be provided"
            )

        manager = await get_access_manager()
        await manager.set_permission(
            folder_id=folder_id,
            user_id=request.user_id,
            department=request.department,
            permission_type=request.permission_type
        )
        return {
            "status": "success",
            "message": f"Permission set for folder {folder_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set folder permission: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/permissions/{permission_id}")
async def remove_folder_permission(
    permission_id: str,
    current_user: User = Depends(require_role([Role.ADMIN]))
):
    """
    폴더 권한 삭제

    - 특정 권한 항목 삭제
    """
    try:
        manager = await get_access_manager()
        await manager.remove_permission(permission_id)
        return {"status": "success", "message": f"Permission {permission_id} removed"}
    except Exception as e:
        logger.error(f"Failed to remove folder permission: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{folder_id}/documents", response_model=list[dict])
async def get_folder_documents(
    folder_id: str,
    current_user: User | None = Depends(get_current_user)
):
    """
    폴더 내 문서 목록 조회

    - 특정 폴더에 속한 모든 문서 반환
    - TODO: Document 모델과 통합 필요
    """
    try:
        # TODO: 실제 문서 모델과 통합
        # 현재는 placeholder 반환
        return []
    except Exception as e:
        logger.error(f"Failed to get folder documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
