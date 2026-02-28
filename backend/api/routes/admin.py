"""Admin endpoints: system info, LLM providers, prompt management."""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.schemas import LLMProviderInfo, PromptUpdateRequest, SystemInfoResponse
from auth.dependencies import require_role
from auth.models import Role, User
from config.settings import settings
from core.llm import list_providers
from core.vectorstore import create_vectorstore
from rag import PromptManager

logger = logging.getLogger(__name__)
router = APIRouter()

_ADMIN_INFO_CACHE_KEY = "admin:info"
_ADMIN_PROMPTS_CACHE_KEY = "admin:prompts"
_ADMIN_CACHE_TTL = 60  # seconds


async def _invalidate_admin_cache() -> None:
    """Invalidate admin-related cache entries."""
    try:
        from core.cache import get_cache

        cache = await get_cache()
        await cache.clear_pattern("admin:*")
    except Exception:
        pass


def _mask_model(model) -> dict:
    """Mask sensitive fields before API response."""
    d = model.__dict__.copy()
    if d.get("api_key"):
        key = d["api_key"]
        d["api_key"] = key[:4] + "***" + key[-4:] if len(key) > 8 else "***"
    return d


@router.get("/admin/info", response_model=SystemInfoResponse)
async def get_system_info(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    # Check cache first
    try:
        from core.cache import get_cache

        cache = await get_cache()
        cached = await cache.get(_ADMIN_INFO_CACHE_KEY)
        if cached is not None:
            return SystemInfoResponse(**cached)
    except Exception:
        cache = None

    store = create_vectorstore()
    doc_count = await store.count()

    from agent import get_memory

    memory = get_memory()
    session_count = await memory.count_sessions()

    result = SystemInfoResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        default_provider=settings.default_llm_provider,
        document_count=doc_count,
        session_count=session_count,
    )

    # Store in cache
    if cache is not None:
        try:
            await cache.set(_ADMIN_INFO_CACHE_KEY, result.model_dump(), ttl=_ADMIN_CACHE_TTL)
        except Exception:
            pass

    return result


@router.get("/admin/system-metrics")
async def get_system_metrics(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """실시간 시스템 메트릭 (CPU, 메모리, 디스크)."""
    from monitoring.alerting import get_alert_manager

    manager = get_alert_manager()
    metrics = await manager.collect_system_metrics()

    # Add extra memory detail if psutil available
    try:
        import psutil
        mem = psutil.virtual_memory()
        metrics["memory_total_gb"] = round(mem.total / (1024**3), 1)
        metrics["memory_used_gb"] = round(mem.used / (1024**3), 1)
        metrics["memory_available_gb"] = round(mem.available / (1024**3), 1)

        # CPU count
        metrics["cpu_count"] = psutil.cpu_count()
    except ImportError:
        pass

    # Disk detail
    import shutil
    try:
        disk = shutil.disk_usage("/")
        metrics["disk_total_gb"] = round(disk.total / (1024**3), 1)
        metrics["disk_used_gb"] = round(disk.used / (1024**3), 1)
        metrics["disk_free_gb"] = round(disk.free / (1024**3), 1)
    except Exception:
        pass

    return {"metrics": metrics}


@router.get("/admin/providers", response_model=list[LLMProviderInfo])
async def get_providers(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    providers = list_providers()
    default = settings.default_llm_provider
    return [
        LLMProviderInfo(
            name=p,
            is_default=(p == default),
        )
        for p in providers
    ]


@router.get("/admin/prompts")
async def get_prompts(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    # Check cache
    try:
        from core.cache import get_cache

        cache = await get_cache()
        cached = await cache.get(_ADMIN_PROMPTS_CACHE_KEY)
        if cached is not None:
            return cached
    except Exception:
        cache = None

    pm = PromptManager()
    result = {"prompts": pm._system_prompts}

    if cache is not None:
        try:
            await cache.set(_ADMIN_PROMPTS_CACHE_KEY, result, ttl=_ADMIN_CACHE_TTL)
        except Exception:
            pass

    return result


@router.put("/admin/prompts")
async def update_prompt(
    request: PromptUpdateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    import yaml

    pm = PromptManager()
    pm._system_prompts[request.name] = request.content

    # Save to YAML
    yaml_path = settings.prompts_dir / "system.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            pm._system_prompts, f, allow_unicode=True, default_flow_style=False
        )

    # Invalidate admin cache after update
    await _invalidate_admin_cache()

    return {"status": "updated", "name": request.name}


# ==================== Request Models ====================


class ModelCreateRequest(BaseModel):
    name: str = Field(..., description="모델 표시명")
    provider: Literal["vllm", "ollama", "openai", "anthropic"] = Field(..., description="프로바이더")
    model_id: str = Field(..., description="모델 식별자")
    base_url: str = Field("", description="API 기본 URL")
    api_key: str = Field("", description="API 키")
    parameters: str = Field("{}", description="모델 파라미터 (JSON)")


class ModelUpdateRequest(BaseModel):
    name: str | None = None
    provider: str | None = None
    model_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    parameters: str | None = None
    is_active: bool | None = None


class PromptSaveRequest(BaseModel):
    name: str = Field(..., description="프롬프트 이름")
    content: str = Field(..., description="프롬프트 내용")
    change_note: str = Field("", description="변경 메모")


# ==================== Model Management Endpoints ====================


@router.get("/admin/models")
async def list_registered_models(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
    active_only: bool = True,
):
    """등록된 LLM 모델 목록 조회 (ADMIN only)."""
    from core.llm.model_registry import get_model_registry

    registry = await get_model_registry()
    models = await registry.list_models(active_only=active_only)
    return {"models": [_mask_model(m) for m in models]}


@router.post("/admin/models")
async def register_model(
    request: ModelCreateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """새 LLM 모델 등록 (ADMIN only)."""
    from core.llm.model_registry import get_model_registry

    registry = await get_model_registry()
    model = await registry.register_model(
        name=request.name,
        provider=request.provider,
        model_id=request.model_id,
        base_url=request.base_url,
        api_key=request.api_key,
        parameters=request.parameters,
    )
    logger.info(f"Model registered by {current_user.username}: {model.name}")
    return {"status": "registered", "model": _mask_model(model)}


@router.put("/admin/models/{model_id}")
async def update_registered_model(
    model_id: str,
    request: ModelUpdateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """등록된 LLM 모델 수정 (ADMIN only)."""
    from core.llm.model_registry import get_model_registry

    registry = await get_model_registry()
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    try:
        model = await registry.update_model(model_id, **updates)
        logger.info(f"Model updated by {current_user.username}: {model.name}")
        return {"status": "updated", "model": _mask_model(model)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/admin/models/{model_id}")
async def delete_registered_model(
    model_id: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """등록된 LLM 모델 삭제 (ADMIN only)."""
    from core.llm.model_registry import get_model_registry

    registry = await get_model_registry()
    await registry.delete_model(model_id)
    logger.info(f"Model deleted by {current_user.username}: {model_id}")
    return {"status": "deleted", "model_id": model_id}


@router.post("/admin/models/{model_id}/test")
async def test_registered_model(
    model_id: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """등록된 LLM 모델 헬스체크 (ADMIN only)."""
    from core.llm.model_registry import get_model_registry

    registry = await get_model_registry()
    try:
        result = await registry.test_model(model_id)
        return {"model_id": model_id, "health": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ==================== Prompt Versioning Endpoints ====================


@router.get("/admin/prompts/versions/{name}")
async def get_prompt_history(
    name: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
    limit: Annotated[int, Query(le=100)] = 20,
):
    """프롬프트 버전 이력 조회 (ADMIN only)."""
    from rag.prompt_versioning import get_version_manager

    manager = await get_version_manager()
    versions = await manager.get_history(name, limit=limit)
    return {"name": name, "versions": [v.__dict__ for v in versions]}


@router.post("/admin/prompts/versions")
async def save_prompt_version(
    request: PromptSaveRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """새 프롬프트 버전 저장 (ADMIN only)."""
    from rag.prompt_versioning import get_version_manager

    manager = await get_version_manager()
    version = await manager.save_version(
        name=request.name,
        content=request.content,
        created_by=current_user.username,
        change_note=request.change_note,
    )
    logger.info(f"Prompt version saved by {current_user.username}: {request.name} v{version.version}")
    return {"status": "saved", "version": version.__dict__}


@router.post("/admin/prompts/rollback/{name}/{version}")
async def rollback_prompt_version(
    name: str,
    version: int,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """프롬프트 버전 롤백 (ADMIN only)."""
    from rag.prompt_versioning import get_version_manager

    manager = await get_version_manager()
    try:
        rolled_back = await manager.rollback(name, version)
        logger.info(f"Prompt rolled back by {current_user.username}: {name} -> v{version}")
        return {"status": "rolled_back", "version": rolled_back.__dict__}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/admin/prompts/all-versions")
async def get_all_active_prompts(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """모든 프롬프트의 활성 버전 조회 (ADMIN only)."""
    from rag.prompt_versioning import get_version_manager

    manager = await get_version_manager()
    prompts = await manager.get_all_prompts()
    return {"prompts": [p.__dict__ for p in prompts]}
