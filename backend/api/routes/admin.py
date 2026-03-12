"""Admin endpoints: system info, LLM providers, prompt management."""

import json
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
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

# ==================== Per-Category Cache TTL (runtime mutable) ====================
# These store the TTL values that are applied per cache category.
# Keys match the category names used in cache operations.
_CACHE_TTL_CONFIG: dict[str, int] = {
    "rag_query": 120,      # RAG stream cache (seconds)
    "llm_response": 300,   # LLM response cache
    "embeddings": 3600,    # Embedding cache
    "documents": 86400,    # Document metadata cache
}

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
    import asyncio

    # Check cache first
    try:
        from core.cache import get_cache

        cache = await get_cache()
        cached = await cache.get(_ADMIN_INFO_CACHE_KEY)
        if cached is not None:
            return SystemInfoResponse(**cached)
    except Exception:
        cache = None

    # Milvus Lite single-process: wrap with timeout to avoid deadlock
    doc_count = 0
    file_count = 0
    session_count = 0

    try:
        store = create_vectorstore()
        doc_count = await asyncio.wait_for(store.count(), timeout=5.0)
        file_count = await asyncio.wait_for(store.count_files(), timeout=5.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Failed to get vectorstore stats: %s", e)

    try:
        from agent import get_memory

        memory = get_memory()
        session_count = await asyncio.wait_for(memory.count_sessions(), timeout=5.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Failed to get session count: %s", e)

    result = SystemInfoResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        default_provider=settings.default_llm_provider,
        document_count=doc_count,
        file_count=file_count,
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
    """실시간 시스템 메트릭 (CPU, 메모리, 디스크, GPU, 프로세스, 업타임)."""
    from monitoring.alerting import get_alert_manager

    manager = get_alert_manager()
    metrics = await manager.collect_system_metrics()

    # Add extra memory detail, process count, and uptime if psutil available
    try:
        import psutil
        mem = psutil.virtual_memory()
        metrics["memory_total_gb"] = round(mem.total / (1024**3), 1)
        metrics["memory_used_gb"] = round(mem.used / (1024**3), 1)
        metrics["memory_available_gb"] = round(mem.available / (1024**3), 1)

        # CPU count
        metrics["cpu_count"] = psutil.cpu_count()

        # Process count
        metrics["process_count"] = len(psutil.pids())

        # System uptime (seconds since boot)
        boot_time = psutil.boot_time()
        uptime_seconds = int(__import__("time").time() - boot_time)
        metrics["uptime_seconds"] = uptime_seconds
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        metrics["uptime_human"] = f"{hours}h {minutes}m"
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

    # GPU info (nvidia-smi via pynvml, optional)
    gpu_info = []
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info.append({
                "index": i,
                "name": name,
                "utilization": util.gpu,
                "memory_used_gb": round(mem_info.used / (1024**3), 1),
                "memory_total_gb": round(mem_info.total / (1024**3), 1),
                "memory_percent": round(mem_info.used / mem_info.total * 100, 1),
            })
        pynvml.nvmlShutdown()
    except Exception:
        # pynvml not available or no NVIDIA GPU — skip silently
        pass

    if gpu_info:
        metrics["gpu"] = gpu_info

    # Average response time from recent assistant messages
    try:
        import json as _json
        import aiosqlite
        memory_db = settings.data_dir / "memory.db"
        async with aiosqlite.connect(memory_db) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT metadata FROM messages WHERE role='assistant' AND metadata IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 100"
            ) as cur:
                rows = await cur.fetchall()
                latencies = []
                for row in rows:
                    try:
                        meta = _json.loads(row["metadata"] or "{}")
                        if "latency_ms" in meta and meta["latency_ms"]:
                            latencies.append(float(meta["latency_ms"]))
                    except (ValueError, TypeError, KeyError):
                        pass
                if latencies:
                    metrics["avg_response_time_ms"] = round(sum(latencies) / len(latencies), 0)
    except Exception:
        pass

    # Cache hit rate
    try:
        from core.cache import get_cache
        cache = await get_cache()
        if hasattr(cache, "stats"):
            cache_stats = cache.stats
            metrics["cache_hit_rate"] = cache_stats.get("hit_rate", 0.0)
            metrics["cache_size"] = cache_stats.get("size", 0)
            metrics["cache_hits"] = cache_stats.get("hits", 0)
            metrics["cache_misses"] = cache_stats.get("misses", 0)
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


# ==================== Settings Endpoint ====================


class SettingsUpdateRequest(BaseModel):
    default_provider: Literal["vllm", "ollama", "openai", "anthropic"] = Field(
        ..., description="기본 LLM 프로바이더"
    )


@router.put("/admin/settings")
async def update_settings(
    request: SettingsUpdateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """시스템 설정 업데이트 (기본 프로바이더 변경 등) (ADMIN only)."""
    old_provider = settings.default_llm_provider
    settings.default_llm_provider = request.default_provider
    await _invalidate_admin_cache()
    logger.info(
        f"Default provider changed by {current_user.username}: {old_provider} → {request.default_provider}"
    )
    return {"status": "updated", "default_provider": request.default_provider}


# ==================== Model Management Endpoints ====================


_DEFAULT_SEED_MODELS = [
    {
        "name": "Qwen2.5 7B (Ollama)",
        "provider": "ollama",
        "model_id": "qwen2.5:7b",
        "parameters": '{"temperature": 0.2, "max_tokens": 2048}',
    },
    {
        "name": "Qwen2.5 14B (Ollama)",
        "provider": "ollama",
        "model_id": "qwen2.5:14b",
        "parameters": '{"temperature": 0.2, "max_tokens": 2048}',
    },
    {
        "name": "GPT-4o (OpenAI)",
        "provider": "openai",
        "model_id": "gpt-4o",
        "parameters": '{"temperature": 0.2, "max_tokens": 4096}',
    },
    {
        "name": "GPT-4o Mini (OpenAI)",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "parameters": '{"temperature": 0.2, "max_tokens": 4096}',
    },
    {
        "name": "Claude Sonnet 4 (Anthropic)",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "parameters": '{"temperature": 0.2, "max_tokens": 4096}',
    },
    {
        "name": "Claude Haiku 4.5 (Anthropic)",
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "parameters": '{"temperature": 0.2, "max_tokens": 4096}',
    },
]


@router.get("/admin/models")
async def list_registered_models(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
    active_only: bool = True,
):
    """등록된 LLM 모델 목록 조회 (ADMIN only). 빈 경우 기본 모델 시드 데이터 자동 삽입."""
    from core.llm.model_registry import get_model_registry

    registry = await get_model_registry()
    # Check all models (not just active) to decide if seeding is needed
    all_models = await registry.list_models(active_only=False)
    if not all_models:
        logger.info("Model registry empty — seeding default models")
        for seed in _DEFAULT_SEED_MODELS:
            try:
                await registry.register_model(**seed)
            except Exception as e:
                logger.warning(f"Seed model registration failed ({seed['model_id']}): {e}")

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


# ==================== Chunking Config Endpoints ====================


@router.get("/admin/chunking-config")
async def get_chunking_config(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """현재 청킹 설정 조회 (ADMIN only)."""
    return {
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


@router.put("/admin/chunking-config")
async def update_chunking_config(
    request: dict,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """청킹 설정 업데이트 (런타임 변경, 재인제스트 필요)."""
    allowed_strategies = ["recursive", "semantic", "table", "hierarchical", "adaptive"]

    if "chunk_strategy" in request:
        if request["chunk_strategy"] not in allowed_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Allowed: {allowed_strategies}",
            )
        settings.chunk_strategy = request["chunk_strategy"]
    if "chunk_size" in request:
        val = int(request["chunk_size"])
        if not (100 <= val <= 4000):
            raise HTTPException(status_code=400, detail="chunk_size must be 100–4000")
        settings.chunk_size = val
    if "chunk_overlap" in request:
        val = int(request["chunk_overlap"])
        if not (0 <= val <= 500):
            raise HTTPException(status_code=400, detail="chunk_overlap must be 0–500")
        settings.chunk_overlap = val

    logger.info(
        f"Chunking config updated by {current_user.username}: "
        f"strategy={settings.chunk_strategy}, size={settings.chunk_size}, overlap={settings.chunk_overlap}"
    )

    return {
        "status": "updated",
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "note": "변경된 설정은 새로 인제스트하는 문서부터 적용됩니다.",
    }


# ==================== Log Level Endpoints ====================


@router.get("/admin/log-level")
async def get_log_level(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """현재 로그 레벨 조회 (ADMIN only)."""
    current = logging.getLogger().getEffectiveLevel()
    return {"level": logging.getLevelName(current)}


@router.put("/admin/log-level")
async def set_log_level(
    level: str,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """런타임 로그 레벨 변경 (ADMIN only)."""
    level_upper = level.upper()
    if level_upper not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise HTTPException(status_code=400, detail=f"Invalid log level: {level}. Allowed: DEBUG, INFO, WARNING, ERROR")
    numeric = getattr(logging, level_upper)
    logging.getLogger().setLevel(numeric)
    for name in ("api", "rag", "pipeline", "core", "agent", "auth", "monitoring"):
        logging.getLogger(name).setLevel(numeric)
    logger.info(f"Log level changed to {level_upper} by {current_user.username}")
    return {"status": "updated", "level": level_upper}


# ==================== Cache Config Endpoints ====================


class CacheConfigUpdateRequest(BaseModel):
    rag_query: int | None = Field(None, ge=0, le=86400, description="RAG 질의 캐시 TTL (초)")
    llm_response: int | None = Field(None, ge=0, le=86400, description="LLM 응답 캐시 TTL (초)")
    embeddings: int | None = Field(None, ge=0, le=604800, description="임베딩 캐시 TTL (초)")
    documents: int | None = Field(None, ge=0, le=604800, description="문서 메타데이터 캐시 TTL (초)")


class CacheClearRequest(BaseModel):
    category: str | None = Field(None, description="지울 카테고리 (없으면 전체 삭제)")


@router.get("/admin/cache-config")
async def get_cache_config(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """캐시 카테고리별 TTL 설정 조회 (ADMIN only)."""
    from core.cache import get_cache, InMemoryCache

    cache = await get_cache()
    cache_type = "memory" if isinstance(cache, InMemoryCache) else "redis"

    return {
        "cache_enabled": settings.cache_enabled,
        "cache_type": cache_type,
        "ttl": dict(_CACHE_TTL_CONFIG),
        "categories": [
            {"key": "rag_query", "label": "RAG 질의", "description": "동일한 질의에 대한 스트리밍 응답 캐시", "unit": "초"},
            {"key": "llm_response", "label": "LLM 응답", "description": "LLM 완성 결과 캐시", "unit": "초"},
            {"key": "embeddings", "label": "임베딩", "description": "텍스트 임베딩 벡터 캐시", "unit": "초"},
            {"key": "documents", "label": "문서 메타데이터", "description": "문서 목록/메타데이터 캐시", "unit": "초"},
        ],
    }


@router.put("/admin/cache-config")
async def update_cache_config(
    request: CacheConfigUpdateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """캐시 카테고리별 TTL 설정 변경 (ADMIN only). 런타임 적용, 재시작 시 초기화."""
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="변경할 TTL 값을 하나 이상 지정해야 합니다.")

    for key, value in updates.items():
        if key in _CACHE_TTL_CONFIG:
            _CACHE_TTL_CONFIG[key] = value

    logger.info(f"Cache TTL config updated by {current_user.username}: {updates}")
    return {
        "status": "updated",
        "ttl": dict(_CACHE_TTL_CONFIG),
        "note": "변경된 TTL은 런타임에만 적용됩니다. 서버 재시작 시 기본값으로 초기화됩니다.",
    }


@router.post("/admin/cache-clear")
async def clear_cache(
    request: CacheClearRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """캐시 초기화 (전체 또는 카테고리별) (ADMIN only)."""
    from core.cache import get_cache, InMemoryCache

    valid_categories = list(_CACHE_TTL_CONFIG.keys())
    category = request.category

    if category and category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Valid: {valid_categories}",
        )

    cache = await get_cache()
    deleted = 0

    try:
        if category:
            # Clear keys matching the category prefix
            # RAG stream cache keys start with the query hash; use a broad pattern
            pattern_map = {
                "rag_query": "rag.chain.*",
                "llm_response": "*.llm.*",
                "embeddings": "*.embedding*",
                "documents": "*.document*",
            }
            pattern = pattern_map.get(category, f"{category}:*")
            deleted = await cache.clear_pattern(pattern)
            # Also try admin cache
            await cache.clear_pattern("admin:*")
        else:
            # Clear all: for InMemoryCache use clear(), for Redis use pattern
            if isinstance(cache, InMemoryCache):
                await cache.clear()
                deleted = -1  # unknown count for full clear
            else:
                deleted = await cache.clear_pattern("*")
                await cache.clear_pattern("flux-rag:*")
    except Exception as e:
        logger.warning(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"캐시 초기화 실패: {e}")

    logger.info(f"Cache cleared by {current_user.username}: category={category or 'ALL'}, deleted={deleted}")
    return {
        "status": "cleared",
        "category": category or "all",
        "deleted_count": deleted,
    }


def get_cache_ttl(category: str) -> int:
    """Helper for other modules to read the current TTL for a category."""
    return _CACHE_TTL_CONFIG.get(category, settings.cache_default_ttl)


# ==================== Storage Management Endpoints ====================


@router.get("/admin/storage-stats")
async def get_storage_stats(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """저장소 통계 조회: 총 사용량, 사용자별 분류, 파일 유형별 통계, 대용량 파일, 보존 기간 임박 파일 (ADMIN only)."""
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        return {
            "total_bytes": 0,
            "total_files": 0,
            "by_user": [],
            "by_type": [],
            "largest_files": [],
            "expiring_soon": [],
            "quota": {
                "max_file_size_mb": settings.max_file_size_mb,
                "max_user_storage_mb": settings.max_user_storage_mb,
                "max_total_storage_gb": settings.max_total_storage_gb,
                "file_retention_days": settings.file_retention_days,
            },
        }

    _HIDDEN = {".DS_Store", ".omc", "Store", ".gitkeep", "Thumbs.db", "__MACOSX"}
    total_bytes = 0
    total_files = 0
    by_type: dict[str, dict] = {}
    file_records: list[dict] = []

    now = datetime.now(tz=timezone.utc)
    retention_threshold = settings.file_retention_days - 30  # warn 30 days before

    for f in upload_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.name in _HIDDEN or f.name.startswith("."):
            continue

        stat = f.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        age_days = (now - mtime).days
        ext = f.suffix.lower() or "unknown"

        total_bytes += size
        total_files += 1

        # By type
        if ext not in by_type:
            by_type[ext] = {"extension": ext, "count": 0, "bytes": 0}
        by_type[ext]["count"] += 1
        by_type[ext]["bytes"] += size

        file_records.append({
            "filename": f.name,
            "size_bytes": size,
            "extension": ext,
            "uploaded_at": mtime.isoformat(),
            "age_days": age_days,
        })

    # Sort largest files
    largest_files = sorted(file_records, key=lambda x: x["size_bytes"], reverse=True)[:10]

    # Files approaching retention limit
    expiring_soon = [
        r for r in file_records
        if r["age_days"] >= retention_threshold
    ]
    expiring_soon = sorted(expiring_soon, key=lambda x: x["age_days"], reverse=True)[:20]

    # by_type list sorted by bytes desc
    by_type_list = sorted(by_type.values(), key=lambda x: x["bytes"], reverse=True)

    # Per-user breakdown (parse user_id from filenames where possible via sub-dirs)
    # Since uploads use flat uuid_filename format without user tagging, aggregate by
    # sub-directory name (personal knowledge uses username subdirs)
    by_user: list[dict] = []
    user_dirs: dict[str, dict] = {}
    for sub in upload_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith(".") and sub.name not in _HIDDEN:
            user_bytes = 0
            user_files = 0
            for uf in sub.rglob("*"):
                if uf.is_file() and not uf.name.startswith("."):
                    user_bytes += uf.stat().st_size
                    user_files += 1
            if user_files > 0:
                user_dirs[sub.name] = {
                    "user": sub.name,
                    "file_count": user_files,
                    "bytes": user_bytes,
                }
    by_user = sorted(user_dirs.values(), key=lambda x: x["bytes"], reverse=True)[:20]

    return {
        "total_bytes": total_bytes,
        "total_files": total_files,
        "by_user": by_user,
        "by_type": by_type_list,
        "largest_files": largest_files,
        "expiring_soon": expiring_soon,
        "quota": {
            "max_file_size_mb": settings.max_file_size_mb,
            "max_user_storage_mb": settings.max_user_storage_mb,
            "max_total_storage_gb": settings.max_total_storage_gb,
            "file_retention_days": settings.file_retention_days,
        },
    }


# ==================== LLM Playground Endpoint ====================


class PlaygroundRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="사용자 프롬프트")
    model: str | None = Field(None, description="모델 이름 (미지정 시 기본 모델)")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="샘플링 온도")
    max_tokens: int = Field(1024, ge=64, le=4096, description="최대 생성 토큰 수")
    system_prompt: str | None = Field(None, description="시스템 프롬프트 (선택)")


class PlaygroundResponse(BaseModel):
    response: str
    latency_ms: int
    model_used: str
    tokens_used: int | None = None


# ==================== Prompt Simulator Endpoint ====================


class PromptSimulateRequest(BaseModel):
    prompt_name: str = Field(
        "rag_system",
        description="프롬프트 이름 (rag_system, rag_legal_system, direct_system 등)",
    )
    query: str = Field(..., min_length=1, description="테스트 질문")
    context_chunks: list[dict] = Field(
        default_factory=list,
        description="컨텍스트 청크 [{content, metadata}]",
    )
    model_hint: str | None = Field(None, description="모델 힌트 (간결성 접미사)")


@router.post("/admin/prompts/simulate")
async def simulate_prompt(
    request: PromptSimulateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """프롬프트 시뮬레이션 — LLM 호출 없이 최종 프롬프트를 조립하여 반환."""
    pm = PromptManager()

    if request.context_chunks:
        system_prompt, user_prompt = pm.build_rag_prompt(
            query=request.query,
            context_chunks=request.context_chunks,
            model_hint=request.model_hint,
        )
        doc_type = pm.detect_document_type(request.context_chunks)
    else:
        system_prompt, user_prompt = pm.build_direct_prompt(request.query)
        doc_type = "direct"

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "detected_doc_type": doc_type,
        "prompt_length_chars": len(system_prompt) + len(user_prompt),
        "system_prompt_length": len(system_prompt),
        "user_prompt_length": len(user_prompt),
    }


# ==================== RAG Settings Endpoint ====================


@router.get("/admin/rag-settings")
async def get_rag_settings(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """현재 RAG 런타임 설정값 조회."""
    return {
        "context_max_chunks": settings.context_max_chunks,
        "llm_max_tokens": settings.llm_max_tokens,
        "llm_temperature": settings.llm_temperature,
        "use_rerank": settings.use_rerank,
        "retrieval_top_k": settings.retrieval_top_k,
        "rerank_top_n": settings.rerank_top_n,
        "multi_query_enabled": settings.multi_query_enabled,
        "self_rag_enabled": settings.self_rag_enabled,
        "agentic_rag_enabled": settings.agentic_rag_enabled,
        "bm25_weight": settings.bm25_weight,
        "vector_weight": settings.vector_weight,
        "few_shot_max_examples": settings.few_shot_max_examples,
        "confidence_low": settings.confidence_low,
        "confidence_high": settings.confidence_high,
        "ocr_max_chars": settings.ocr_max_chars,
    }


class RagSettingsUpdateRequest(BaseModel):
    context_max_chunks: int | None = Field(None, ge=1, le=30)
    llm_max_tokens: int | None = Field(None, ge=64, le=8192)
    llm_temperature: float | None = Field(None, ge=0.0, le=1.0)
    use_rerank: bool | None = None
    retrieval_top_k: int | None = Field(None, ge=1, le=100)
    rerank_top_n: int | None = Field(None, ge=1, le=30)
    multi_query_enabled: bool | None = None
    self_rag_enabled: bool | None = None
    agentic_rag_enabled: bool | None = None
    bm25_weight: float | None = Field(None, ge=0.0, le=1.0)
    vector_weight: float | None = Field(None, ge=0.0, le=1.0)
    few_shot_max_examples: int | None = Field(None, ge=0, le=10)
    confidence_low: float | None = Field(None, ge=0.0, le=1.0)
    confidence_high: float | None = Field(None, ge=0.0, le=1.0)
    ocr_max_chars: int | None = Field(None, ge=5000, le=200000)


@router.put("/admin/rag-settings")
async def update_rag_settings(
    request: RagSettingsUpdateRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """RAG 런타임 설정 변경. 서버 재시작 시 .env 기본값으로 복원."""
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="변경할 설정을 하나 이상 지정해야 합니다.")

    for key, value in updates.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    logger.info(f"RAG settings updated by {current_user.username}: {updates}")
    return {
        "status": "updated",
        "changes": updates,
        "note": "런타임 변경만 적용됩니다. 서버 재시작 시 .env 기본값으로 초기화됩니다.",
    }


# ==================== Recent Errors Endpoint ====================


@router.get("/admin/recent-errors")
async def get_recent_errors(
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    """최근 WARNING/ERROR 로그 조회 (인메모리 링버퍼)."""
    from monitoring.log_handler import get_memory_log_handler

    handler = get_memory_log_handler()
    entries = handler.get_recent(limit=limit)
    return {"errors": entries, "total": len(entries)}


@router.post("/admin/playground", response_model=PlaygroundResponse)
async def llm_playground(
    request: PlaygroundRequest,
    current_user: Annotated[User, Depends(require_role([Role.ADMIN]))],
):
    """LLM 직접 테스트 (RAG 파이프라인 우회, ADMIN only)."""
    import time

    from core.llm import create_llm

    try:
        llm = create_llm(
            model=request.model or None,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 초기화 실패: {e}")

    start = time.monotonic()
    try:
        result = await llm.generate(
            prompt=request.prompt,
            system=request.system_prompt or None,
        )
    except Exception as e:
        logger.error(f"Playground LLM error for {current_user.username}: {e}")
        raise HTTPException(status_code=502, detail=f"LLM 응답 실패: {e}")
    finally:
        await llm.close()

    latency_ms = int((time.monotonic() - start) * 1000)
    tokens_used: int | None = None
    if result.usage:
        tokens_used = result.usage.get("completion_tokens") or result.usage.get("total_tokens")

    logger.info(
        f"Playground request by {current_user.username}: model={result.model}, "
        f"latency={latency_ms}ms, tokens={tokens_used}"
    )

    return PlaygroundResponse(
        response=result.content,
        latency_ms=latency_ms,
        model_used=result.model,
        tokens_used=tokens_used,
    )


# ==================== Batch Evaluate (Golden Dataset) ====================


@router.get("/batch-evaluate/datasets")
async def list_eval_datasets(
    current_user: Annotated[User, Depends(require_role(Role.ADMIN))],
):
    """List available golden dataset files."""
    from rag.batch_evaluator import BatchEvaluator
    return {"datasets": BatchEvaluator.list_datasets()}


@router.get("/batch-evaluate/history")
async def list_eval_history(
    current_user: Annotated[User, Depends(require_role(Role.ADMIN))],
):
    """List previous batch evaluation results."""
    from rag.batch_evaluator import BatchEvaluator
    return {"history": BatchEvaluator.list_history()}


@router.get("/batch-evaluate/history/{filename}")
async def get_eval_history_result(
    filename: str,
    current_user: Annotated[User, Depends(require_role(Role.ADMIN))],
):
    """Get a specific historical evaluation result."""
    from rag.batch_evaluator import BatchEvaluator
    result = BatchEvaluator.get_history_result(filename)
    if result is None:
        raise HTTPException(404, "결과 파일을 찾을 수 없습니다.")
    return result


@router.get("/batch-evaluate/stream")
async def stream_batch_evaluate(
    dataset: str = Query(..., description="Dataset name (stem of JSON file)"),
    limit: int | None = Query(None, ge=1, le=500, description="Max questions to evaluate"),
    current_user: User = Depends(require_role(Role.ADMIN)),
):
    """Run batch evaluation via SSE stream.

    Streams events: init, progress (per question), complete, error.
    """
    from rag.batch_evaluator import BatchEvaluator

    evaluator = BatchEvaluator()

    async def event_generator():
        try:
            async for event in evaluator.run_stream(dataset, limit):
                event_type = event.get("event", "message")
                data = json.dumps(event.get("data", {}), ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"
        except Exception as e:
            logger.error("Batch evaluate stream error: %s", e)
            error_data = json.dumps({"message": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
