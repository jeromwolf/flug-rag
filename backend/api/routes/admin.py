"""Admin endpoints: system info, LLM providers, prompt management."""

import logging

from fastapi import APIRouter

from api.schemas import LLMProviderInfo, PromptUpdateRequest, SystemInfoResponse
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


@router.get("/admin/info", response_model=SystemInfoResponse)
async def get_system_info():
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

    from agent import ConversationMemory

    memory = ConversationMemory()
    sessions = await memory.get_sessions(limit=1)
    session_count = len(sessions)

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


@router.get("/admin/providers", response_model=list[LLMProviderInfo])
async def get_providers():
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
async def get_prompts():
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
async def update_prompt(request: PromptUpdateRequest):
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
