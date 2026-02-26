"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    security_logger = logging.getLogger("flux-rag.security")

    # Security checks
    # C-02: JWT secret check – block startup with default secret
    if settings.auth_enabled and settings.jwt_secret_key in (
        "change-me-in-production-jwt-secret", "change-me-in-production",
    ):
        raise RuntimeError(
            "FATAL: Default JWT secret key detected with AUTH_ENABLED=true. "
            "Set JWT_SECRET_KEY environment variable to a secure random value."
        )

    # C-03: Default password warning
    if settings.auth_enabled:
        security_logger.warning(
            "WARNING: DEFAULT CREDENTIALS DETECTED: Demo users (admin/manager/user/viewer) "
            "have default passwords. Change them immediately in production! "
            "Use POST /api/auth/change-password to update."
        )

    # H-07: Auth disabled – strong warning
    if not settings.auth_enabled:
        for _ in range(3):
            security_logger.critical(
                ">>> AUTH_ENABLED=false: ALL endpoints accessible WITHOUT authentication. "
                "Set AUTH_ENABLED=true for production. <<<"
            )

    # 1. Cache initialization
    if settings.cache_enabled:
        logger.info("Cache enabled; initializing cache backend...")
        try:
            from core.cache import get_cache
            cache = await get_cache()
            logger.info("Cache initialized: %s", type(cache).__name__)
        except Exception as e:
            logger.warning("Cache initialization failed (continuing without cache): %s", e)

    # 2. Batch processor initialization
    if settings.batch_inference_enabled:
        logger.info("Batch inference enabled; initializing batch processor...")
        try:
            from core.performance import BatchProcessor
            from core.embeddings.local import LocalEmbedding

            embedder = LocalEmbedding()
            batch_processor = BatchProcessor(
                process_fn=embedder.embed_texts,
                batch_size=settings.batch_size,
                max_wait_ms=settings.batch_max_wait_ms,
                name="embedding",
            )
            await batch_processor.start()
            app.state.batch_processor = batch_processor

            from core.embeddings.local import set_embedding_batch_processor
            set_embedding_batch_processor(batch_processor)

            logger.info("Batch processor started (batch_size=%d, max_wait=%dms)",
                       settings.batch_size, settings.batch_max_wait_ms)
        except Exception as e:
            logger.warning("Batch processor initialization failed: %s", e)

    # 3. Connection pool initialization
    try:
        from core.performance import ConnectionPoolManager
        pool_manager = ConnectionPoolManager()
        app.state.pool_manager = pool_manager

        from core.llm.ollama_provider import set_llm_pool_manager
        set_llm_pool_manager(pool_manager)

        logger.info("Connection pool manager initialized")
    except Exception as e:
        logger.warning("Connection pool initialization failed: %s", e)

    yield

    # Shutdown
    # Close batch processor
    if hasattr(app.state, "batch_processor"):
        try:
            await app.state.batch_processor.stop()
        except Exception:
            pass

    # Close connection pools
    if hasattr(app.state, "pool_manager"):
        try:
            await app.state.pool_manager.close_all()
        except Exception:
            pass

    # Close cache connections
    try:
        from core.cache import _cache_instance, reset_cache
        if _cache_instance is not None and hasattr(_cache_instance, "close"):
            await _cache_instance.close()
        reset_cache()
    except Exception:
        pass


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


cors_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

app.add_middleware(SecurityHeadersMiddleware)

# Prometheus metrics middleware and /metrics endpoint (optional)
if settings.prometheus_enabled:
    try:
        from monitoring.middleware import PrometheusMiddleware, create_metrics_endpoint

        app.add_middleware(PrometheusMiddleware)
        app.add_route("/metrics", create_metrics_endpoint())
        logger.info("Prometheus metrics enabled at /metrics")
    except Exception as e:
        logger.warning("Failed to enable Prometheus metrics: %s", e)

# Register auth routes (login/refresh are public; other auth endpoints use their own deps)
from auth.routes import router as auth_router

app.include_router(auth_router, prefix="/api", tags=["auth"])

# Register application routes
from api.routes import admin, agents, chat, content, documents, feedback, folders, guardrails, logs, mcp, ocr, ocr_training, personal_knowledge, quality, sessions, statistics, sync, workflows

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(mcp.router, prefix="/api", tags=["mcp"])
app.include_router(workflows.router, prefix="/api", tags=["workflows"])
app.include_router(agents.router, prefix="/api", tags=["agents"])
app.include_router(quality.router, prefix="/api", tags=["quality"])
app.include_router(sync.router, prefix="/api", tags=["sync"])
app.include_router(folders.router, prefix="/api", tags=["folders"])
app.include_router(personal_knowledge.router, prefix="/api", tags=["personal-knowledge"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
app.include_router(statistics.router, prefix="/api", tags=["statistics"])
app.include_router(logs.router, prefix="/api", tags=["logs"])
app.include_router(guardrails.router, prefix="/api", tags=["guardrails"])
app.include_router(content.router, prefix="/api", tags=["content"])
app.include_router(ocr_training.router, prefix="/api", tags=["ocr-training"])


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "auth_enabled": settings.auth_enabled,
    }
