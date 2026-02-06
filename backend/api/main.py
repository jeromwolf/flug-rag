"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if settings.cache_enabled:
        logger.info("Cache enabled; initializing cache backend...")
        try:
            from core.cache import get_cache

            cache = await get_cache()
            logger.info("Cache initialized: %s", type(cache).__name__)
        except Exception as e:
            logger.warning("Cache initialization failed (continuing without cache): %s", e)
    yield
    # Shutdown: close cache connections
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

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
from api.routes import admin, agents, chat, documents, feedback, mcp, sessions, workflows

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(mcp.router, prefix="/api", tags=["mcp"])
app.include_router(workflows.router, prefix="/api", tags=["workflows"])
app.include_router(agents.router, prefix="/api", tags=["agents"])


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "auth_enabled": settings.auth_enabled,
    }
