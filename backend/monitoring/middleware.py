"""FastAPI middleware for automatic Prometheus HTTP metrics collection."""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .metrics import (
    PROMETHEUS_AVAILABLE,
    active_connections,
    http_request_duration_seconds,
    http_requests_total,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records HTTP request count and latency as Prometheus metrics.

    Also tracks active connections via a gauge.
    Skips the /metrics and /health endpoints to avoid noise.
    """

    SKIP_PATHS = {"/metrics", "/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip instrumentation for metrics/health endpoints
        if path in self.SKIP_PATHS:
            return await call_next(request)

        # Normalize path: collapse path params to placeholders
        endpoint = self._normalize_path(path)
        method = request.method

        active_connections.inc()
        start_time = time.time()

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.time() - start_time
            active_connections.dec()

            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status,
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path by replacing UUID-like segments with {id}.

        Prevents high-cardinality labels in Prometheus.
        """
        import re

        parts = path.strip("/").split("/")
        normalized = []
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
        )
        for part in parts:
            if uuid_pattern.match(part) or part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/" + "/".join(normalized)


def create_metrics_endpoint():
    """Create a /metrics endpoint for Prometheus scraping.

    Returns a FastAPI route function. Add it to your app::

        from monitoring.middleware import create_metrics_endpoint
        app.add_route("/metrics", create_metrics_endpoint())
    """
    if not PROMETHEUS_AVAILABLE:

        async def metrics_unavailable(request: Request) -> Response:
            return Response(
                content="prometheus_client not installed",
                status_code=501,
                media_type="text/plain",
            )

        return metrics_unavailable

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    async def metrics(request: Request) -> Response:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return metrics
