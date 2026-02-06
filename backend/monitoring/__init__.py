"""Monitoring and observability for flux-rag (Prometheus + Grafana)."""

from .metrics import (
    active_connections,
    cache_hits_total,
    cache_misses_total,
    document_count,
    http_request_duration_seconds,
    http_requests_total,
    llm_request_duration_seconds,
    llm_tokens_total,
    rag_relevance_score,
    rag_retrieval_duration_seconds,
)
from .middleware import PrometheusMiddleware

__all__ = [
    "PrometheusMiddleware",
    "http_requests_total",
    "http_request_duration_seconds",
    "llm_request_duration_seconds",
    "llm_tokens_total",
    "rag_retrieval_duration_seconds",
    "rag_relevance_score",
    "cache_hits_total",
    "cache_misses_total",
    "active_connections",
    "document_count",
]
