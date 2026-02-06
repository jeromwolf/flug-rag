"""Prometheus metrics definitions for flux-rag.

All metrics use the 'fluxrag_' prefix for namespace isolation.
These are module-level singletons; import and use directly.
"""

try:
    from prometheus_client import Counter, Gauge, Histogram

    # --- HTTP metrics ---
    http_requests_total = Counter(
        "fluxrag_http_requests_total",
        "Total HTTP requests",
        labelnames=["method", "endpoint", "status"],
    )

    http_request_duration_seconds = Histogram(
        "fluxrag_http_request_duration_seconds",
        "HTTP request latency in seconds",
        labelnames=["method", "endpoint"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    # --- LLM metrics ---
    llm_request_duration_seconds = Histogram(
        "fluxrag_llm_request_duration_seconds",
        "LLM inference latency in seconds",
        labelnames=["provider", "model"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )

    llm_tokens_total = Counter(
        "fluxrag_llm_tokens_total",
        "Total LLM tokens processed",
        labelnames=["provider", "direction"],  # direction: input / output
    )

    # --- RAG metrics ---
    rag_retrieval_duration_seconds = Histogram(
        "fluxrag_rag_retrieval_duration_seconds",
        "RAG retrieval latency in seconds",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    rag_relevance_score = Histogram(
        "fluxrag_rag_relevance_score",
        "Distribution of RAG relevance scores",
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )

    # --- Cache metrics ---
    cache_hits_total = Counter(
        "fluxrag_cache_hits_total",
        "Total cache hits",
        labelnames=["cache_name"],
    )

    cache_misses_total = Counter(
        "fluxrag_cache_misses_total",
        "Total cache misses",
        labelnames=["cache_name"],
    )

    # --- System metrics ---
    active_connections = Gauge(
        "fluxrag_active_connections",
        "Number of active WebSocket/SSE connections",
    )

    document_count = Gauge(
        "fluxrag_document_count",
        "Total number of documents in the vectorstore",
    )

    PROMETHEUS_AVAILABLE = True

except ImportError:
    # Fallback stubs when prometheus_client is not installed.
    # All operations become no-ops so application code does not need guards.

    class _NoopMetric:
        """No-op metric stub."""

        def labels(self, **kwargs):
            return self

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

    http_requests_total = _NoopMetric()
    http_request_duration_seconds = _NoopMetric()
    llm_request_duration_seconds = _NoopMetric()
    llm_tokens_total = _NoopMetric()
    rag_retrieval_duration_seconds = _NoopMetric()
    rag_relevance_score = _NoopMetric()
    cache_hits_total = _NoopMetric()
    cache_misses_total = _NoopMetric()
    active_connections = _NoopMetric()
    document_count = _NoopMetric()

    PROMETHEUS_AVAILABLE = False
