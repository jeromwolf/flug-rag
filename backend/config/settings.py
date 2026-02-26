"""Central configuration using Pydantic BaseSettings."""

from pathlib import Path
from typing import Literal, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "flux-rag"
    app_version: str = "0.1.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]  # Override with CORS_ORIGINS env var

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/sqlite.db"

    # Database - PostgreSQL (production)
    postgres_dsn: str = "postgresql://flux_rag:flux_rag@localhost:5432/flux_rag"
    database_backend: str = "sqlite"  # "sqlite" or "postgres"

    # LLM - vLLM (primary)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str = "token-placeholder"

    # LLM - Ollama (dev/test)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # LLM - OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # LLM - Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Default LLM provider
    default_llm_provider: Literal["vllm", "ollama", "openai", "anthropic"] = "vllm"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024

    # Vector DB - ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # Vector DB - Milvus (production)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_index_type: str = "IVF_FLAT"
    milvus_metric_type: str = "COSINE"

    # RAG - Chunking
    chunk_strategy: str = "recursive"  # "recursive" or "semantic"
    chunk_size: int = 800
    chunk_overlap: int = 80
    semantic_breakpoint_threshold: float = 0.5  # for semantic chunking: similarity drop threshold
    semantic_min_chunk_size: int = 100  # for semantic chunking: minimum chunk size (chars)

    # RAG - Retrieval
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    confidence_high: float = 0.8
    confidence_low: float = 0.5

    # RAG - Score filtering
    retrieval_score_threshold: float = 0.15  # minimum retrieval score (0 = no filtering)
    context_max_chunks: int = 0  # max chunks for LLM context (0 = use rerank_top_n)

    # RAG - LLM control
    llm_max_tokens: int = 1024  # Optimized via parametric grid search
    llm_temperature: float = 0.2  # Balanced: 0.1=negative-optimal, 0.3=factual-optimal

    # RAG - Query expansion
    query_expansion_enabled: bool = False  # HyDE (Hypothetical Document Embeddings)

    # RAG - BM25 tuning
    bm25_k1: float = 1.5  # BM25 term frequency saturation
    bm25_b: float = 0.75  # BM25 document length normalization

    # RAG - Reranking
    use_rerank: bool = True  # enable/disable reranking

    # RAG - Advanced techniques
    multi_query_enabled: bool = False  # Multi-query retrieval (multiple perspectives)
    multi_query_count: int = 3  # Number of alternative queries to generate
    self_rag_enabled: bool = True  # Self-RAG (self-reflective RAG with hallucination check)
    self_rag_max_retries: int = 1  # Max retries if answer is not grounded
    agentic_rag_enabled: bool = False  # Agentic RAG (dynamic strategy routing)

    # OCR - Upstage
    upstage_api_key: str = ""
    upstage_api_url: str = "https://api.upstage.ai/v1/document-ai/document-parse"
    ocr_provider: str = "cloud"  # "cloud" or "onprem"
    ocr_onprem_url: str = "http://localhost:8501"

    # OCR Training Data Collection
    ocr_training_enabled: bool = False
    ocr_training_dir: str = "./data/ocr_training"
    ocr_training_image_dpi: int = 150

    # Auth (legacy fields kept for backward compatibility)
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 480
    algorithm: str = "HS256"

    # Auth - JWT / SSO / EAM
    jwt_secret_key: str = "change-me-in-production-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    auth_enabled: bool = False  # Set True to enforce authentication

    trusted_proxy_ips: str = ""  # Comma-separated trusted proxy IPs (e.g., "10.0.0.1,10.0.0.2")

    # Auth - LDAP / Active Directory
    ldap_server_url: str = ""
    ldap_base_dn: str = ""
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""

    # File Upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_dir: str = "./data/uploads"
    allowed_extensions: set[str] = {".pdf", ".hwp", ".docx", ".xlsx", ".pptx", ".txt"}

    # Sync / Scheduler
    sync_enabled: bool = False
    sync_cron: str = "0 2 * * *"  # 매일 새벽 2시
    sync_watch_dirs: list[str] = []
    sync_batch_size: int = 10

    # Agent
    max_history: int = 5

    # Redis / Cache
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    cache_enabled: bool = False  # Disabled by default for demo/dev
    cache_default_ttl: int = 300  # 5 minutes

    # MinIO Object Storage
    minio_enabled: bool = False  # Enable MinIO for document storage
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "flux-rag-documents"
    minio_secure: bool = False  # True for HTTPS

    # Monitoring
    prometheus_enabled: bool = False  # Disabled by default

    # Batch Inference
    batch_inference_enabled: bool = False
    batch_size: int = 8
    batch_max_wait_ms: int = 100

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    prompts_dir: Path = Path(__file__).resolve().parent.parent / "prompts"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, list[str]]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
