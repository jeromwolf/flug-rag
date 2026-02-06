"""Central configuration using Pydantic BaseSettings."""

from pathlib import Path
from typing import Literal

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

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 80
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    confidence_high: float = 0.8
    confidence_low: float = 0.5

    # OCR - Upstage
    upstage_api_key: str = ""
    upstage_api_url: str = "https://api.upstage.ai/v1/document-ai/document-parse"

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

    # Auth - LDAP / Active Directory
    ldap_server_url: str = ""
    ldap_base_dn: str = ""
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""

    # File Upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_dir: str = "./data/uploads"
    allowed_extensions: set[str] = {".pdf", ".hwp", ".docx", ".xlsx", ".pptx", ".txt"}

    # Agent
    max_history: int = 5

    # Redis / Cache
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    cache_enabled: bool = False  # Disabled by default for demo/dev
    cache_default_ttl: int = 300  # 5 minutes

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
