"""Object storage abstraction for document files."""

from .minio_client import MinIOStorage, get_storage

__all__ = ["MinIOStorage", "get_storage"]
