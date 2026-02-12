"""MinIO object storage client for document file management."""

import asyncio
import logging
from functools import partial
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_storage_instance: "MinIOStorage | None" = None
_storage_initialized = False


class MinIOStorage:
    """MinIO object storage wrapper with async support."""

    def __init__(self):
        from minio import Minio

        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.default_bucket = settings.minio_bucket
        self._ensure_bucket(self.default_bucket)

    def _ensure_bucket(self, bucket: str) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info("Created MinIO bucket: %s", bucket)
        except Exception as e:
            logger.warning("Failed to ensure bucket %s: %s", bucket, e)

    def _run_sync(self, fn, *args, **kwargs):
        """Run synchronous MinIO operation."""
        return fn(*args, **kwargs)

    async def upload_file(
        self,
        file_path: str | Path,
        object_name: str | None = None,
        bucket: str | None = None,
        content_type: str = "application/octet-stream",
    ) -> str | None:
        """Upload a file to MinIO. Returns object name on success, None on failure."""
        bucket = bucket or self.default_bucket
        path = Path(file_path)
        object_name = object_name or path.name

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(
                    self.client.fput_object,
                    bucket,
                    object_name,
                    str(path),
                    content_type=content_type,
                ),
            )
            logger.info("Uploaded %s to MinIO: %s/%s", path.name, bucket, object_name)
            return object_name
        except Exception as e:
            logger.warning("MinIO upload failed for %s: %s", path.name, e)
            return None

    async def download_file(
        self,
        object_name: str,
        bucket: str | None = None,
    ) -> bytes | None:
        """Download file content from MinIO. Returns bytes or None on failure."""
        bucket = bucket or self.default_bucket

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(self.client.get_object, bucket, object_name),
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            logger.warning("MinIO download failed for %s: %s", object_name, e)
            return None

    async def delete_file(
        self,
        object_name: str,
        bucket: str | None = None,
    ) -> bool:
        """Delete a file from MinIO. Returns True on success."""
        bucket = bucket or self.default_bucket

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(self.client.remove_object, bucket, object_name),
            )
            logger.info("Deleted from MinIO: %s/%s", bucket, object_name)
            return True
        except Exception as e:
            logger.warning("MinIO delete failed for %s: %s", object_name, e)
            return False

    async def file_exists(
        self,
        object_name: str,
        bucket: str | None = None,
    ) -> bool:
        """Check if file exists in MinIO."""
        bucket = bucket or self.default_bucket

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(self.client.stat_object, bucket, object_name),
            )
            return True
        except Exception:
            return False


def get_storage() -> MinIOStorage | None:
    """Get MinIO storage instance. Returns None if disabled."""
    global _storage_instance, _storage_initialized

    if not settings.minio_enabled:
        return None

    if _storage_initialized:
        return _storage_instance

    try:
        _storage_instance = MinIOStorage()
        _storage_initialized = True
        logger.info("MinIO storage initialized: %s", settings.minio_endpoint)
    except Exception as e:
        logger.warning("MinIO storage unavailable (continuing without object storage): %s", e)
        _storage_instance = None
        _storage_initialized = True

    return _storage_instance
