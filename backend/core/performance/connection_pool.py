"""Unified HTTP connection pool management for external services."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default pool limits
DEFAULT_MAX_CONNECTIONS = 20
DEFAULT_MAX_KEEPALIVE = 10
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 10.0


class ConnectionPoolManager:
    """Manages httpx.AsyncClient instances with connection pooling.

    Provides a single pool per base URL so multiple callers share connections.

    Usage::

        pool = ConnectionPoolManager()
        client = pool.get_client("http://vllm:8000")
        response = await client.post("/v1/completions", json={...})
        await pool.close_all()
    """

    def __init__(
        self,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
        timeout: float = DEFAULT_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ):
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._clients: dict[str, httpx.AsyncClient] = {}

    def get_client(
        self,
        base_url: str,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> httpx.AsyncClient:
        """Get or create an AsyncClient for the given base URL.

        Args:
            base_url: The base URL for the HTTP client.
            headers: Default headers for all requests.
            timeout: Override timeout for this client.

        Returns:
            An httpx.AsyncClient with connection pooling.
        """
        if base_url in self._clients:
            return self._clients[base_url]

        effective_timeout = timeout or self._timeout
        pool_limits = httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_keepalive,
        )
        client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or {},
            limits=pool_limits,
            timeout=httpx.Timeout(effective_timeout, connect=self._connect_timeout),
        )
        self._clients[base_url] = client
        logger.debug("Created connection pool for %s", base_url)
        return client

    async def health_check(self, base_url: str, path: str = "/health") -> bool:
        """Check if a service is reachable."""
        try:
            client = self.get_client(base_url)
            resp = await client.get(path)
            return resp.status_code < 500
        except Exception as e:
            logger.warning("Health check failed for %s: %s", base_url, e)
            return False

    async def close(self, base_url: str) -> None:
        """Close the client for a specific base URL."""
        client = self._clients.pop(base_url, None)
        if client is not None:
            await client.aclose()

    async def close_all(self) -> None:
        """Close all managed clients."""
        for url, client in list(self._clients.items()):
            try:
                await client.aclose()
            except Exception as e:
                logger.warning("Error closing client for %s: %s", url, e)
        self._clients.clear()

    @property
    def active_pools(self) -> list[str]:
        """List of base URLs with active connection pools."""
        return list(self._clients.keys())
