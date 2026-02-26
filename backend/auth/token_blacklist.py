"""Token blacklist for JWT revocation.

In-memory blacklist with TTL-based auto-expiry.  Revoked JTIs are stored
alongside their original ``exp`` timestamp so they are automatically cleaned
up once the token would have expired anyway.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone


class TokenBlacklist:
    """In-memory token blacklist with TTL-based auto-expiry."""

    def __init__(self) -> None:
        self._blacklist: dict[str, datetime] = {}  # jti -> expires_at
        self._user_tokens: dict[str, set[str]] = {}  # user_id -> set of jtis
        self._lock = asyncio.Lock()
        self._cleanup_counter = 0

    async def add(
        self,
        jti: str,
        expires_at: datetime,
        user_id: str | None = None,
    ) -> None:
        """Add a token JTI to the blacklist.

        *expires_at* should match the token's ``exp`` claim so the entry is
        automatically purged after the token would have expired.
        """
        async with self._lock:
            self._blacklist[jti] = expires_at
            if user_id:
                self._user_tokens.setdefault(user_id, set()).add(jti)
            self._cleanup_counter += 1
            if self._cleanup_counter >= 100:
                self._cleanup()
                self._cleanup_counter = 0

    async def is_revoked(self, jti: str) -> bool:
        """Return ``True`` if *jti* has been revoked and has not yet expired."""
        return jti in self._blacklist and self._blacklist[jti] > datetime.now(
            timezone.utc
        )

    async def revoke_all_for_user(
        self,
        user_id: str,
        default_ttl_seconds: int = 1800,
    ) -> None:
        """Revoke all known tokens for *user_id*.

        Tokens already in the blacklist keep their original TTL.  Tokens that
        are tracked but not yet blacklisted receive *default_ttl_seconds*.
        """
        async with self._lock:
            jtis = self._user_tokens.get(user_id, set())
            now = datetime.now(timezone.utc)
            for jti in jtis:
                if jti not in self._blacklist:
                    self._blacklist[jti] = now + timedelta(seconds=default_ttl_seconds)

    # -- internal helpers ----------------------------------------------------

    def _cleanup(self) -> None:
        """Remove expired entries (called under lock)."""
        now = datetime.now(timezone.utc)
        expired = [jti for jti, exp in self._blacklist.items() if exp <= now]
        for jti in expired:
            del self._blacklist[jti]
        # Clean user mapping
        expired_set = set(expired)
        for uid in list(self._user_tokens):
            self._user_tokens[uid] -= expired_set
            if not self._user_tokens[uid]:
                del self._user_tokens[uid]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_blacklist: TokenBlacklist | None = None
_blacklist_lock = asyncio.Lock()


async def get_token_blacklist() -> TokenBlacklist:
    """Return the module-level :class:`TokenBlacklist` singleton."""
    global _blacklist
    if _blacklist is not None:
        return _blacklist
    async with _blacklist_lock:
        if _blacklist is None:
            _blacklist = TokenBlacklist()
    return _blacklist
