"""LDAP/Active Directory authentication provider (stub).

This module provides a pluggable LDAP authentication backend.  In the current
demo configuration it always falls back to the local ``UserStore`` because the
``ldap_server_url`` setting is empty by default.

To enable real LDAP authentication install ``ldap3`` and configure the LDAP
settings in ``.env``::

    LDAP_SERVER_URL=ldap://ad.kogas-tech.co.kr:389
    LDAP_BASE_DN=DC=kogas-tech,DC=co,DC=kr
    LDAP_BIND_DN=CN=svc-ai,OU=ServiceAccounts,DC=kogas-tech,DC=co,DC=kr
    LDAP_BIND_PASSWORD=secret
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LDAPUserInfo:
    """Minimal user info returned by an LDAP bind."""

    username: str
    email: str
    full_name: str
    department: str
    groups: list[str]


class LDAPAuthProvider:
    """LDAP / Active Directory authentication provider.

    When LDAP settings are not configured (``ldap_server_url`` is empty),
    ``authenticate`` always returns ``None`` so the caller should fall back
    to the local user store.
    """

    def __init__(self) -> None:
        self.server_url: str = settings.ldap_server_url
        self.base_dn: str = settings.ldap_base_dn
        self.bind_dn: str = settings.ldap_bind_dn
        self.bind_password: str = settings.ldap_bind_password
        self._available = bool(self.server_url)

    @property
    def is_configured(self) -> bool:
        return self._available

    def authenticate(self, username: str, password: str) -> LDAPUserInfo | None:
        """Attempt to authenticate *username* / *password* against LDAP.

        Returns ``LDAPUserInfo`` on success, ``None`` on failure or when LDAP
        is not configured.
        """
        if not self._available:
            return None

        try:
            return self._ldap_bind(username, password)
        except Exception:
            logger.exception("LDAP authentication error for user %s", username)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ldap_bind(self, username: str, password: str) -> LDAPUserInfo | None:
        """Perform the actual LDAP bind.

        Requires the ``ldap3`` package.  If it is not installed the method
        logs a warning and returns ``None``.
        """
        try:
            import ldap3  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("ldap3 package not installed -- LDAP auth unavailable")
            return None

        server = ldap3.Server(self.server_url, get_info=ldap3.ALL)
        user_dn = f"CN={username},{self.base_dn}"

        conn = ldap3.Connection(server, user=user_dn, password=password)
        if not conn.bind():
            logger.info("LDAP bind failed for %s", username)
            return None

        # Search for user attributes
        search_filter = f"(sAMAccountName={username})"
        conn.search(
            self.base_dn,
            search_filter,
            attributes=["mail", "displayName", "department", "memberOf"],
        )

        if not conn.entries:
            conn.unbind()
            return None

        entry = conn.entries[0]
        info = LDAPUserInfo(
            username=username,
            email=str(entry.mail) if hasattr(entry, "mail") else "",
            full_name=str(entry.displayName) if hasattr(entry, "displayName") else username,
            department=str(entry.department) if hasattr(entry, "department") else "",
            groups=[str(g) for g in entry.memberOf] if hasattr(entry, "memberOf") else [],
        )
        conn.unbind()
        return info


# Module-level singleton
ldap_provider = LDAPAuthProvider()
