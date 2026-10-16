# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Standalone mode's ``Authenticator``.

Returns a fixed :class:`Principal` from settings — email from
``standalone_email`` or the ``SYSTEM_STANDALONE_EMAIL`` sentinel, ignoring
every credential — the shape both reference implementations independently
converged on: a permissive implementation of the normal seam, never a
separate code path.
"""

from __future__ import annotations

from autotunex.core.config import ADMIN_ROLE, Settings
from autotunex.models.auth import Principal

STANDALONE_PROVIDER = "standalone"
"""The ``Principal.provider`` value standalone mode issues.

``api/deps.get_principal`` branches on this to decide that ``standalone_role``
outranks the matched ``users.role``, so it is a constant rather than a literal
repeated in two modules.
"""

SYSTEM_STANDALONE_EMAIL = "standalone@autotunex.local"
"""The default owner email standalone mode attributes writes to.

A reserved ``.local`` address that can never be a real person, so the ``users``
row it maps to reads unmistakably as a system account. Overridden by setting
``standalone_email`` to a chosen value. Its ``users`` row is provisioned lazily
by ``api.deps.get_principal`` (see the design spec), which is why standalone
writes succeed instead of being refused for want of an owner.
"""


class DisabledAuthenticator:
    """Satisfies :class:`autotunex.core.auth.protocols.Authenticator`."""

    def __init__(self, settings: Settings) -> None:
        self._email = settings.standalone_email or SYSTEM_STANDALONE_EMAIL
        self._is_admin = settings.standalone_role == ADMIN_ROLE

    async def authenticate(
        self, *, bearer: str | None, api_key: str | None, session: str | None
    ) -> Principal:
        """Return the standalone principal, ignoring every credential.

        The owner email is ``standalone_email`` if set, else the default
        ``SYSTEM_STANDALONE_EMAIL`` sentinel — never ``None``, so stage two
        (``api.deps.get_principal``) can resolve and lazily provision a ``users``
        row and standalone writes have an owner. ``is_admin`` comes from
        ``standalone_role`` (the setting wins over the row). Reads are scoped to
        the owner's own rows either way — the default ``admin`` only grants the
        *ability* to widen to every row with ``?scope=all``, which
        ``standalone_role="user"`` refuses with a 403.
        """
        return Principal(
            email=self._email,
            provider=STANDALONE_PROVIDER,
            is_admin=self._is_admin,
        )
