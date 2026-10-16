# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The caller's identity, as resolved from a validated credential."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Principal(BaseModel):
    """Who is calling, and how much they can see.

    Resolved in two stages (see ``api/deps.py``): stage one
    (:class:`autotunex.core.auth.protocols.Authenticator`) sets ``email`` and
    ``provider`` only, with no database access. Stage two
    (``get_principal``) resolves ``user_id`` and ``is_admin`` from ``users``.
    Standalone mode can carry ``is_admin=True`` with no database row at all —
    see the design spec's standalone semantics.
    """

    model_config = ConfigDict(frozen=True)

    email: str | None
    provider: str
    user_id: UUID | None = None
    is_admin: bool = False
    impersonator: str | None = None
    """When set, an admin is impersonating another user.

    ``email`` and ``user_id`` are the *target's* (effective) identity, ``is_admin``
    is the *real admin's* flag (preserved), and this holds the real admin's email.
    ``None`` for an ordinary, non-impersonated principal. Resolved by
    ``get_effective_principal`` (``api/deps.py``); never set by ``get_principal``.
    """
