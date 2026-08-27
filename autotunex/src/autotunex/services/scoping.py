# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Ownership-scope resolution shared by every owner-scoped service.

One rule, one place: jobs, configurations and datasets all resolve "whose rows
may this request see?" here, so the read/update/delete paths cannot drift apart.
Pure functions, not a base class — CLAUDE.md forbids service base classes, and
this keeps the rule as reviewable as ``ALLOWED_JOB_TRANSITIONS``. No HTTP, no SQL.
"""

from __future__ import annotations

from uuid import UUID

from autotunex.core.exceptions import ScopeNotPermittedError
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope


def resolve_owner_filter(principal: Principal, scope: DataScope) -> UUID | None:
    """Return the ownership filter to pass to the repository for this request.

    - ``scope=ALL`` by an admin -> ``None`` (the repository reads ``None`` as
      "no ownership filter": the cross-user view).
    - ``scope=ALL`` by a non-admin -> raises :class:`ScopeNotPermittedError`.
    - ``scope=OWN`` (anyone, admin included) -> ``principal.user_id`` (may be
      ``None`` when the caller has no resolvable identity).

    A ``None`` return is therefore ambiguous on its own and must be read
    together with ``scope``: under ``ALL`` it means "admin, unscoped"; under
    ``OWN`` it means "no resolvable identity", and the caller must short-circuit
    to an empty page / 404 (see :func:`sees_nothing`) rather than hand ``None``
    to the repository, which would leak the whole table.

    Raises:
        ScopeNotPermittedError: a non-admin requested ``DataScope.ALL``.
    """
    if scope is DataScope.ALL:
        if not principal.is_admin:
            raise ScopeNotPermittedError()
        return None
    return principal.user_id


def sees_nothing(principal: Principal, scope: DataScope) -> bool:
    """Whether an ``OWN``-scope caller has no resolvable identity to filter by.

    Call this *after* :func:`resolve_owner_filter`, so the non-admin ``ALL``
    403 fires first. It is only ever ``True`` for ``scope=OWN``: an ``ALL``
    request has already either raised (non-admin) or returned the unscoped view
    (admin). When ``True`` the caller returns an empty page (list) or raises
    the resource's not-found error (get/update/delete).
    """
    return scope is DataScope.OWN and principal.user_id is None
