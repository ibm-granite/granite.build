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

from autotunex.core.constants import SYSTEM_USER_ID
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


def is_system_owned(owner_user_id: str | UUID | None) -> bool:
    """Whether ``owner_user_id`` is the reserved system user's.

    ``configurations.user_id`` / ``datasets.user_id`` are ``VARCHAR`` columns, so
    a row's owner arrives as a string while a principal's arrives as a ``UUID``;
    both are compared through ``str()`` here rather than at each call site. The
    comparison is exact, matching the unfolded string predicate the repository
    filters with (``_owner_or_shared``) — the constant is already canonical
    lowercase and folding case would only invent a second, looser notion of
    "is this the system row" for the two to drift apart on.
    """
    return owner_user_id is not None and str(owner_user_id) == str(SYSTEM_USER_ID)


def is_delete_protected(principal: Principal, owner_user_id: str | UUID | None) -> bool:
    """Whether a row owned by ``owner_user_id`` is undeletable by ``principal``.

    The shared tier is readable by everyone and deletable by (almost) no one: a
    system-owned row is starter content the whole deployment launches from, and
    removing it takes it from every caller at once. Ownership scoping alone does
    not express that — ``resolve_owner_filter`` hands an admin who asked for
    ``scope=all`` an unfiltered ``owner_id=None``, and a caller whose own identity
    resolves to the system row passes the strict filter outright — so the rule is
    stated here, once, and enforced before any scope is consulted.

    The single exemption is an *active impersonation overlay* onto the system user:
    ``impersonator`` is only ever set by ``api.deps.get_effective_principal`` for a
    genuinely-admin caller presenting a valid ``autotunex_assume`` cookie, and
    ``POST /auth/assume`` logs it. Requiring the overlay rather than merely
    ``user_id == SYSTEM_USER_ID`` is the load-bearing half: it keeps a deployment
    whose ambient principal happens to *be* the system owner — the single-owner
    standalone case — from handing every caller a delete on shared content.

    Update is deliberately not covered: editing curated content in place is
    recoverable and stays available to an admin via ``scope=all``, while a delete
    is not. See :class:`~autotunex.core.exceptions.SystemResourceProtectedError`.
    """
    if not is_system_owned(owner_user_id):
        return False
    is_assuming_system_user = principal.impersonator is not None and is_system_owned(
        principal.user_id
    )
    return not is_assuming_system_user
