# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""User-management business logic.

Owns the domain rules for the user-management endpoints: the role-change
guardrails (an admin may not change their own role, and the last administrator
may not be demoted) and the caller's own usage metadata. Knows nothing about
HTTP — it raises the exceptions in :mod:`autotunex.core.exceptions` for the API
layer to translate.

The coarse "is the caller an admin at all" gate is *not* here: it is the
``require_admin`` route dependency in ``api/deps.py``. A user is an identity,
not an owned row, so there is no ``?scope=all`` widening — the list/get/set_role
methods assume an admin caller (guaranteed by that gate), and ``my_metadata``
serves the caller's own view.
"""

from __future__ import annotations

from uuid import UUID

from autotunex.core.config import ADMIN_ROLE
from autotunex.core.exceptions import (
    CannotChangeOwnRoleError,
    LastAdminError,
    UserNotFoundError,
)
from autotunex.db.repositories.protocols import UserRepository
from autotunex.models.auth import Principal
from autotunex.models.common import Page
from autotunex.models.user import Role, UserMetadata, UserRead
from autotunex.services.mappers import user_to_read


class UserService:
    """Admin-only user management plus a self-service metadata view."""

    def __init__(self, repository: UserRepository, principal: Principal) -> None:
        self._repository = repository
        self._principal = principal

    async def list(self, *, limit: int = 20, offset: int = 0) -> Page[UserRead]:
        """Return one page of users, newest first."""
        users, total = await self._repository.list(limit=limit, offset=offset)
        return Page[UserRead](
            items=[user_to_read(user) for user in users],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get(self, user_id: UUID) -> UserRead:
        """Return the user with ``user_id``.

        Raises:
            UserNotFoundError: no such user.
        """
        user = await self._repository.get(user_id)
        if user is None:
            raise UserNotFoundError(user_id)
        return user_to_read(user)

    async def set_role(self, user_id: UUID, role: Role) -> UserRead:
        """Change a user's role, enforcing the guardrails in order.

        Order matters: an unknown id is a 404 before any other check; changing
        your own role is refused next; and demoting the final admin is refused
        last. See the design spec for why the last-admin guard mostly protects
        the ``user_id=None`` admin case.

        Raises:
            UserNotFoundError: no such user (also if it vanished mid-call).
            CannotChangeOwnRoleError: the target is the calling admin.
            LastAdminError: the change would leave no administrators.
        """
        user = await self._repository.get(user_id)
        if user is None:
            raise UserNotFoundError(user_id)
        if self._principal.user_id is not None and user.id == self._principal.user_id:
            raise CannotChangeOwnRoleError()
        if (
            user.role == ADMIN_ROLE
            and role is not Role.ADMIN
            and await self._repository.count_admins() <= 1
        ):
            raise LastAdminError()
        updated = await self._repository.set_role(user_id, role.value)
        if updated is None:
            raise UserNotFoundError(user_id)
        return user_to_read(updated)

    async def my_metadata(self) -> UserMetadata:
        """Return the calling principal's own usage counts.

        A caller with no resolvable ``user_id`` (unprovisioned) owns nothing, so
        this returns zeros without a database call — it is the caller's own view,
        so zeros leak nothing and no 403 applies.
        """
        if self._principal.user_id is None:
            return UserMetadata(number_of_jobs=0, number_of_configurations=0, number_of_datasets=0)
        jobs, configurations, datasets = await self._repository.metadata(self._principal.user_id)
        return UserMetadata(
            number_of_jobs=jobs,
            number_of_configurations=configurations,
            number_of_datasets=datasets,
        )
