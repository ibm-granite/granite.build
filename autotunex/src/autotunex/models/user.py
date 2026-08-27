# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""User schemas.

A *user* is an identity that owns configurations, datasets and jobs — not an
owned resource itself. These schemas back the admin-only management endpoints
(``GET /users``, ``GET /users/{id}``, ``PATCH /users/{id}``) and the
self-service ``GET /users/me/metadata``. See
``docs/superpowers/specs/2026-08-10-user-management-endpoints-design.md``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    """The two roles a user row may carry.

    Values match ``core/config.ADMIN_ROLE`` and ``_KNOWN_ROLES``. Used as the
    strict input type on :class:`UserRoleUpdate`, so any other value is a 422 at
    request validation; reads stay lenient (see :attr:`UserRead.role`).
    """

    ADMIN = "admin"
    USER = "user"


class UserRoleUpdate(BaseModel):
    """Request body for ``PATCH /users/{user_id}`` — a role change, nothing else.

    Email is an identity key linked to the auth provider and is deliberately not
    editable here (design spec, decision 4).
    """

    model_config = ConfigDict(extra="forbid")

    role: Role


class UserRead(BaseModel):
    """A user as returned by every user endpoint.

    ``role`` is ``str | None`` on read even though writes use the strict
    :class:`Role` enum: the column is nullable and may hold a legacy or
    pipeline-written value, and a read must not choke on it — the same
    strict-in / lenient-out stance as ``ConfigurationRead.config_data``.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    role: str | None = None
    created_at: datetime
    updated_at: datetime


class UserMetadata(BaseModel):
    """Per-user counts returned by ``GET /users/me/metadata``.

    Field names kept from the 2025 repo's ``UserMetadata`` for continuity.
    """

    number_of_jobs: int
    number_of_configurations: int
    number_of_datasets: int
