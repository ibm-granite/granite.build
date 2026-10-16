# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Shared API schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class DataScope(StrEnum):
    """How much a caller wants to see on an owner-scoped read.

    ``OWN`` (the default for every endpoint) restricts results to the caller's
    own rows. ``ALL`` removes the ownership filter and is honored only for an
    admin; a non-admin requesting it is refused (see
    :class:`autotunex.core.exceptions.ScopeNotPermittedError`). The parameter,
    not ``is_admin`` alone, is what unlocks the cross-user view — being an admin
    grants the *ability* to ask, not an automatic all-tenants result.
    """

    OWN = "own"
    ALL = "all"


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health`` and ``GET /health/live``."""

    status: str = "ok"
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness payload returned by ``GET /health/ready`` on success.

    A 200 means the service answered *and* a probe query against the database
    succeeded. A database outage returns 503 (a ``ProblemDetail``) instead, so an
    orchestrator can gate traffic on database reachability, not just liveness.
    """

    status: str = "ready"
    database: str = "ok"


class ProblemDetail(BaseModel):
    """RFC 9457 problem detail. The single error shape for every 4xx/5xx."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    errors: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-field details, present on request validation failures.",
    )


class Page(BaseModel, Generic[T]):
    """A single page of results."""

    items: list[T]
    total: int = Field(ge=0, description="Total matching records, ignoring pagination.")
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
