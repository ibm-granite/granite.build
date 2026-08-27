# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Build task schemas.

A *task* is a build or deployment step attached to a job — a RITS deployment, a
tuning run, or an artifact download.

Field names mirror the ``autotunex_jobs`` view's aliases (``task_id``,
``task_status``, ``github_pr_url``, ``rits_url``) so existing consumers of that
view need no renaming.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from autotunex.models.status import GbTaskType, RunStatus


class GbTaskRead(BaseModel):
    """A build task as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    task_id: UUID
    build_id: UUID | None = None
    task_status: RunStatus
    task_type: GbTaskType
    github_pr_url: str | None = None
    artifact_id: UUID | None = None
    artifact_uri: str | None = None
    build_status: dict[str, Any] | None = None
    task_started_at: str | None = Field(
        default=None,
        description=(
            "Free-text start time. Typed as a string because the column is "
            "VARCHAR(255), not a timestamp — see docs/schema-review.md item A5."
        ),
    )
    task_updated_at: str | None = Field(
        default=None, description="Free-text update time; VARCHAR(255) in the schema."
    )
    rits_url: str | None = None
