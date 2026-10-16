# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Trial schemas.

A *trial* is one training run inside a job, evaluating a single concrete
parameter assignment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from autotunex.models.status import RunStatus


class TrialRead(BaseModel):
    """A trial as returned by the API.

    ``id`` is a short opaque string, not a UUID — ``trials.id`` is ``VARCHAR(16)``
    in the schema, assigned by the tuning pipeline.

    ``metric`` and ``metrics`` are sourced from the one-to-one ``results`` row
    rather than from ``trials`` itself, so a trial that has not reported yet
    carries an empty mapping instead of nulls.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: UUID
    status: RunStatus
    config: dict[str, Any] | None = Field(
        default=None, description="The concrete parameter assignment this trial tested."
    )
    metric: str | None = Field(
        default=None, description="Name of the objective metric, from results.metric."
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics the backend reported, e.g. {'eval_loss': 0.42}.",
    )
    created_at: datetime | None = Field(
        default=None,
        description=(
            "Absent when the tuning pipeline wrote the row without a timestamp: the "
            "live trials/results columns are DATETIME NOT NULL with no default, so a "
            "lax sql_mode stores MySQL's zero date, which reads back as null."
        ),
    )
    updated_at: datetime | None = Field(default=None)
