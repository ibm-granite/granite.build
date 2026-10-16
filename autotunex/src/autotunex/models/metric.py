# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""API models for the per-step training-metrics endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class MetricPointRead(BaseModel):
    """One ``training_metrics`` row as returned by the metrics endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    trial_id: str | None = None
    global_step: int
    epoch: float | None = None
    loss: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None
    split: str
    extra: dict[str, Any] | None = None
    created_at: datetime | None = None


class MetricPage(BaseModel):
    """One ascending keyset page of metric points (oldest first)."""

    metrics: list[MetricPointRead]
    has_more: bool
    next_after_id: int | None = None
