# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Schemas for resource estimation (start-tuning wizard, Step 3)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator


class EstimateUsagesRequest(BaseModel):
    """Estimate for a saved config (``config_id``) OR an unsaved one (``config_data``).

    Exactly one of ``config_id`` / ``config_data`` must be set. The inline
    ``config_data`` path is what makes mid-wizard estimation possible before a
    configuration is persisted. ``protected_namespaces=()`` because ``model_name``
    collides with Pydantic's ``model_`` prefix.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_name: str
    gpu_memory: int = 80
    config_id: UUID | None = None
    config_data: dict[str, Any] | None = None
    tuner_type: str | None = None
    rl_tuner_type: str | None = None

    @model_validator(mode="after")
    def _exactly_one_config_source(self) -> EstimateUsagesRequest:
        if (self.config_id is None) == (self.config_data is None):
            raise ValueError("exactly one of config_id or config_data must be provided")
        if self.gpu_memory < 1:
            raise ValueError("gpu_memory must be >= 1")
        return self


class EstimateUsagesResponse(BaseModel):
    """The eight-field estimate the wizard renders."""

    model_config = ConfigDict(protected_namespaces=())

    model_size_billion_params: float
    gpu_memory_gb: float
    cpu_memory_gb: float
    num_gpus: int
    weights_memory: float
    optimizer_memory: float
    gradients_memory: float
    activations_memory: float
