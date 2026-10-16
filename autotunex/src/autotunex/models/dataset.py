# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Dataset schemas.

A *dataset* is a named reference to training data. Unlike a job (read-only) this
resource has full CRUD plus a file upload, mirroring the Configuration
endpoints. Server-owned fields — ``status``, ``artifact_id``, ``artifact_url``,
the generated ``train_file``/``validation_file`` — are absent from
:class:`DatasetCreate`: a client names and describes a dataset and picks a
format, and everything else is set by the service, the upload runner, or the
database. In particular ``artifact_url`` being server-only closes the old repo's
caller-settable-``artifact_url`` hole.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autotunex.models.status import DatasetStatus, RunStatus


class DatasetCreate(BaseModel):
    """Request body for creating or fully replacing a dataset's metadata.

    Reused for ``POST`` and ``PUT`` (a full replacement), like
    :class:`~autotunex.models.configuration.ConfigurationCreate`. ``data_format``
    is validated against ``{jsonl, csv, parquet}`` in the service
    (:class:`~autotunex.core.exceptions.InvalidDatasetFormatError`, 422), not
    here, so the domain rule lives in one place. ``description`` is optional; the
    ``datasets.description`` column is nullable, so an omitted (or ``null``)
    description is stored as ``NULL``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None)
    data_format: str = Field(default="jsonl")

    @field_validator("name")
    @classmethod
    def _reject_path_separators(cls, value: str) -> str:
        """Names become filesystem path segments; forbid traversal characters."""
        if any(sep in value for sep in ("/", "\\")) or ".." in value:
            raise ValueError("name must not contain '/', '\\\\', or '..'")
        return value


class DatasetJobRef(BaseModel):
    """A compact reference to a job that uses this dataset.

    Kept small deliberately — a full :class:`~autotunex.models.job.JobRead` would
    nest a job's trials and tasks under every dataset. Scoped to the caller's own
    jobs in the service (an admin sees all referencing jobs).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_name: str | None = None
    status: RunStatus


class DatasetPreview(BaseModel):
    """A bounded peek at a ready dataset's rows, read via the active backend."""

    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    viewer_ready: bool = True
    """Whether the backend's viewer could serve a definitive answer.

    ``False`` only when the HuggingFace dataset viewer was unavailable or still
    precomputing the repo, so empty ``train``/``validation`` mean "not ready yet"
    rather than "genuinely empty" — the UI shows "preview will appear shortly"
    instead of a blank table. Defaults ``True`` so every other backend and every
    empty-preview literal keeps its existing meaning untouched.
    """


class DatasetRead(BaseModel):
    """A dataset as returned by every dataset endpoint.

    ``preview`` is populated only when ``?preview=true`` and ``status='ready'``;
    a backend failure while previewing degrades it to ``None`` and never fails
    the metadata read. ``artifact_id`` is surfaced as a string (the ORM stores it
    as a ``Uuid36``) so the contract does not assume the identifier's shape.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str = Field(description="Owner's id, from datasets.user_id.")
    name: str
    description: str | None = None
    data_format: str
    status: DatasetStatus
    status_detail: str | None = None
    train_file: str
    train_records: int | None = None
    train_file_size: int | None = None
    validation_file: str
    validation_records: int | None = None
    validation_file_size: int | None = None
    artifact_id: str | None = None
    artifact_url: str | None = None
    associated_jobs: list[DatasetJobRef] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    preview: DatasetPreview | None = None
