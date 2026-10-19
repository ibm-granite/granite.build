# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Schemas for the HuggingFace dataset-import endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autotunex.models.dataset import DatasetName

HfRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
"""Immutable commit of the converted parquet branch, returned by the split picker."""

HfRepoId = Annotated[
    str,
    Field(
        min_length=3,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$",
    ),
]
"""A HuggingFace repo id (dataset or model) — ``owner/name``, or a bare canonical name.

Pattern-validated rather than only length-validated because ``hf_hub``
interpolates it straight into the Hub URL path (``/api/datasets/{repo_id}``) and
``httpx`` normalizes ``..`` segments per RFC 3986: an unconstrained value lets a
caller aim the request at any path under ``hf_hub_base_url``. Harmless against
``huggingface.co`` itself, but that setting is documented as overridable to a
mirror or a test double, which turns it into request forgery against an internal
host. Requiring every segment to start with an alphanumeric matches the Hub's own
naming rule and leaves no ``.`` or ``..`` segment expressible; the length ceiling
bounds the request path.
"""


class HfDatasetSplits(BaseModel):
    """The parquet branch's configs and splits, for the picker."""

    repo_id: str
    revision: HfRevision
    configs: dict[str, list[str]]
    """Config name -> split names."""


class HfImportPreview(BaseModel):
    """Raw and mapped samples, plus the survival count."""

    revision: HfRevision
    columns: list[str]
    raw_rows: list[dict[str, Any]]
    mapped_rows: list[dict[str, Any]]
    sampled: int
    survived: int


class HfImportRequest(BaseModel):
    """Commit an import. ``artifact_url`` remains server-only by design."""

    model_config = ConfigDict(extra="forbid")

    name: DatasetName
    description: str | None = None
    repo_id: HfRepoId
    revision: HfRevision
    config: str = Field(min_length=1)
    train_split: str = Field(min_length=1)
    validation_split: str | None = None
    validation_percentage: int | None = Field(default=None, ge=1, le=50)
    column_mapping: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_two_validation_sources(self) -> Self:
        """Refuse a validation split *and* a percentage: only one can take effect.

        ``dataset_runner._prepare_files`` tests its ``validation`` file first and
        returns, so a request carrying both would silently get the split and have
        the percentage dropped with no error. ``DatasetService.upload`` already
        rejects this exact combination, so refusing it here keeps the two ingest
        paths agreeing on one invariant instead of diverging.
        """
        if self.validation_split is not None and self.validation_percentage is not None:
            raise ValueError(
                "Provide either a validation split or a validation_percentage, not both."
            )
        return self


class HfPreviewRequest(BaseModel):
    """Preview a mapping before committing an import.

    A companion to :class:`HfImportRequest`, minus the dataset-creation fields
    (``name``, ``description``, ``validation_percentage``) a preview does not
    need.
    """

    model_config = ConfigDict(extra="forbid")

    repo_id: HfRepoId
    revision: HfRevision
    config: str = Field(min_length=1)
    train_split: str = Field(min_length=1)
    validation_split: str | None = None
    column_mapping: dict[str, str] = Field(min_length=1)
