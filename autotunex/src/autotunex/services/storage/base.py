# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The ``StorageBackend`` seam — the ``ArtifactStore`` abstraction (open decision 4).

Backends move persisted dataset files somewhere durable and can read a bounded
preview back. They never touch FastAPI or the database — they work in file
paths and return artifact references (``artifact_id``, ``artifact_url``) for the
runner to record. All methods are keyword-only for call-site clarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import UUID

from autotunex.models.dataset import DatasetPreview


class StorageBackend(Protocol):
    """Where a dataset's files live after upload processing."""

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
        """Store the processed files; return ``(artifact_id, artifact_url)``.

        Local storage returns ``(None, None)`` (files stay on disk); HuggingFace
        returns the pushed repo's identifier and URL.
        """
        ...

    async def preview(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        artifact_url: str | None,
        rows: int,
    ) -> DatasetPreview:
        """Return at most ``rows`` rows per split; best-effort (may be empty)."""
        ...

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        """Remove the dataset's stored files; idempotent, best-effort."""
        ...
