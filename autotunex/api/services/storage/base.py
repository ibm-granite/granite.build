# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Storage backend contract for datasets.

A StorageBackend is the ONLY place that knows where dataset bytes live. The
dataset service streams an upload to a local staging dir, then hands the
finalized files to a backend's `persist`. Viewing calls `preview`; deletion
calls `delete`. Backends never touch FastAPI types or the database.

To add a backend (e.g. object storage): implement StorageBackend, then
register it in services/storage/__init__.get_storage_backend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DatasetRef:
    """Identity of a dataset, enough to locate its files in any backend."""

    dataset_id: str
    name: str
    data_format: Literal["jsonl", "parquet"]


@dataclass
class DatasetFiles(DatasetRef):
    """A dataset whose finalized train/val files are on local disk, ready to persist."""

    local_dir: str  # absolute staging dir holding the two files
    train_file: str  # filename only (no path)
    validation_file: str


@dataclass
class StorageLocator:
    """Where a persisted dataset lives. Maps onto datasets.artifact_id / artifact_url.
    Local backend returns (None, None); GB/object backends return their identifiers."""

    artifact_id: Optional[str] = None
    artifact_url: Optional[str] = None


class StorageError(Exception):
    """Base for storage failures. Mapped to HTTP 400 by the service unless more specific."""


class StorageNotFound(StorageError):
    """A requested dataset/file is absent in the backend. Mapped to HTTP 404."""


class StorageValidationError(StorageError):
    """Input to the backend was invalid. Mapped to HTTP 400."""


class StorageBackend(ABC):
    """A place datasets live (local disk, Granite Build, object storage, ...).
    Owns persist + preview + delete only."""

    @abstractmethod
    async def persist(self, files: DatasetFiles) -> StorageLocator:
        """Store the finalized train/val files; return their locator."""

    @abstractmethod
    async def preview(self, ref: DatasetRef, file: str, limit: int) -> list[dict]:
        """Return at most `limit` rows for the view page. MUST NOT load the whole file.

        file: filename-only basename (e.g. "ds_train.jsonl"), no path component;
              the backend resolves the full path from ref.dataset_id and ref.name.
        """

    @abstractmethod
    async def delete(self, ref: DatasetRef) -> None:
        """Remove all stored objects for this dataset (idempotent)."""
