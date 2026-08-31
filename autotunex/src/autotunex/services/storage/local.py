# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Filesystem-backed dataset storage.

Files live at ``<root>/<dataset_id>/<name>_train.<ext>`` (and ``_validation``).
Blocking file IO is fine here: the runner already calls the backend inside its
own off-request coroutine and wraps heavy steps in ``asyncio.to_thread``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from autotunex.core.logging import get_logger
from autotunex.models.dataset import DatasetPreview
from autotunex.services.datasets_io import read_records

logger = get_logger(__name__)


class LocalStorageBackend:
    """Satisfies :class:`~autotunex.services.storage.base.StorageBackend`."""

    def __init__(self, root: Path, *, emit_file_uri: bool = False) -> None:
        """Store each dataset's files under ``<root>/<dataset_id>/``.

        Args:
            root: directory the dataset id-scoped folders live under.
            emit_file_uri: when ``True``, ``persist`` returns the dataset
                directory as an absolute ``file://`` ``artifact_url`` so a
                *same-host* consumer — the granite.build local-bash build in
                standalone mode — can mount it as its ``dataset_files`` input.
                Default ``False`` keeps the historical ``(None, None)`` return
                (files stay on disk with no external locator), which is correct
                for the OSS ``local`` backend and the in-process ``local`` job
                runner, both of which read the files by path directly. A remote
                consumer (LSF/SkyPilot) must not be handed a local path, so the
                registry leaves this ``False`` outside the bash case.
        """
        self._root = root
        self._emit_file_uri = emit_file_uri

    def _dir(self, dataset_id: UUID) -> Path:
        return self._root / str(dataset_id)

    def _path(self, dataset_id: UUID, name: str, data_format: str, *, split: str) -> Path:
        """Build the on-disk path for one split, refusing to escape the dataset dir.

        ``name`` is expected to already be validated at the API boundary
        (``DatasetCreate``'s path-separator check), but this is a second,
        independent layer: even if a traversal-bearing name reaches this method
        by some other route, the resolved path is asserted to stay within
        ``_dir(dataset_id)`` before it is ever used for a filesystem write or read.
        """
        candidate = self._dir(dataset_id) / f"{name}_{split}.{data_format}"
        base = self._dir(dataset_id).resolve()
        if not candidate.resolve().is_relative_to(base):
            raise ValueError("resolved dataset path escapes its dataset directory")
        return candidate

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
        """Move staged files into ``<root>/<id>/``; no artifact refs for local."""
        target_dir = self._dir(dataset_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(train), self._path(dataset_id, name, data_format, split="train"))
        if validation is not None:
            shutil.move(
                str(validation), self._path(dataset_id, name, data_format, split="validation")
            )
        # A same-host bash build reads the dataset from this directory; return it
        # as an absolute file:// locator when asked. `.resolve()` makes a relative
        # `dataset_storage_dir` absolute (as_uri() requires that); the directory
        # exists by now (created above). Otherwise no external locator exists.
        artifact_url = self._dir(dataset_id).resolve().as_uri() if self._emit_file_uri else None
        return None, artifact_url

    async def preview(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        artifact_url: str | None,
        rows: int,
    ) -> DatasetPreview:
        """Read bounded rows from the stored train/validation files."""
        train_path = self._path(dataset_id, name, data_format, split="train")
        validation_path = self._path(dataset_id, name, data_format, split="validation")
        train = read_records(train_path, data_format, limit=rows) if train_path.exists() else []
        validation = (
            read_records(validation_path, data_format, limit=rows)
            if validation_path.exists()
            else []
        )
        return DatasetPreview(train=train, validation=validation)

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        """Remove the dataset-id directory; a missing directory is not an error."""
        shutil.rmtree(self._dir(dataset_id), ignore_errors=True)
