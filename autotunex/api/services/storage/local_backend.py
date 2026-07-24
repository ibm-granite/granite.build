# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Local-disk backend: the default when Granite Build is not enabled.

Files are already streamed to UPLOAD_DIR/<id>/<name>/ by the ingest pipeline,
so persist only returns an (empty) locator — matching today's behavior where a
non-GB dataset has artifact_id/url = None. Preview reads a bounded number of
rows; delete removes the dataset folder.

The underlying file_service helpers raise FastAPI HTTPException; this backend
translates those into the storage contract's StorageNotFound / StorageError so
callers never see FastAPI types leak across the abstraction boundary.
"""

import logging

from fastapi import HTTPException
from services import file_service

from ._utils import translate_http_exc
from .base import (
    DatasetFiles,
    DatasetRef,
    StorageBackend,
    StorageLocator,
)

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    async def persist(self, files: DatasetFiles) -> StorageLocator:
        # Files already on disk under UPLOAD_DIR; nothing to push. Mirrors the
        # is_gb_enabled()==False branch of the old _finalize_upload.
        logger.debug("LocalStorageBackend.persist: %s/%s", files.dataset_id, files.name)
        return StorageLocator(artifact_id=None, artifact_url=None)

    async def preview(self, ref: DatasetRef, file: str, limit: int) -> list[dict]:
        # file is "<name>.<ext>"; the relative path under UPLOAD_DIR is <id>/<name>/<file>.
        rel = f"{ref.dataset_id}/{ref.name}/{file}"
        try:
            return await file_service.get_dataset_data(
                rel, data_format=ref.data_format, limit=limit
            )
        except HTTPException as exc:
            raise translate_http_exc(exc, rel) from exc

    async def delete(self, ref: DatasetRef) -> None:
        # delete_dataset_folder takes the top-level folder name, which is the dataset id
        # (matches today's Dataset.delete_dataset call: delete_dataset_folder(dataset["id"])).
        # It is idempotent: a missing folder is a no-op, not an error.
        try:
            await file_service.delete_dataset_folder(dataset_name=ref.dataset_id)
        except HTTPException as exc:
            raise translate_http_exc(exc, ref.dataset_id) from exc
