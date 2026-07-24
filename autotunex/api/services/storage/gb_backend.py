# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Granite Build backend: pushes the finalized dataset as a GB fileset artifact
and records the returned (uuid, uri) as the locator. This is the is_gb_enabled()
branch of the old dataset_service._finalize_upload, lifted verbatim.

Preview reads the retained local copy (the GB push does not remove local files),
so viewing behavior is identical to the local backend today. file_service's
HTTPExceptions are translated into the storage contract (StorageNotFound /
StorageError) so FastAPI types never leak across the abstraction boundary.
"""

import logging

from fastapi import HTTPException

from services import gb_service, file_service
from utils import extract_uuid_uri
from .base import (
    StorageBackend,
    DatasetRef,
    DatasetFiles,
    StorageLocator,
    StorageError,
)
from ._utils import translate_http_exc

logger = logging.getLogger(__name__)


class GBStorageBackend(StorageBackend):
    def __init__(self):
        # GBService is a singleton; constructing here (not at module import) keeps
        # the heavy gb_service/gbcli chain out of non-GB deployments — the factory
        # only constructs GBStorageBackend when GB is enabled.
        self.gb = gb_service.GBService()

    async def persist(self, files: DatasetFiles) -> StorageLocator:
        try:
            artifact_name = f"{files.name}_{str(files.dataset_id)[:8]}"
            command = [
                "artifact",
                "push",
                "--from-local",
                files.local_dir,
                "--artifact-name",
                artifact_name,
                "--type",
                "dataset", "--store", "hf", "--tags", "autotunex",
                "--certify-no-restrictions",
            ]
            result = await self.gb.command_executor(command)
            logger.debug("GB artifact push result: %s", result)
            uuid, uri = extract_uuid_uri(result)
            return StorageLocator(
                artifact_id=uuid.strip() if uuid is not None else None,
                artifact_url=uri.strip() if uri is not None else None,
            )
        except (
            Exception
        ) as e:  # surface as a storage failure the service maps to HTTP 400
            logger.error("GBStorageBackend.persist failed: %s", e)
            raise StorageError(str(e)) from e

    async def preview(self, ref: DatasetRef, file: str, limit: int) -> list[dict]:
        rel = f"{ref.dataset_id}/{ref.name}/{file}"
        try:
            return await file_service.get_dataset_data(
                rel, data_format=ref.data_format, limit=limit
            )
        except HTTPException as exc:
            raise translate_http_exc(exc, rel) from exc

    async def delete(self, ref: DatasetRef) -> None:
        # Matches today's behavior: removing the dataset deletes the local folder.
        # (GB artifact lifecycle is unchanged by this refactor.) Idempotent.
        try:
            await file_service.delete_dataset_folder(dataset_name=ref.dataset_id)
        except HTTPException as exc:
            raise translate_http_exc(exc, ref.dataset_id) from exc
