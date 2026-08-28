# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Dataset business logic: CRUD plus the synchronous half of upload.

Ownership scoping is the same shape as
:class:`~autotunex.services.configurations.ConfigurationService` (one rule, one
place, so read and write cannot drift). Upload does only cheap synchronous work
here — validate, stream to staging, flip to ``uploading`` — then hands off to the
:class:`~autotunex.services.dataset_runner.DatasetUploadRunner` and returns
``202``. Deep validation (does it parse, is a split empty) happens in the runner
and surfaces as ``status='error'``. Knows nothing about HTTP; raises the domain
exceptions in :mod:`autotunex.core.exceptions`.
"""

from __future__ import annotations

import json
import shutil
from uuid import UUID

from starlette.datastructures import UploadFile

from autotunex.core.config import Settings
from autotunex.core.exceptions import (
    CallerNotProvisionedError,
    DatasetNotFoundError,
    DatasetNotReadyError,
    DomainValidationError,
    EmptyDatasetError,
    InvalidDatasetFormatError,
)
from autotunex.core.logging import get_logger
from autotunex.db.repositories.protocols import DatasetRepository
from autotunex.db.tables import DatasetTable, JobTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope, Page
from autotunex.models.dataset import DatasetCreate, DatasetJobRef, DatasetPreview, DatasetRead
from autotunex.models.status import DatasetStatus
from autotunex.services.dataset_runner import DatasetUploadRunner
from autotunex.services.datasets_io import ALLOWED_FORMATS, sniff_format, stream_to_staging
from autotunex.services.mappers import dataset_to_read
from autotunex.services.scoping import resolve_owner_filter, sees_nothing
from autotunex.services.storage.base import StorageBackend

logger = get_logger(__name__)


class DatasetService:
    """Full CRUD over datasets plus file upload, scoped to the calling principal."""

    def __init__(
        self,
        *,
        repository: DatasetRepository,
        principal: Principal,
        storage: StorageBackend,
        runner: DatasetUploadRunner,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._principal = principal
        self._storage = storage
        self._runner = runner
        self._settings = settings

    @staticmethod
    def _validate_data_format(data_format: str) -> None:
        """Reject a ``data_format`` outside ``{jsonl, csv, parquet}``.

        Raises:
            InvalidDatasetFormatError: unsupported format.
        """
        if data_format not in ALLOWED_FORMATS:
            raise InvalidDatasetFormatError(
                f"data_format must be one of {ALLOWED_FORMATS}, got {data_format!r}."
            )

    @staticmethod
    def _job_refs(jobs: list[JobTable]) -> list[DatasetJobRef]:
        """Convert referencing jobs to compact refs (already caller-scoped)."""
        return [
            DatasetJobRef(id=job.id, experiment_name=job.experiment_name, status=job.status)
            for job in jobs
        ]

    @staticmethod
    def _parse_column_mapping(raw: str | None) -> dict[str, str] | None:
        """Parse the optional ``column_mapping`` form field (a JSON string).

        Raises:
            DomainValidationError: not valid JSON, or not a string→string object.
        """
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DomainValidationError("column_mapping is not valid JSON.") from exc
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
        ):
            raise DomainValidationError(
                "column_mapping must be a JSON object mapping target to source names."
            )
        return parsed or None

    async def _associated(self, dataset_id: UUID, *, owner_id: UUID | None) -> list[DatasetJobRef]:
        """Return caller-scoped job refs for one dataset, using an already-resolved filter."""
        grouped = await self._repository.jobs_for_dataset([dataset_id], owner_id=owner_id)
        return self._job_refs(grouped.get(dataset_id, []))

    async def get(
        self,
        dataset_id: UUID,
        *,
        preview: bool = False,
        preview_rows: int = 10,
        scope: DataScope = DataScope.OWN,
    ) -> DatasetRead:
        """Return one dataset, scoped to the caller, optionally with a preview.

        Preview is attempted when ``preview`` is set and the dataset has data to
        read — either ``status='ready'`` (the local upload path) or a live
        ``artifact_url`` (a dataset registered out-of-band by the tuning pipeline
        via api-bridge, which never flips ``status`` to ``ready``). Without the
        ``artifact_url`` allowance such a dataset would report ``preview=None``
        despite plainly having data, and the UX would show "Unable to load
        dataset". Any backend failure still degrades to ``preview=None`` and is
        logged — a preview never fails the metadata read.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            DatasetNotFoundError: no such dataset, or it belongs to someone else.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise DatasetNotFoundError(dataset_id)
        dataset = await self._repository.get(dataset_id, owner_id=owner_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        preview_result = None
        if preview and (dataset.status == DatasetStatus.READY or dataset.artifact_url):
            preview_result = await self._safe_preview(dataset, preview_rows)
        return dataset_to_read(
            dataset, await self._associated(dataset_id, owner_id=owner_id), preview_result
        )

    async def _safe_preview(self, dataset: DatasetTable, rows: int) -> DatasetPreview | None:
        """Best-effort preview; a backend failure logs and yields ``None``."""
        try:
            return await self._storage.preview(
                dataset_id=dataset.id,
                name=dataset.name,
                data_format=dataset.data_format,
                artifact_url=dataset.artifact_url,
                rows=rows,
            )
        # Broad except is intentional: a preview must never fail the metadata
        # read. `BLE001` is not in this project's selected ruff rules, so it
        # needs no suppression comment.
        except Exception:
            logger.warning("Preview failed for dataset %s; returning null preview.", dataset.id)
            return None

    async def list(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        scope: DataScope = DataScope.OWN,
        q: str | None = None,
    ) -> Page[DatasetRead]:
        """Return one page of datasets, newest first — own rows by default.

        ``q`` is an optional case-insensitive substring filter on ``name``.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            return Page[DatasetRead](items=[], total=0, limit=limit, offset=offset)
        datasets, total = await self._repository.list(
            limit=limit, offset=offset, owner_id=owner_id, q=q
        )
        grouped = await self._repository.jobs_for_dataset(
            [dataset.id for dataset in datasets], owner_id=owner_id
        )
        items = [
            dataset_to_read(dataset, self._job_refs(grouped.get(dataset.id, [])))
            for dataset in datasets
        ]
        return Page[DatasetRead](items=items, total=total, limit=limit, offset=offset)

    async def create(self, data: DatasetCreate) -> DatasetRead:
        """Create an ``empty`` dataset owned by the calling principal.

        Raises:
            CallerNotProvisionedError: the caller has no ``user_id`` to own the row.
            InvalidDatasetFormatError: unsupported ``data_format``.
            DatasetNameConflictError: the caller already owns a dataset with this name.
        """
        owner_id = self._principal.user_id
        if owner_id is None:
            raise CallerNotProvisionedError()
        self._validate_data_format(data.data_format)
        dataset = await self._repository.create(
            user_id=str(owner_id),
            name=data.name,
            description=data.description,
            data_format=data.data_format,
        )
        return dataset_to_read(dataset, [])

    async def update(
        self, dataset_id: UUID, data: DatasetCreate, *, scope: DataScope = DataScope.OWN
    ) -> DatasetRead:
        """Fully replace a dataset's metadata, scoped to the caller.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            DatasetNotFoundError: no such dataset, or not the caller's.
            InvalidDatasetFormatError: unsupported ``data_format``.
            DatasetNameConflictError: the new name collides for the same owner.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise DatasetNotFoundError(dataset_id)
        self._validate_data_format(data.data_format)
        dataset = await self._repository.update(
            dataset_id,
            owner_id=owner_id,
            name=data.name,
            description=data.description,
            data_format=data.data_format,
        )
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        return dataset_to_read(dataset, await self._associated(dataset_id, owner_id=owner_id))

    async def delete(self, dataset_id: UUID, *, scope: DataScope = DataScope.OWN) -> None:
        """Delete a dataset and best-effort clean its stored files.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            DatasetNotFoundError: no such dataset, or not the caller's.
            DatasetInUseError: a job still references the dataset (raised by the repo).
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise DatasetNotFoundError(dataset_id)
        dataset = await self._repository.get(dataset_id, owner_id=owner_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        name, artifact_url = dataset.name, dataset.artifact_url
        deleted = await self._repository.delete(dataset_id, owner_id=owner_id)
        if not deleted:
            raise DatasetNotFoundError(dataset_id)
        # Broad except is intentional: storage cleanup is best-effort and the DB
        # delete has already committed. `BLE001` is not in this project's
        # selected ruff rules, so no noqa is needed to silence it.
        try:
            await self._storage.delete(dataset_id=dataset_id, name=name, artifact_url=artifact_url)
        except Exception:
            logger.warning("Storage cleanup failed for deleted dataset %s.", dataset_id)

    async def upload(
        self,
        dataset_id: UUID,
        *,
        train: UploadFile,
        validation: UploadFile | None,
        validation_percentage: int | None,
        column_mapping_json: str | None,
        gzip_encoded: bool,
    ) -> DatasetRead:
        """Validate cheaply, stream to staging, flip to ``uploading``, hand off.

        Returns ``DatasetRead`` with ``status='uploading'`` (the caller polls
        ``GET`` for the terminal state). Deep validation is deferred to the runner.

        Raises:
            DatasetNotFoundError: unknown dataset or not the caller's.
            DatasetNotReadyError: the dataset is already uploading.
            DomainValidationError: both a validation file and a percentage given,
                or an invalid ``column_mapping`` JSON, or the validation file's
                format differs from the train file's.
            UnsupportedDatasetFormatError: a file extension outside the allowed set.
            DatasetTooLargeError: a file exceeds the configured cap.
            EmptyDatasetError: a file has zero bytes.
        """
        # Upload targets an existing row by id and is a mutation, so it never
        # widens scope — an admin cannot upload into another owner's dataset.
        owner_id = self._principal.user_id
        if owner_id is None:
            raise DatasetNotFoundError(dataset_id)
        dataset = await self._repository.get(dataset_id, owner_id=owner_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        if dataset.status == DatasetStatus.UPLOADING:
            raise DatasetNotReadyError(dataset_id)
        if validation is not None and validation_percentage is not None:
            raise DomainValidationError(
                "Provide either a validation file or a validation_percentage, not both."
            )
        column_mapping = self._parse_column_mapping(column_mapping_json)

        data_format = sniff_format(train.filename or "")
        if validation is not None and sniff_format(validation.filename or "") != data_format:
            raise DomainValidationError("The validation file's format must match the train file's.")

        staging = self._settings.dataset_staging_dir / str(dataset_id)
        max_bytes = self._settings.dataset_upload_max_bytes
        train_path = staging / f"train.{data_format}"
        # Everything from here through the runner hand-off can leave a partial
        # or complete file under `staging` (stream_to_staging writes as it goes,
        # so DatasetTooLargeError fires with bytes already on disk). Any raise
        # in this block — before the runner has taken ownership — must not
        # orphan that directory: nothing else ever reclaims it, since the
        # dataset row never transitions to `uploading`. Once `submit` returns
        # normally, the runner owns `staging` and cleans it in its own
        # `finally`; do not also clean up on that success path, or the files
        # get deleted out from under it.
        try:
            if (
                await stream_to_staging(
                    train, train_path, max_bytes=max_bytes, gzip_encoded=gzip_encoded
                )
                == 0
            ):
                raise EmptyDatasetError("The uploaded train file is empty.")
            validation_path = None
            if validation is not None:
                validation_path = staging / f"validation.{data_format}"
                if (
                    await stream_to_staging(
                        validation, validation_path, max_bytes=max_bytes, gzip_encoded=gzip_encoded
                    )
                    == 0
                ):
                    raise EmptyDatasetError("The uploaded validation file is empty.")

            await self._repository.set_status(dataset_id, DatasetStatus.UPLOADING)
            await self._runner.submit(
                dataset_id,
                name=dataset.name,
                data_format=data_format,
                train=train_path,
                validation=validation_path,
                validation_percentage=validation_percentage,
                column_mapping=column_mapping,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return dataset_to_read(dataset, await self._associated(dataset_id, owner_id=owner_id))
