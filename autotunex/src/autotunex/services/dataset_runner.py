# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The dataset upload runner seam and its in-process implementation.

Unlike ``JobRunner`` (training needs a worker/GPU, so ``NoOpJobRunner`` is a
stub), the upload's heavy work — parse, count, split, column-remap, and the
storage ``persist`` — can legitimately run in the API process, off the request
path. ``submit`` schedules ``process`` and returns immediately (the request has
already sent its ``202``); ``process`` opens its OWN database session, because
the request-scoped session is closed by the time it runs. A queue-backed runner
can replace this later by swapping the provider in ``api/deps.py``.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.exceptions import (
    DatasetProcessingTimeoutError,
    DomainValidationError,
    InsufficientStorageError,
    UploadProcessingError,
)
from autotunex.core.logging import get_logger
from autotunex.db.repositories.sqlalchemy import SqlAlchemyDatasetRepository
from autotunex.models.status import DatasetStatus
from autotunex.services.datasets_io import (
    count_records,
    normalize_json_array_to_jsonl,
    remap_records,
    split_by_percentage,
)
from autotunex.services.storage.base import StorageBackend

logger = get_logger(__name__)


class DatasetUploadRunner(Protocol):
    """Hands off the async half of an upload after the ``202`` is returned."""

    async def submit(
        self,
        dataset_id: UUID,
        *,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
        validation_percentage: int | None,
        column_mapping: dict[str, str] | None,
    ) -> None:
        """Schedule processing of the staged files; return without awaiting it."""
        ...


def _safe_detail(exc: Exception) -> str:
    """Return a client-safe ``status_detail`` — never raw exception text.

    Domain-validation failures (e.g. the empty-split guard) and authored
    ``UploadProcessingError``s (e.g. a processing timeout) carry a message this
    codebase wrote, so they are safe to surface verbatim. Anything else — a
    subprocess failure, a parse error — collapses to a fixed generic string.
    """
    if isinstance(exc, (DomainValidationError, UploadProcessingError)):
        return exc.detail
    return "Upload processing failed; check the file's format and contents, then re-upload."


class InProcessDatasetUploadRunner:
    """Runs the upload's heavy work in-process. Satisfies :class:`DatasetUploadRunner`."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        storage: StorageBackend,
        staging_dir: Path,
        max_concurrent: int = 2,
        processing_timeout_seconds: float = 3600.0,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._staging_dir = staging_dir
        self._processing_timeout_seconds = processing_timeout_seconds
        # One shared slot-set for THIS runner instance; production shares one
        # runner process-wide (see api/deps.get_dataset_runner), so this bounds
        # concurrent processing across all uploads in the API process.
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(
        self,
        dataset_id: UUID,
        *,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
        validation_percentage: int | None,
        column_mapping: dict[str, str] | None,
    ) -> None:
        """Fire ``process`` as a background task and return immediately.

        A strong reference is kept in ``self._tasks`` until completion so the
        task is not garbage-collected mid-flight (the coroutine also references
        ``self``, keeping this runner alive).
        """
        task = asyncio.create_task(
            self.process(
                dataset_id,
                name=name,
                data_format=data_format,
                train=train,
                validation=validation,
                validation_percentage=validation_percentage,
                column_mapping=column_mapping,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def process(
        self,
        dataset_id: UUID,
        *,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
        validation_percentage: int | None,
        column_mapping: dict[str, str] | None,
    ) -> None:
        """Guard, time, and run the upload's heavy work; record the terminal state.

        Holds one concurrency slot for the whole of processing (including the
        storage ``persist``), so simultaneous large uploads queue rather than
        pile up in memory. The work is bounded by ``processing_timeout_seconds``
        via ``asyncio.wait_for``; note this cancels the awaiting coroutine but
        cannot kill a blocking ``llmb`` subprocess mid-push — that is bounded by
        the HF backend's own ``subprocess`` timeout. On any failure the dataset is
        marked ``error`` with a safe detail and the row is left intact for a
        re-upload; staging is always cleaned up.
        """
        async with self._semaphore, self._session_factory() as session:
            repository = SqlAlchemyDatasetRepository(session)
            try:
                try:
                    await asyncio.wait_for(
                        self._run_processing(
                            repository,
                            dataset_id,
                            name=name,
                            data_format=data_format,
                            train=train,
                            validation=validation,
                            validation_percentage=validation_percentage,
                            column_mapping=column_mapping,
                        ),
                        self._processing_timeout_seconds,
                    )
                except TimeoutError as exc:
                    raise DatasetProcessingTimeoutError(self._processing_timeout_seconds) from exc
            # Broad except is intentional: any failure (parse, split, remap,
            # storage, timeout) must still land the dataset in `error`.
            except Exception as exc:
                logger.exception("Dataset upload processing failed for %s", dataset_id)
                # A timeout/cancellation can leave the session mid-transaction;
                # roll back before recording the error so set_status can't fail
                # on a pending rollback and leave the dataset stuck in `processing`.
                await session.rollback()
                await repository.set_status(
                    dataset_id, DatasetStatus.ERROR, status_detail=_safe_detail(exc)
                )
            finally:
                shutil.rmtree(self._staging_dir / str(dataset_id), ignore_errors=True)

    async def _run_processing(
        self,
        repository: SqlAlchemyDatasetRepository,
        dataset_id: UUID,
        *,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
        validation_percentage: int | None,
        column_mapping: dict[str, str] | None,
    ) -> None:
        """The success path: parse/split/remap, persist, and record the result.

        Raises on any failure; ``process`` translates that into ``status='error'``.
        """
        # A `.json` upload is mapped to the `jsonl` format but its raw bytes are
        # streamed to disk, so a standard (often pretty-printed) JSON array must
        # be normalized to JSONL before counting/splitting/remapping — all of
        # which read line-by-line. A file that is already JSONL is left untouched.
        if data_format == "jsonl":
            await asyncio.to_thread(normalize_json_array_to_jsonl, train)
            if validation is not None:
                await asyncio.to_thread(normalize_json_array_to_jsonl, validation)
        (
            train_path,
            validation_path,
            train_records,
            validation_records,
        ) = await self._prepare_files(
            dataset_id,
            data_format=data_format,
            train=train,
            validation=validation,
            validation_percentage=validation_percentage,
        )
        if column_mapping:
            await asyncio.to_thread(
                remap_records, train_path, column_mapping, data_format=data_format
            )
            if validation_path is not None:
                await asyncio.to_thread(
                    remap_records, validation_path, column_mapping, data_format=data_format
                )
        train_size = train_path.stat().st_size
        validation_size = validation_path.stat().st_size if validation_path is not None else None
        self._check_free_space(train_size, validation_size)
        artifact_id, artifact_url = await self._storage.persist(
            dataset_id=dataset_id,
            name=name,
            data_format=data_format,
            train=train_path,
            validation=validation_path,
        )
        await repository.set_upload_result(
            dataset_id,
            train_records=train_records,
            train_file_size=train_size,
            validation_records=validation_records,
            validation_file_size=validation_size,
            data_format=data_format,
            artifact_id=artifact_id,
            artifact_url=artifact_url,
        )

    def _check_free_space(self, train_size: int, validation_size: int | None) -> None:
        """Refuse to persist when free disk is below the finalized files' size.

        Best-effort guard: ``persist`` copies the finalized files once more (a
        cross-filesystem local move copies too), so a volume with less free space
        than the payload would fail mid-copy with a cryptic ``OSError``. Checking
        first yields a clear ``error`` detail.

        This checks the staging root's filesystem — the real copy *origin* for
        every backend, and, for ``LocalStorageBackend``, the real copy *target*
        too. It is only an honest signal for that backend. The HuggingFace
        backend's ``_run_push`` instead copies into ``tempfile.mkdtemp(...)`` —
        the OS temp dir, which may be a different, smaller volume than the
        staging root. So this guard can pass while the temp volume is full; the
        mid-copy ``OSError`` in that case still lands the dataset in ``error``
        via the broad ``except`` around the push, just without this guard's
        earlier, clearer detail.
        """
        required = train_size + (validation_size or 0)
        free = shutil.disk_usage(self._staging_dir).free
        if free < required:
            logger.error(
                "Insufficient disk to finalize upload under %s: need %d bytes, %d free.",
                self._staging_dir,
                required,
                free,
            )
            raise InsufficientStorageError

    async def _prepare_files(
        self,
        dataset_id: UUID,
        *,
        data_format: str,
        train: Path,
        validation: Path | None,
        validation_percentage: int | None,
    ) -> tuple[Path, Path | None, int, int | None]:
        """Resolve the final train/validation files and their record counts.

        Three cases: an explicit validation file (count both), a
        ``validation_percentage`` (seeded split of ``train``), or neither (train
        only). The split seed is derived from the dataset id so a re-run is
        identical.
        """
        if validation is not None:
            train_records = await asyncio.to_thread(count_records, train, data_format)
            validation_records = await asyncio.to_thread(count_records, validation, data_format)
            return train, validation, train_records, validation_records
        if validation_percentage is not None:
            seed = dataset_id.int & 0xFFFFFFFF
            train_split = train.with_name(f"train_split.{data_format}")
            validation_split = train.with_name(f"validation_split.{data_format}")
            train_records, validation_records = await asyncio.to_thread(
                split_by_percentage,
                train,
                train_split,
                validation_split,
                data_format=data_format,
                validation_percentage=validation_percentage,
                seed=seed,
            )
            return train_split, validation_split, train_records, validation_records
        train_records = await asyncio.to_thread(count_records, train, data_format)
        return train, None, train_records, None


class NoOpDatasetUploadRunner:
    """A runner that records submissions and does nothing. For tests.

    Satisfies :class:`DatasetUploadRunner`. Lets a service test assert the
    hand-off happened without spawning real background work.
    """

    def __init__(self) -> None:
        self.submitted: list[UUID] = []

    async def submit(
        self,
        dataset_id: UUID,
        *,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
        validation_percentage: int | None,
        column_mapping: dict[str, str] | None,
    ) -> None:
        """Record the dataset id; do no work."""
        self.submitted.append(dataset_id)
