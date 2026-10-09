# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace dataset import: search, splits, preview and the import commit.

Composes the Hub client, HF's dataset viewer and the import planner with bounded
parquet staging. Knows
nothing about HTTP routing — raises the domain exceptions in
:mod:`autotunex.core.exceptions`, translated to responses by the router.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from autotunex.core.config import Settings
from autotunex.core.exceptions import (
    CallerNotProvisionedError,
    DomainValidationError,
    HfDatasetNotConvertedError,
    HfDatasetNotTabularError,
    HfHubUnreachableError,
    HfPreviewUnavailableError,
)
from autotunex.core.logging import get_logger
from autotunex.db.repositories.protocols import DatasetRepository
from autotunex.db.repositories.sqlalchemy import SqlAlchemyDatasetRepository
from autotunex.models.auth import Principal
from autotunex.models.dataset import DatasetRead
from autotunex.models.hf_import import (
    HfDatasetSplits,
    HfImportPreview,
    HfImportRequest,
    HfPreviewRequest,
)
from autotunex.models.status import DatasetStatus
from autotunex.services import hf_import
from autotunex.services.dataset_runner import DatasetUploadRunner
from autotunex.services.datasets_io import PARQUET_BATCH_SIZE
from autotunex.services.mappers import dataset_to_read
from autotunex.services.storage import hf_hub, hf_viewer

logger = get_logger(__name__)

_PREVIEW_SAMPLE_ROWS = 100

_DOWNLOAD_TIMEOUT = httpx.Timeout(10.0, read=60.0)
"""The read budget for a bulk shard download, stated rather than inherited.

``get_hf_http_client`` hands staging the shared ``app.state.http_client``, whose
``httpx.Timeout(10.0)`` ``main.py`` picked for the OIDC token-endpoint call. A
multi-GiB shard from HF's CDN can stall longer than that without being dead, and a
``ReadTimeout`` here discards the whole partial download with no retry -- so this
path overrides the read leg only, leaving connect/write/pool at the shared value.
"""

_MAX_STATUS_DETAIL = 1_000
"""Cap on a staging failure's ``status_detail``, which is bounded storage.

``datasets.status_detail`` is ``Text`` (64 KiB on MySQL) and the write happens
*inside* the staging error handler, so an oversized detail makes that write raise
and leaves the row in ``importing`` -- a state the upload path refuses and only a
delete can clear. Capping at the one place every staging detail passes through
bounds the column whatever a future message does.
"""
"""Matches the viewer's own page cap (``_VIEWER_ROW_CAP``); asking for more buys nothing."""

_PARTITIONED_SPLIT = re.compile(r"-part\d+$")
"""HF shards a split past 10 000 files into sibling ``<split>-part<n>`` directories."""


def _viewer_split(split: str) -> str:
    """Map a parquet-branch split directory onto the name the viewer knows.

    ``list_parquet_files`` derives split names from directories on the converted
    branch, where a split too large for one directory appears as ``train-part0``,
    ``train-part1``, ... The viewer only ever calls that split ``train``, so a
    picker name would 404 there for exactly the biggest datasets. Translated here,
    at the one call site that talks to the viewer, rather than in the picker --
    grouping the parts upstream would also change which shards an *import* reads,
    which is a separate decision (see the design spec).
    """
    return _PARTITIONED_SPLIT.sub("", split)


def _join_capped(names: list[str], *, limit: int = 20) -> str:
    """Join column names for a refusal message, naming at most ``limit`` of them.

    A caller controls both lists this formats: ``column_mapping``'s values (capped
    by neither entry count nor length) and, by choosing the repo, the staged
    schema. Naming every one of several thousand produces a message no reader can
    use and, worse, one too large for the column it lands in -- see
    :data:`_MAX_STATUS_DETAIL`. The first few names are what makes a refusal
    actionable; the count carries the rest.
    """
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} (and {len(names) - limit} more)"


def _safe_staging_detail(exc: Exception) -> str:
    """Return a client-safe ``status_detail`` for a staging-phase failure.

    Mirrors ``dataset_runner._safe_detail``'s collapse-to-generic behavior
    without importing that module-private helper across modules. A
    ``DomainValidationError``'s message is authored and safe to surface
    verbatim; anything else (a network error, a malformed parquet file)
    collapses to a fixed generic string.
    """
    if isinstance(exc, DomainValidationError):
        detail = exc.detail
        if len(detail) > _MAX_STATUS_DETAIL:
            return f"{detail[:_MAX_STATUS_DETAIL]}..."
        return detail
    return "HuggingFace import failed while fetching the dataset; please try again."


def _copy_capped_rows(
    tmp_path: Path,
    *,
    destination: Path,
    writer: pq.ParquetWriter | None,
    max_rows: int,
    retained_rows: int,
) -> tuple[pq.ParquetWriter | None, int, int]:
    """Copy ``tmp_path``'s rows into ``destination``, stopping at ``max_rows``.

    Synchronous by design, and called via ``asyncio.to_thread``: every pyarrow
    call here (``ParquetFile``, ``iter_batches``, ``write_batch``) is a blocking,
    CPU-bound decode/re-encode, and running a whole shard's transcode inline in a
    coroutine stalls the event loop — every other request, ``/health/ready``
    included — for its duration. ``dataset_runner._run_processing`` gives
    ``count_records``/``remap_records`` the same treatment.

    Takes and returns the shared ``writer`` because a single ``ParquetWriter``
    spans every shard of one split; it is created lazily on the first batch since
    its schema comes from the data. Returns
    ``(writer, shard_original_rows, retained_rows)``.
    """
    with pq.ParquetFile(tmp_path) as source:
        original_rows = source.metadata.num_rows
        for batch in source.iter_batches(batch_size=PARQUET_BATCH_SIZE):
            remaining = max_rows - retained_rows
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
            if writer is None:
                writer = pq.ParquetWriter(destination, batch.schema)
            writer.write_batch(batch)
            retained_rows += batch.num_rows
    return writer, original_rows, retained_rows


_TranscodeResult = tuple[pq.ParquetWriter | None, int, int]

_pending_writer_closes: set[asyncio.Task[_TranscodeResult]] = set()
"""Transcodes outliving a cancelled ``_stage_split``, rooted so nothing collects them.

Same reason as ``_background_tasks`` below: asyncio holds only a weak reference
to a task, and a pending one whose last strong reference was a dead frame is
collectible mid-run.
"""


def _close_transcoded_writer(task: asyncio.Task[_TranscodeResult]) -> None:
    """Close the ``ParquetWriter`` a cancelled ``_stage_split`` left mid-transcode.

    Runs as the transcode task's done-callback, so by the time it fires the worker
    thread has finished and nothing else holds the writer. A cancelled or failed
    transcode has nothing to close: either the writer was never created or the
    exception already unwound past it.
    """
    _pending_writer_closes.discard(task)
    if task.cancelled() or task.exception() is not None:
        return
    writer, _, _ = task.result()
    if writer is not None:
        writer.close()


async def _stage_split(
    client: httpx.AsyncClient,
    *,
    urls: list[str],
    destination: Path,
    max_rows: int,
    max_bytes: int,
    spent_bytes: int = 0,
    token: str | None,
) -> tuple[int, int, int, int]:
    """Fetch ``urls`` in order into ``destination``, truncated to ``max_rows`` rows.

    Each URL is streamed to a temp file next to ``destination`` first (a
    ``pq.ParquetFile`` needs a seekable source, and an ``httpx`` stream is not
    one), then read back with ``iter_batches`` and copied into a single
    ``pq.ParquetWriter`` — never buffering a whole file in memory. Stops as
    soon as ``max_rows`` is reached, so a shard beyond the cap is never
    downloaded.

    ``max_bytes`` caps the *downloaded* bytes, checked chunk by chunk as they are
    written, exactly as ``datasets_io.stream_to_staging`` caps an upload.
    ``hf_import``'s plan-time gate cannot be relied on for this: a CDN 200 that
    omits ``content-length`` contributes 0 to ``head_total_bytes``, so the gate
    passes any size, and ``max_rows`` truncates only *after* each shard has landed
    on disk in full. Without this cap one oversized shard fills the staging volume.

    ``spent_bytes`` is what an earlier split of the same import already drew on
    that budget, so the cap applies to the import as a whole while the message
    still names the real configured ceiling rather than a leftover remainder.

    Returns ``(original_rows, retained_rows, downloaded_bytes, unread_shards)``:
    ``original_rows`` sums ``.metadata.num_rows`` for every shard actually opened
    (one never downloaded because the cap was already reached is not counted — an
    honest number given the download-first approach, not a claim about the whole
    split); ``retained_rows`` is what was written to ``destination``.
    ``unread_shards`` distinguishes a complete split from a cap reached exactly
    at a shard boundary, where the two row counts alone would be equal.

    Raises:
        DomainValidationError: the download exceeded ``max_bytes``.
    """
    headers = hf_hub._headers(token)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    transcode: asyncio.Task[_TranscodeResult] | None = None
    original_rows = 0
    retained_rows = 0
    downloaded_bytes = 0
    read_shards = 0
    try:
        for index, url in enumerate(urls):
            if retained_rows >= max_rows:
                break
            tmp_path = destination.parent / f".{destination.name}.{index}.tmp"
            try:
                async with client.stream(
                    "GET",
                    url,
                    follow_redirects=True,
                    headers=headers,
                    timeout=_DOWNLOAD_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as tmp_file:
                        async for chunk in response.aiter_bytes():
                            downloaded_bytes += len(chunk)
                            if spent_bytes + downloaded_bytes > max_bytes:
                                raise DomainValidationError(
                                    "The selected HuggingFace split exceeds the "
                                    f"{max_bytes}-byte import limit."
                                )
                            tmp_file.write(chunk)
                transcode = asyncio.create_task(
                    asyncio.to_thread(
                        _copy_capped_rows,
                        tmp_path,
                        destination=destination,
                        writer=writer,
                        max_rows=max_rows,
                        retained_rows=retained_rows,
                    )
                )
                # Shielded so a cancellation of *this* coroutine cannot mark the
                # transcode cancelled while its thread is still running: the
                # `finally` below needs a task that will actually deliver the
                # writer it created.
                writer, shard_rows, retained_rows = await asyncio.shield(transcode)
                original_rows += shard_rows
                read_shards += 1
            finally:
                tmp_path.unlink(missing_ok=True)
    finally:
        if transcode is not None and not transcode.done():
            # A worker thread is still inside `write_batch` on this very writer --
            # `asyncio.to_thread` cannot be cancelled mid-run, so reaching the
            # `close()` below would drive one non-thread-safe ParquetWriter from
            # two threads at once (a pyarrow error, or a native crash). Only the
            # cancellation paths get here: the caller's `wait_for` timeout, or
            # `main.py`'s shutdown drain. Hand the close to the transcode's own
            # completion, which by definition runs after the thread is done. If
            # the loop closes first the footer goes unwritten, but staging is
            # discarded on both of those paths anyway.
            _pending_writer_closes.add(transcode)
            transcode.add_done_callback(_close_transcoded_writer)
        elif writer is not None:
            # Deliberately not off-thread: closing only flushes the footer, and an
            # await in this `finally` could itself be interrupted when the caller's
            # `asyncio.wait_for` cancels the staging task, leaving the file open.
            writer.close()
    return original_rows, retained_rows, downloaded_bytes, len(urls) - read_shards


@dataclass(frozen=True)
class StagedParquet:
    """Paths and row counts from staging a plan's train/validation split."""

    train_path: Path
    train_original_rows: int
    train_retained_rows: int
    validation_path: Path | None
    validation_original_rows: int | None
    validation_retained_rows: int | None
    train_unread_shards: int
    validation_unread_shards: int | None

    @property
    def train_truncated(self) -> bool:
        """Whether any train rows or shards were omitted by the cap."""
        return self.train_original_rows > self.train_retained_rows or self.train_unread_shards > 0

    @property
    def validation_truncated(self) -> bool | None:
        """Whether validation was capped, or ``None`` when no split was selected."""
        if self.validation_original_rows is None or self.validation_retained_rows is None:
            return None
        return self.validation_original_rows > self.validation_retained_rows or bool(
            self.validation_unread_shards
        )


def _reject_unmapped_sources(path: Path, mapping: dict[str, str]) -> None:
    """Refuse a ``column_mapping`` whose sources are absent from ``path``'s schema.

    Synchronous (a cheap footer read) and called via ``asyncio.to_thread``.

    ``datasets_io.remap_records`` is deliberately *tolerant*: it silently drops a
    target whose source is missing, which the upload path wants for ragged data.
    On a parquet file that tolerance is silent data loss instead — the schema is
    fixed, so a source absent from it is absent from every row, and if **no**
    source matches, ``chunk.select([])`` writes a file that round-trips to zero
    rows and zero columns while ``count_records`` reports 0 and the dataset still
    flips to ``ready``. The import path is where that is reachable without the
    user ever having seen the real column names: the preview is optional.
    So this validates against the file that was actually staged, rather
    than loosening ``remap_records`` for both paths.

    A *blank* source is the same zero-column hazard by another route, and
    ``HfImportRequest.column_mapping`` constrains the dict rather than its values,
    so ``{"input": "", "output": ""}`` reaches here having passed validation. It
    cannot be reported as "not in the dataset" (it names nothing), so it is
    refused separately. The two checks together leave exactly the invariant this
    guard exists for: at least one source is non-blank *and* present, so the
    remap cannot reduce to ``select([])``. A partly-blank mapping still passes --
    that is the raggedness ``apply_mapping`` documents tolerating, and the
    preview shows the same projection.

    Raises:
        DomainValidationError: a mapped source is not in the schema, or no source
            is named at all.
    """
    available = pq.ParquetFile(path).schema_arrow.names
    missing = sorted({source for source in mapping.values() if source and source not in available})
    if missing:
        raise DomainValidationError(
            f"These mapped columns are not in the imported dataset: {_join_capped(missing)}. "
            f"Available columns: {_join_capped(available)}."
        )
    if not any(mapping.values()):
        raise DomainValidationError(
            "The column mapping names no source column, so the imported dataset "
            f"would have no columns. Available columns: {_join_capped(available)}."
        )


async def stage_parquet(
    client: httpx.AsyncClient,
    *,
    plan: hf_import.HfImportPlan,
    staging_dir: Path,
    dataset_id: UUID,
    name: str,
    max_rows: int,
    max_bytes: int,
    token: str | None,
) -> StagedParquet:
    """Fetch the plan's train (and optional validation) parquet into staging.

    Writes ``<staging_dir>/<dataset_id>/<name>_train.parquet`` — the filename the
    upload runner, the local runner and the trainer's ``--train_file`` all expect.
    Streaming row groups rather than loading whole files mirrors
    ``datasets_io``'s ``PARQUET_BATCH_SIZE`` convention. When
    ``plan.validation_split`` is set, also writes a ``<name>_validation.parquet``
    sibling from ``plan.validation_urls``, at the same ``max_rows`` cap.

    ``max_bytes`` is the download budget for the whole import, *shared* by the two
    splits — matching the plan-time gate, which sizes train and validation
    together — so validation is charged for what train already spent.

    Raises:
        DomainValidationError: a split downloaded more than ``max_bytes``, or
            yielded no rows at all. An empty split is refused rather than handed
            on because a lazily-created ``ParquetWriter`` never opens
            ``destination`` in that case: the runner would then fail on a
            nonexistent path and report a misleading "check the file's format"
            detail, and a zero-row dataset cannot train regardless.
    """
    root = staging_dir / str(dataset_id)
    train_path = root / f"{name}_train.parquet"
    train_original, train_retained, train_bytes, train_unread = await _stage_split(
        client,
        urls=plan.train_urls,
        destination=train_path,
        max_rows=max_rows,
        max_bytes=max_bytes,
        token=token,
    )
    if train_retained == 0:
        raise DomainValidationError(
            f"The {plan.train_split!r} split of {plan.repo_id} has no rows to import."
        )
    validation_path: Path | None = None
    validation_original: int | None = None
    validation_retained: int | None = None
    validation_unread: int | None = None
    if plan.validation_split is not None:
        validation_path = root / f"{name}_validation.parquet"
        validation_original, validation_retained, _, validation_unread = await _stage_split(
            client,
            urls=plan.validation_urls,
            destination=validation_path,
            max_rows=max_rows,
            max_bytes=max_bytes,
            spent_bytes=train_bytes,
            token=token,
        )
        if validation_retained == 0:
            raise DomainValidationError(
                f"The {plan.validation_split!r} split of {plan.repo_id} has no rows to import."
            )
    return StagedParquet(
        train_path=train_path,
        train_original_rows=train_original,
        train_retained_rows=train_retained,
        validation_path=validation_path,
        validation_original_rows=validation_original,
        validation_retained_rows=validation_retained,
        train_unread_shards=train_unread,
        validation_unread_shards=validation_unread,
    )


# `HfImportService` is built fresh per request (see `api/deps.get_hf_import_service`),
# so once a request returns, nothing keeps that instance's own `self` alive. A task
# set living on `self` would then have no external root — unlike
# `InProcessDatasetUploadRunner`'s `_tasks`, which lives on a process-wide singleton
# and is rooted by that. Rooting the set here, at module scope, keeps a strong
# reference to every in-flight background import for the life of the process.
_background_tasks: set[asyncio.Task[None]] = set()

_fetch_semaphore: asyncio.Semaphore | None = None
"""Process-wide cap on concurrent HF-import fetches; built on first use by
``_get_fetch_semaphore`` below, from whichever request's settings gets there first.
"""


def _get_fetch_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """Return the process-wide semaphore bounding concurrent HF-import fetches.

    Built once — the same one-time-construction shape as
    ``api.deps._shared_upload_runner``'s own semaphore — but sized from the
    settings the constructing ``HfImportService`` was actually given, rather
    than a raw ``get_settings()`` call, so an overridden
    ``dataset_upload_max_concurrent`` is respected rather than silently bypassed.
    """
    global _fetch_semaphore
    if _fetch_semaphore is None:
        _fetch_semaphore = asyncio.Semaphore(max_concurrent)
    return _fetch_semaphore


class HfImportService:
    """Search, inspect, preview and commit a HuggingFace dataset import."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        viewer_client: httpx.AsyncClient,
        settings: Settings,
        repository: DatasetRepository,
        principal: Principal,
        runner: DatasetUploadRunner | None = None,
        staging_dir: Path | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._client = client
        self._viewer_client = viewer_client
        self._settings = settings
        self._repository = repository
        self._principal = principal
        self._token = os.environ.get(settings.hf_token_env)
        # All three are optional so tests exercising search/splits/preview need
        # not supply them; `import_dataset` only schedules staging when they are
        # all present (see `api/deps.get_hf_import_service`, the one production
        # caller that supplies them). `session_factory` lets the background task
        # open its OWN session — the request-scoped one behind `repository` is
        # closed by the time that task runs, the same reason
        # `InProcessDatasetUploadRunner.process` opens its own.
        self._runner = runner
        self._staging_dir = staging_dir
        self._session_factory = session_factory

    def _token_for(self, repo_id: str) -> str | None:
        """Narrow the resolved server token to ``repo_id``'s allowlisted namespace."""
        return hf_hub.token_for(
            repo_id, token=self._token, namespaces=self._settings.hf_import_namespaces
        )

    async def search(self, *, query: str, limit: int) -> list[str]:
        """Search the Hub for dataset repo ids matching ``query``.

        No single ``repo_id`` exists here to narrow a token against via
        ``token_for``, so this always searches anonymously (public repos only).

        Raises:
            HfHubUnreachableError: the Hub could not be reached.
        """
        try:
            return await hf_hub.search_datasets(
                self._client,
                base_url=self._settings.hf_hub_base_url,
                query=query,
                limit=limit,
                token=None,
            )
        except hf_hub.HfHubUnavailable as exc:
            raise HfHubUnreachableError() from exc

    async def splits(self, *, repo_id: str) -> HfDatasetSplits:
        """Resolve ``repo_id``'s pinned revision and its configs/splits.

        Raises:
            HfDatasetNotConvertedError: no parquet branch yet (try later).
            HfDatasetNotTabularError: the parquet branch lists no configs/splits.
            HfHubUnreachableError: the Hub could not be reached.
        """
        token = self._token_for(repo_id)
        try:
            revision = await hf_hub.resolve_revision(
                self._client,
                base_url=self._settings.hf_hub_base_url,
                repo_id=repo_id,
                token=token,
            )
            branch = await hf_hub.list_parquet_files(
                self._client,
                base_url=self._settings.hf_hub_base_url,
                repo_id=repo_id,
                revision=revision,
                token=token,
            )
        except hf_hub.HfDatasetNotConverted as exc:
            raise HfDatasetNotConvertedError(repo_id) from exc
        except hf_hub.HfNoTabularData as exc:
            raise HfDatasetNotTabularError(repo_id) from exc
        except hf_hub.HfHubUnavailable as exc:
            raise HfHubUnreachableError() from exc
        configs = {config: sorted(splits) for config, splits in branch.items()}
        return HfDatasetSplits(repo_id=repo_id, revision=revision, configs=configs)

    async def _plan(
        self,
        *,
        repo_id: str,
        revision: str,
        config: str,
        train_split: str,
        validation_split: str | None,
    ) -> hf_import.HfImportPlan:
        """Pin the revision, pick split URLs, and size-gate (shared by preview/import).

        Raises:
            DomainValidationError: ``config``/a split is absent from the parquet
                branch, or the selected files exceed ``hf_import_max_bytes``.
            HfDatasetNotConvertedError: no parquet branch yet (try later).
            HfDatasetNotTabularError: the parquet branch lists no configs/splits.
            HfHubUnreachableError: the Hub could not be reached.
        """
        try:
            return await hf_import.build_plan(
                self._client,
                base_url=self._settings.hf_hub_base_url,
                repo_id=repo_id,
                revision=revision,
                config=config,
                train_split=train_split,
                validation_split=validation_split,
                token=self._token_for(repo_id),
                max_bytes=self._settings.hf_import_max_bytes,
            )
        except (hf_import.HfImportTooLarge, ValueError) as exc:
            raise DomainValidationError(str(exc)) from exc
        except hf_hub.HfDatasetNotConverted as exc:
            raise HfDatasetNotConvertedError(repo_id) from exc
        except hf_hub.HfNoTabularData as exc:
            raise HfDatasetNotTabularError(repo_id) from exc
        except hf_hub.HfHubUnavailable as exc:
            raise HfHubUnreachableError() from exc

    async def preview(self, request: HfPreviewRequest) -> HfImportPreview:
        """Size-gate the selection, then sample and map up to 100 rows.

        The sample comes from HF's dataset *viewer*, which serves the latest
        conversion and accepts no revision parameter -- so it can differ from the
        snapshot ``revision`` pins, and the ``revision`` returned here is what an
        import *will* read rather than where this sample came from. Deliberate:
        sampling the pinned parquet instead costs a whole-shard download (2 GiB on
        some repos, ~19s even on a 14 MiB one) on the wizard's most-repeated route,
        and a mapping made against stale column names cannot reach a stored
        dataset -- ``_reject_unmapped_sources`` re-checks the real staged columns
        at import. The ranged-read alternative that would buy back byte-identical
        sampling is measured and deferred in
        ``docs/superpowers/specs/2026-09-16-hf-dataset-import-design.md``.

        Raises:
            DomainValidationError: ``config``/a split is absent, or the dataset
                exceeds ``hf_import_max_bytes``.
            HfDatasetNotConvertedError: no parquet branch yet (try later).
            HfDatasetNotTabularError: the parquet branch lists no configs/splits.
            HfHubUnreachableError: the Hub could not be reached.
            HfPreviewUnavailableError: the viewer cannot sample this dataset.
        """
        plan = await self._plan(
            repo_id=request.repo_id,
            revision=request.revision,
            config=request.config,
            train_split=request.train_split,
            validation_split=request.validation_split,
        )
        try:
            raw_rows = await hf_viewer.fetch_rows(
                self._viewer_client,
                base_url=self._settings.hf_viewer_base_url,
                repo_id=plan.repo_id,
                split=_viewer_split(plan.train_split),
                limit=_PREVIEW_SAMPLE_ROWS,
                token=self._token_for(plan.repo_id),
                config=plan.config,
            )
        except hf_viewer.HFViewerUnavailable as exc:
            raise HfPreviewUnavailableError(plan.repo_id) from exc
        mapped_rows = hf_import.apply_mapping(raw_rows, request.column_mapping)
        # Only targets with a source can be required. `apply_mapping` skips a blank
        # one, so requiring that target made `survived` 0 for every partly-mapped
        # wizard state -- which `survival_count` documents as "the mapping is simply
        # wrong and the caller blocks", while the import path accepts exactly those
        # mappings (see `_reject_unmapped_sources`). This module promises the preview
        # computes what the import will compute, so the two must agree. An all-blank
        # mapping is the case the import *does* refuse, and it stays 0 here rather
        # than counting every row as intact over an empty required set.
        required = [target for target, source in request.column_mapping.items() if source]
        survived = (
            hf_import.survival_count(raw_rows, request.column_mapping, required) if required else 0
        )
        # Union of keys across every sampled row, first-seen order preserved --
        # not just row 0's keys, which would drop a column absent there on
        # ragged data (e.g. a column empty/missing in a majority of rows).
        columns = list(dict.fromkeys(key for row in raw_rows for key in row))
        return HfImportPreview(
            revision=plan.revision,
            columns=columns,
            raw_rows=raw_rows,
            mapped_rows=mapped_rows,
            sampled=len(raw_rows),
            survived=survived,
        )

    async def import_dataset(self, request: HfImportRequest) -> DatasetRead:
        """Create a dataset row owned by the caller and flip it to ``importing``.

        Re-validates the plan (revision, split, size) before creating any row,
        so an obviously-bad request never leaves a stuck ``importing`` dataset
        behind. Once the row exists, schedules ``stage_and_submit`` as a
        background task and returns without awaiting it — the same ``202``-then-
        background shape the regular upload path uses (see
        ``DatasetUploadRunner.submit``) — so this call stays fast even though the
        fetch it kicks off can take minutes.

        Raises:
            CallerNotProvisionedError: the caller has no ``user_id`` to own the row.
            DomainValidationError / HfDatasetNotConvertedError /
                HfDatasetNotTabularError / HfHubUnreachableError: as raised by
                ``_plan``.
        """
        plan = await self._plan(
            repo_id=request.repo_id,
            revision=request.revision,
            config=request.config,
            train_split=request.train_split,
            validation_split=request.validation_split,
        )
        owner_id = self._principal.user_id
        if owner_id is None:
            raise CallerNotProvisionedError()
        dataset = await self._repository.create(
            user_id=str(owner_id),
            name=request.name,
            description=request.description,
            data_format="parquet",
        )
        await self._repository.set_status(dataset.id, DatasetStatus.IMPORTING)
        if (
            self._runner is not None
            and self._staging_dir is not None
            and (self._session_factory is not None)
        ):
            task = asyncio.create_task(
                self._stage_and_submit_in_background(
                    dataset.id,
                    plan=plan,
                    name=request.name,
                    validation_percentage=request.validation_percentage,
                    column_mapping=request.column_mapping,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        return dataset_to_read(dataset, [])

    async def _stage_and_submit_in_background(
        self,
        dataset_id: UUID,
        *,
        plan: hf_import.HfImportPlan,
        name: str,
        validation_percentage: int | None,
        column_mapping: dict[str, str] | None,
    ) -> None:
        """Run ``stage_and_submit`` off the request path, in its own DB session.

        ``import_dataset``'s own ``self._repository`` is request-scoped and gone
        by the time this task runs, so a fresh repository is opened from
        ``self._session_factory`` here — the caller (``import_dataset``) only
        schedules this when a factory was configured. Bounded by
        ``_get_fetch_semaphore`` (so N concurrent imports do not mean N concurrent
        fetches of up to ``hf_import_max_bytes`` each) and by
        ``dataset_processing_timeout_seconds`` via ``asyncio.wait_for``, mirroring
        ``InProcessDatasetUploadRunner.process``. Any failure before
        ``stage_and_submit`` hands off to the runner (the fetch, the provenance
        write) is caught here, lands the row in ``error``, and cleans up staging —
        the runner's own failure handling in ``process`` only covers what happens
        after ``submit`` is called, and its ``finally: shutil.rmtree`` only runs
        once ``submit`` has succeeded.
        """
        if self._session_factory is None:
            raise RuntimeError(
                "_stage_and_submit_in_background requires a configured session_factory; "
                "import_dataset only schedules this once one is set."
            )
        try:
            async with (
                _get_fetch_semaphore(self._settings.dataset_upload_max_concurrent),
                self._session_factory() as session,
            ):
                repository = SqlAlchemyDatasetRepository(session)
                try:
                    await asyncio.wait_for(
                        self.stage_and_submit(
                            dataset_id,
                            plan=plan,
                            name=name,
                            validation_percentage=validation_percentage,
                            column_mapping=column_mapping,
                            repository=repository,
                        ),
                        self._settings.dataset_processing_timeout_seconds,
                    )
                except Exception as exc:
                    logger.exception("HuggingFace import staging failed for %s", dataset_id)
                    await session.rollback()
                    await repository.set_status(
                        dataset_id, DatasetStatus.ERROR, status_detail=_safe_staging_detail(exc)
                    )
                    if self._staging_dir is not None:
                        shutil.rmtree(self._staging_dir / str(dataset_id), ignore_errors=True)
        except asyncio.CancelledError:
            # `main.py`'s shutdown drain cancels whatever is still fetching once its
            # grace period expires, and CancelledError is a BaseException, so neither
            # `except Exception` here sees it. Staging would then keep its
            # partially-written parquet for good: the row stays `importing`, and the
            # only remedy -- delete it and import again -- frees
            # `<dataset_storage_dir>/<id>`, never the `.staging/<id>` sibling that
            # `LocalStorageBackend.delete` does not know about.
            #
            # Deliberately not a `finally`: on the success path `stage_and_submit`
            # has already returned from `runner.submit`, which only *schedules*
            # processing, so the runner's own task is still reading these files.
            # Clearing them unconditionally would delete them out from under it.
            if self._staging_dir is not None:
                shutil.rmtree(self._staging_dir / str(dataset_id), ignore_errors=True)
            raise
        except Exception:
            # Everything the inner handler needs to land `error` -- the semaphore,
            # the session -- is itself acquired above it, and the status write is
            # a DB call of its own. A failure in either (the DB restarted between
            # the 202 and this first tick; a module-global semaphore awaited from a
            # second event loop in one process) would otherwise escape the task
            # body entirely: nothing awaits this task, so asyncio would only log
            # "Task exception was never retrieved" and the row would sit in
            # `importing` with no record of why. There is no session to write a
            # status with on this path, so the best available outcome is a logged
            # failure and no staging left behind.
            logger.exception("HuggingFace import could not run for %s", dataset_id)
            if self._staging_dir is not None:
                shutil.rmtree(self._staging_dir / str(dataset_id), ignore_errors=True)

    async def stage_and_submit(
        self,
        dataset_id: UUID,
        *,
        plan: hf_import.HfImportPlan,
        name: str,
        validation_percentage: int | None,
        column_mapping: dict[str, str] | None,
        repository: DatasetRepository | None = None,
    ) -> None:
        """Fetch ``plan`` into staging, record provenance, and hand off to the runner.

        Staging goes through :func:`stage_parquet`. ``repository`` defaults to
        ``self._repository`` (used by a direct, synchronous call, e.g. in tests);
        ``import_dataset`` instead passes one bound to a fresh session opened from
        ``self._session_factory``, since its own ``self._repository`` is
        request-scoped and gone by the time the background task backing this call
        runs.

        Raises:
            RuntimeError: no ``runner``/``staging_dir`` was configured at
                construction time.
            DomainValidationError: staging exceeded ``hf_import_max_bytes``, a
                selected split had no rows, or ``column_mapping`` names a column
                the staged parquet does not have (see
                :func:`_reject_unmapped_sources`). The caller
                (``_stage_and_submit_in_background``) lands the row in ``error``
                with that message and clears staging.
        """
        if self._runner is None or self._staging_dir is None:
            raise RuntimeError(
                "HfImportService has no runner/staging_dir configured; both must "
                "be passed to __init__ before calling stage_and_submit."
            )
        repo = repository if repository is not None else self._repository
        staged = await stage_parquet(
            self._client,
            plan=plan,
            staging_dir=self._staging_dir,
            dataset_id=dataset_id,
            name=name,
            max_rows=self._settings.hf_import_max_rows,
            max_bytes=self._settings.hf_import_max_bytes,
            token=self._token_for(plan.repo_id),
        )
        if column_mapping:
            await asyncio.to_thread(_reject_unmapped_sources, staged.train_path, column_mapping)
            if staged.validation_path is not None:
                await asyncio.to_thread(
                    _reject_unmapped_sources, staged.validation_path, column_mapping
                )
        provenance = {
            "column_mapping": column_mapping,
            "train_original_rows": staged.train_original_rows,
            "train_retained_rows": staged.train_retained_rows,
            "validation_split": plan.validation_split,
            "validation_original_rows": staged.validation_original_rows,
            "validation_retained_rows": staged.validation_retained_rows,
            "train_unread_shards": staged.train_unread_shards,
            "train_truncated": staged.train_truncated,
            "validation_unread_shards": staged.validation_unread_shards,
            "validation_truncated": staged.validation_truncated,
        }
        await repo.set_hf_provenance(
            dataset_id,
            repo_id=plan.repo_id,
            revision=plan.revision,
            config=plan.config,
            split=plan.train_split,
            provenance=provenance,
        )
        await self._runner.submit(
            dataset_id,
            name=name,
            data_format="parquet",
            train=staged.train_path,
            validation=staged.validation_path,
            validation_percentage=validation_percentage,
            column_mapping=column_mapping,
        )
