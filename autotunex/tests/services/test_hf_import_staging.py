# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Staging a HuggingFace import into the existing upload runner's staging dir."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.core.exceptions import DomainValidationError
from autotunex.db.repositories.sqlalchemy import SqlAlchemyDatasetRepository
from autotunex.models.auth import Principal
from autotunex.services.dataset_runner import NoOpDatasetUploadRunner
from autotunex.services.hf_import import HfImportPlan
from tests.conftest import make_settings
from tests.services.test_dataset_runner import _seed_dataset

_ROOMY_BYTES = 1 << 30
"""A download budget far above every payload here, so only the row cap binds.

Tests that exercise the byte cap itself pass their own small ``max_bytes``.
"""


@pytest.mark.parametrize("first_rows, shard_count", [(5, 2), (6, 1), (5, 1), (4, 1)])
async def test_provenance_records_unread_shards_at_the_row_cap(
    engine: AsyncEngine, tmp_path: Path, first_rows: int, shard_count: int
) -> None:
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    downloaded: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        downloaded.append(str(request.url))
        return httpx.Response(200, content=_parquet_bytes(first_rows))

    urls = [f"https://cdn/{i}.parquet" for i in range(shard_count)]
    async with (
        factory() as session,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client,
    ):
        repository = SqlAlchemyDatasetRepository(session)
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings().model_copy(update={"hf_import_max_rows": 5}),
            repository=repository,
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=NoOpDatasetUploadRunner(),
            staging_dir=tmp_path,
        )
        await service.stage_and_submit(
            dataset_id,
            plan=_plan(train_urls=urls),
            name="ds",
            validation_percentage=None,
            column_mapping={"input": "instruction"},
        )
        dataset = await repository.get(dataset_id)
        assert dataset is not None and dataset.hf_provenance is not None
        assert dataset.hf_provenance["train_original_rows"] == first_rows
        assert dataset.hf_provenance["train_retained_rows"] == min(first_rows, 5)
        assert dataset.hf_provenance["train_unread_shards"] == shard_count - 1
        assert dataset.hf_provenance["train_truncated"] == (first_rows > 5 or shard_count > 1)
        assert dataset.hf_provenance["validation_truncated"] is None
    assert downloaded == urls[:1]


def _parquet_bytes(rows: int, *, row_group_size: int | None = None) -> bytes:
    """A minimal in-memory parquet with the columns a mapping would target.

    ``row_group_size`` forces multiple pyarrow row groups (and, in turn,
    multiple ``iter_batches`` batches at ``PARQUET_BATCH_SIZE``) so a caller
    can exercise truncation that spans more than one batch. Omitted, pyarrow
    puts every row in a single row group -- the cheap single-batch case most
    tests here want.
    """
    table = pa.table(
        {
            "instruction": [f"q{n}" for n in range(rows)],
            "response": [f"a{n}" for n in range(rows)],
        }
    )
    sink = pa.BufferOutputStream()
    if row_group_size is None:
        pq.write_table(table, sink)
    else:
        pq.write_table(table, sink, row_group_size=row_group_size)
    return bytes(sink.getvalue().to_pybytes())


def _plan(
    *,
    validation_split: str | None = None,
    validation_urls: list[str] | None = None,
    train_urls: list[str] | None = None,
) -> HfImportPlan:
    return HfImportPlan(
        repo_id="o/r",
        revision="abc123",
        config="default",
        train_split="train",
        validation_split=validation_split,
        train_urls=train_urls or ["https://cdn/0.parquet"],
        validation_urls=validation_urls or [],
        gated_bytes=1024,
    )


async def test_staging_writes_the_runners_expected_filename(tmp_path: Path) -> None:
    """The staged name must match what the runner and the trainer both expect."""
    from autotunex.services.hf_import_service import stage_parquet

    dataset_id = uuid4()
    staging = tmp_path / ".staging"
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        staged = await stage_parquet(
            client,
            plan=_plan(),
            staging_dir=staging,
            dataset_id=dataset_id,
            name="ds",
            max_rows=50_000,
            max_bytes=_ROOMY_BYTES,
            token=None,
        )

    assert staged.train_path == staging / str(dataset_id) / "ds_train.parquet"
    assert pq.ParquetFile(staged.train_path).metadata.num_rows == 5


async def test_staging_sends_the_bearer_token_when_given(tmp_path: Path) -> None:
    """The bulk-download path (``_stage_split``) delegates to ``hf_hub._headers``.

    It no longer builds its own header dict inline; confirm the delegation
    actually reaches the wire, since every other test in this file passes
    ``token=None``.
    """
    from autotunex.services.hf_import_service import _stage_split

    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer tok"
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await _stage_split(
            client,
            urls=_plan().train_urls,
            destination=tmp_path / "train.parquet",
            max_rows=50_000,
            max_bytes=_ROOMY_BYTES,
            token="tok",
        )


async def test_staging_truncates_at_the_row_cap(tmp_path: Path) -> None:
    """Bounded ingest: a larger dataset imports its first N rows, not all of them."""
    from autotunex.services.hf_import_service import stage_parquet

    dataset_id = uuid4()
    payload = _parquet_bytes(100)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        staged = await stage_parquet(
            client,
            plan=_plan(),
            staging_dir=tmp_path / ".staging",
            dataset_id=dataset_id,
            name="ds",
            max_rows=10,
            max_bytes=_ROOMY_BYTES,
            token=None,
        )

    assert pq.ParquetFile(staged.train_path).metadata.num_rows == 10


async def test_staging_truncates_the_cap_across_batches_row_groups_and_shards(
    tmp_path: Path,
) -> None:
    """The row cap holds across pyarrow batches/row-groups and across shards.

    ``test_staging_truncates_at_the_row_cap`` writes a single row group, so
    ``iter_batches`` only ever yields one batch and the cap is enforced by a
    single ``batch.slice``. Here shard 0 (5,000 rows) is *under* the cap on
    its own, so it must be fully retained and the loop must continue into
    shard 1 (25,000 rows split into five row groups, so ``iter_batches``
    yields several batches -- verified separately to be
    ``[10_000, 10_000, 5_000]``), where the 12,345-row cap lands mid-batch on
    a non-round boundary: 5,000 from shard 0 plus a mid-batch slice of
    shard 1's remaining 7,345. A regression that reset
    ``retained_rows``/``original_rows`` to 0 at the top of each URL
    iteration would compute ``remaining`` against 0 once inside shard 1 and
    over-retain there -- the ``retained_rows``/staged-row-count assertions
    below catch that directly. A third shard is wired in and must never be
    fetched, since the cap is already met once shard 1 is done -- exercising
    the cross-URL ``break`` too.
    """
    from autotunex.services.hf_import_service import _stage_split

    shard0 = _parquet_bytes(5_000)
    shard1 = _parquet_bytes(25_000, row_group_size=5_000)
    plan = _plan(
        train_urls=[
            "https://cdn/0.parquet",
            "https://cdn/1.parquet",
            "https://cdn/2.parquet",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "2.parquet" in url:
            pytest.fail("shard 2 must never be fetched once the cap is already met")
        if "1.parquet" in url:
            return httpx.Response(200, content=shard1)
        return httpx.Response(200, content=shard0)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        original_rows, retained_rows, _, _ = await _stage_split(
            client,
            urls=plan.train_urls,
            destination=tmp_path / "train.parquet",
            max_rows=12_345,
            max_bytes=_ROOMY_BYTES,
            token=None,
        )

    assert retained_rows == 12_345
    assert pq.ParquetFile(tmp_path / "train.parquet").metadata.num_rows == 12_345
    # Shards 0 and 1 were both opened (5_000 + 25_000 rows); shard 2 never was,
    # so its rows are not counted: original_rows is honest about what was
    # actually downloaded, not a claim about the whole split -- see
    # _stage_split's own docstring.
    assert original_rows == 30_000


async def test_staging_writes_a_validation_sibling_when_the_plan_has_one(tmp_path: Path) -> None:
    """A plan with a validation split gets a `_validation.parquet` sibling too."""
    from autotunex.services.hf_import_service import stage_parquet

    dataset_id = uuid4()
    train_payload = _parquet_bytes(5)
    validation_payload = _parquet_bytes(3)

    def handler(request: httpx.Request) -> httpx.Response:
        if "validation" in str(request.url):
            return httpx.Response(200, content=validation_payload)
        return httpx.Response(200, content=train_payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        staged = await stage_parquet(
            client,
            plan=_plan(
                validation_split="validation",
                validation_urls=["https://cdn/validation-0.parquet"],
            ),
            staging_dir=tmp_path / ".staging",
            dataset_id=dataset_id,
            name="ds",
            max_rows=50_000,
            max_bytes=_ROOMY_BYTES,
            token=None,
        )

    validation = tmp_path / ".staging" / str(dataset_id) / "ds_validation.parquet"
    assert pq.ParquetFile(staged.train_path).metadata.num_rows == 5
    assert pq.ParquetFile(validation).metadata.num_rows == 3
    assert staged.validation_path == validation


async def _seeded_repository_and_dataset(
    engine: AsyncEngine,
) -> tuple[SqlAlchemyDatasetRepository, UUID]:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    session = factory()
    return SqlAlchemyDatasetRepository(session), dataset_id


async def test_set_hf_provenance_writes_the_source_columns_and_refreshes(
    engine: AsyncEngine,
) -> None:
    """The provenance setter must follow set_status's get -> commit -> refresh discipline."""
    repository, dataset_id = await _seeded_repository_and_dataset(engine)

    # Hold the same identity-mapped object set_hf_provenance will mutate, fetched
    # *before* the call under test. Reading train_file off this same reference --
    # with no intervening get() -- is what makes the assertion below depend on
    # set_hf_provenance's own internal refresh(): a second get() would issue its
    # own select() and incidentally repopulate the expired Computed columns
    # regardless of whether that refresh() line exists.
    dataset = await repository.get(dataset_id)
    assert dataset is not None

    await repository.set_hf_provenance(
        dataset_id,
        repo_id="o/r",
        revision="abc123",
        config="default",
        split="train",
        provenance={"train_original_rows": 5, "train_retained_rows": 5},
    )

    assert dataset.hf_repo_id == "o/r"
    assert dataset.hf_revision == "abc123"
    assert dataset.hf_config == "default"
    assert dataset.hf_split == "train"
    assert dataset.hf_provenance == {"train_original_rows": 5, "train_retained_rows": 5}
    # The Computed train_file column is expired by the commit above regardless of
    # which columns changed; reading it here without an intervening get() is only
    # safe because set_hf_provenance's internal refresh() repopulated it. Removing
    # that refresh() makes this line raise MissingGreenlet.
    assert dataset.train_file == "ds_train"


async def test_stage_and_submit_records_provenance_and_calls_the_runner(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """stage_and_submit fetches into staging, writes provenance, then hands off."""
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    repository = SqlAlchemyDatasetRepository(factory())
    staging = tmp_path / ".staging"
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    runner = NoOpDatasetUploadRunner()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=repository,
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=runner,
            staging_dir=staging,
        )

        await service.stage_and_submit(
            dataset_id,
            plan=_plan(),
            name="ds",
            validation_percentage=None,
            column_mapping={"input": "instruction"},
        )

    refreshed = await repository.get(dataset_id)
    assert refreshed is not None
    assert refreshed.hf_repo_id == "o/r"
    assert refreshed.hf_provenance is not None
    assert refreshed.hf_provenance["train_retained_rows"] == 5
    assert refreshed.hf_provenance["column_mapping"] == {"input": "instruction"}
    assert runner.submitted == [dataset_id]


async def test_stage_and_submit_requires_a_configured_runner(tmp_path: Path) -> None:
    """Without runner/staging_dir, stage_and_submit refuses rather than guessing."""
    from autotunex.services.hf_import_service import HfImportService

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    ) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=None,  # type: ignore[arg-type]  # never touched: the runner check runs first
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
        )

        with pytest.raises(RuntimeError):
            await service.stage_and_submit(
                uuid4(),
                plan=_plan(),
                name="ds",
                validation_percentage=None,
                column_mapping=None,
            )


async def test_staging_refuses_a_download_past_the_byte_budget(tmp_path: Path) -> None:
    """The byte cap must bind even when the plan-time size gate saw nothing.

    ``head_total_bytes`` contributes 0 for a 200 that omits ``content-length``, so
    a plan can clear ``hf_import_max_bytes`` at any real size. And ``max_rows``
    truncates only *after* a shard is on disk in full, so it bounds rows, not
    bytes. This inline cap is the only thing between that pair of gaps and a
    filled staging volume.
    """
    from autotunex.services.hf_import_service import _stage_split

    payload = _parquet_bytes(5_000)

    def handler(request: httpx.Request) -> httpx.Response:
        # No content-length claim: exactly the response shape the plan gate scores
        # as 0 bytes.
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainValidationError, match="import limit"):
            await _stage_split(
                client,
                urls=["https://cdn/0.parquet"],
                destination=tmp_path / "train.parquet",
                max_rows=50_000,
                max_bytes=64,
                token=None,
            )

    assert not (tmp_path / ".train.parquet.0.tmp").exists()


async def test_staging_shares_one_byte_budget_across_both_splits(tmp_path: Path) -> None:
    """Validation draws on what train left, matching the plan gate's combined sizing.

    ``head_total_bytes`` scores ``[*train_urls, *validation_urls]`` against a single
    ``hf_import_max_bytes``, so a per-split budget would let an import land at twice
    the ceiling it was gated on.
    """
    from autotunex.services.hf_import_service import stage_parquet

    train_payload = _parquet_bytes(5)
    validation_payload = _parquet_bytes(3)

    def handler(request: httpx.Request) -> httpx.Response:
        if "validation" in str(request.url):
            return httpx.Response(200, content=validation_payload)
        return httpx.Response(200, content=train_payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainValidationError) as caught:
            await stage_parquet(
                client,
                plan=_plan(
                    validation_split="validation",
                    validation_urls=["https://cdn/validation-0.parquet"],
                ),
                staging_dir=tmp_path / ".staging",
                dataset_id=uuid4(),
                name="ds",
                max_rows=50_000,
                # Exactly train's size: train fits, and validation then has nothing
                # left to spend.
                max_bytes=len(train_payload),
                token=None,
            )

    # The detail names the configured ceiling, not validation's leftover remainder
    # -- which is 0 here, and "exceeds the 0-byte import limit" tells a user nothing.
    assert f"{len(train_payload)}-byte import limit" in caught.value.detail


async def test_staging_refuses_a_split_with_no_rows(tmp_path: Path) -> None:
    """A zero-row split must fail here, not downstream on a file that was never made.

    The ``ParquetWriter`` is created lazily on the first batch and a zero-row
    parquet yields none, so ``destination`` is never opened. Handing that path to
    the runner produced a ``FileNotFoundError`` reported as the misleading "check
    the file's format and contents" detail.
    """
    from autotunex.services.hf_import_service import stage_parquet

    dataset_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_parquet_bytes(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainValidationError, match="no rows to import"):
            await stage_parquet(
                client,
                plan=_plan(),
                staging_dir=tmp_path / ".staging",
                dataset_id=dataset_id,
                name="ds",
                max_rows=50_000,
                max_bytes=_ROOMY_BYTES,
                token=None,
            )

    assert not (tmp_path / ".staging" / str(dataset_id) / "ds_train.parquet").exists()


async def test_staging_transcodes_off_the_event_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pyarrow decode/re-encode must not run inline in the coroutine.

    ``iter_batches``/``write_batch`` over a shard is blocking and CPU-bound; inline
    it stalls every other request in the process for the length of the transcode.
    Asserting the thread identity is the deterministic form of that claim -- a
    wall-clock "the loop stayed responsive" probe would be timing-dependent.
    """
    from autotunex.services import hf_import_service

    loop_thread = threading.get_ident()
    ran_on: list[int] = []
    original = hf_import_service._copy_capped_rows

    def spy(*args: object, **kwargs: object) -> object:
        ran_on.append(threading.get_ident())
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(hf_import_service, "_copy_capped_rows", spy)
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await hf_import_service._stage_split(
            client,
            urls=["https://cdn/0.parquet"],
            destination=tmp_path / "train.parquet",
            max_rows=50_000,
            max_bytes=_ROOMY_BYTES,
            token=None,
        )

    assert ran_on, "the transcode helper never ran"
    assert all(thread != loop_thread for thread in ran_on)


async def test_a_cancelled_staging_closes_the_writer_only_after_its_thread_is_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer belongs to the worker thread until its transcode returns.

    One ``ParquetWriter`` spans every shard of a split, and ``asyncio.to_thread``
    cannot be cancelled mid-run: the thread keeps going. So a cancellation while
    shard 2 is being transcoded -- the caller's ``wait_for`` timeout, or
    ``main.py``'s shutdown drain -- used to reach ``finally: writer.close()`` on
    the event loop while that thread was still inside ``write_batch`` on the very
    same, non-thread-safe object. Two shards are needed to show it: with one, the
    local ``writer`` is still ``None`` and the close is skipped for the wrong
    reason.
    """
    from autotunex.services import hf_import_service

    started = threading.Event()
    worker_done = threading.Event()
    closed_while_worker_running: list[bool] = []
    calls = 0

    class _RecordingWriter:
        def close(self) -> None:
            closed_while_worker_running.append(not worker_done.is_set())

    shared_writer = _RecordingWriter()

    def slow_copy(*args: object, **kwargs: object) -> object:
        # Shards run strictly one after another, so this needs no locking.
        nonlocal calls
        calls += 1
        if calls == 1:
            return shared_writer, 5, 5
        started.set()
        time.sleep(0.3)
        worker_done.set()
        return shared_writer, 5, 10

    monkeypatch.setattr(hf_import_service, "_copy_capped_rows", slow_copy)
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(
            hf_import_service._stage_split(
                client,
                urls=["https://cdn/0.parquet", "https://cdn/1.parquet"],
                destination=tmp_path / "train.parquet",
                max_rows=50_000,
                max_bytes=_ROOMY_BYTES,
                token=None,
            )
        )
        await asyncio.to_thread(started.wait, 5)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(worker_done.wait, 5)
        for _ in range(200):
            if closed_while_worker_running:
                break
            await asyncio.sleep(0.01)

    assert closed_while_worker_running == [False], (
        "the writer was closed on the event loop while its worker thread still held it"
    )


async def test_a_background_import_whose_session_never_opens_stays_inside_its_task(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing awaits the background task, so an escaping exception is invisible.

    The semaphore and the session are acquired *outside* the handler that lands the
    row in ``error``, so a failure in either -- the DB restarted between the 202 and
    this first tick, or a module-global semaphore awaited from a second event loop --
    would escape the task body entirely. asyncio would then only log "Task exception
    was never retrieved" at some later collection, with nothing tying it to the
    dataset now stuck in ``importing``.
    """
    from autotunex.services.hf_import_service import HfImportService

    def _boom() -> AsyncSession:
        raise RuntimeError("database is gone")

    service = HfImportService(
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))),
        viewer_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ),
        settings=make_settings(hf_import_enabled=True),
        repository=SqlAlchemyDatasetRepository(cast("AsyncSession", None)),
        principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
        runner=NoOpDatasetUploadRunner(),
        staging_dir=tmp_path / ".staging",
        session_factory=cast("async_sessionmaker[AsyncSession]", _boom),
    )

    await service._stage_and_submit_in_background(
        uuid4(), plan=_plan(), name="ds", validation_percentage=None, column_mapping=None
    )

    assert "could not run" in caplog.text
    assert "database is gone" in caplog.text


async def test_a_cancelled_background_import_clears_its_staging_directory(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """``CancelledError`` is a BaseException, so no ``except Exception`` here sees it.

    That is the path ``main.py``'s shutdown drain takes once its grace period
    expires, and it used to skip both cleanup branches: the partially-written
    parquet stayed under ``.staging/<id>`` for good, since the row's only remedy
    (delete and import again) frees ``<dataset_storage_dir>/<id>`` and
    ``LocalStorageBackend.delete`` never touches the staging sibling.
    """
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    staging = tmp_path / ".staging"
    fetching = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        fetching.set()
        # Still downloading when the drain gives up on it.
        await asyncio.sleep(30)
        return httpx.Response(200, content=_parquet_bytes(5))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=SqlAlchemyDatasetRepository(factory()),
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=NoOpDatasetUploadRunner(),
            staging_dir=staging,
            session_factory=factory,
        )
        task = asyncio.create_task(
            service._stage_and_submit_in_background(
                dataset_id, plan=_plan(), name="ds", validation_percentage=None, column_mapping=None
            )
        )
        await asyncio.wait_for(fetching.wait(), 5)
        assert (staging / str(dataset_id)).is_dir(), "nothing was staged, so the check is vacuous"

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert not (staging / str(dataset_id)).exists()


async def test_stage_and_submit_refuses_a_mapping_that_names_no_source_column(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """An all-blank mapping is the same zero-column file by another route.

    ``HfImportRequest.column_mapping`` constrains the dict, not its values, so
    ``{"input": "", "output": ""}`` is a valid request body. Every source is then
    falsy, which the absent-from-schema check skipped, and ``remap_records``
    projected to ``select([])``: a parquet that round-trips to zero rows and zero
    columns, while the row flipped to ``ready`` reporting the *pre-remap* count.
    """
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    repository = SqlAlchemyDatasetRepository(factory())
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    runner = NoOpDatasetUploadRunner()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=repository,
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=runner,
            staging_dir=tmp_path / ".staging",
        )

        with pytest.raises(DomainValidationError, match="names no source column"):
            await service.stage_and_submit(
                dataset_id,
                plan=_plan(),
                name="ds",
                validation_percentage=None,
                column_mapping={"input": "", "output": ""},
            )

    assert runner.submitted == []
    refreshed = await repository.get(dataset_id)
    assert refreshed is not None
    assert refreshed.hf_repo_id is None


async def test_stage_and_submit_accepts_a_partly_blank_mapping(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """One blank target among several must stay tolerated, not become a refusal.

    ``apply_mapping`` documents skipping a blank source, and the preview shows that
    same projection, so a wizard that has only had one column picked so far still
    previews and imports. The guard's job is the all-blank case, where *no* column
    would survive.
    """
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    repository = SqlAlchemyDatasetRepository(factory())
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    runner = NoOpDatasetUploadRunner()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=repository,
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=runner,
            staging_dir=tmp_path / ".staging",
        )

        await service.stage_and_submit(
            dataset_id,
            plan=_plan(),
            name="ds",
            validation_percentage=None,
            column_mapping={"input": "instruction", "output": ""},
        )

    assert runner.submitted == [dataset_id]


async def test_stage_and_submit_refuses_a_mapping_the_staged_parquet_cannot_satisfy(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """A mapped column absent from the staged schema must fail before the hand-off.

    ``remap_records`` silently drops an unmatched source, so with no source
    matching it writes a file that round-trips to zero rows and zero columns while
    ``count_records`` reports 0 and the dataset still flips to ``ready`` -- a job
    would then train on an empty file. Preview cannot be relied on to catch it: it
    is optional, and it samples the dataset *viewer*, whose column names can differ
    from the parquet branch's.
    """
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    repository = SqlAlchemyDatasetRepository(factory())
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    runner = NoOpDatasetUploadRunner()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=repository,
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=runner,
            staging_dir=tmp_path / ".staging",
        )

        with pytest.raises(DomainValidationError, match="not in the imported dataset"):
            await service.stage_and_submit(
                dataset_id,
                plan=_plan(),
                name="ds",
                validation_percentage=None,
                column_mapping={"input": "no_such_column"},
            )

    assert runner.submitted == []
    refreshed = await repository.get(dataset_id)
    assert refreshed is not None
    assert refreshed.hf_repo_id is None


async def test_stage_and_submit_names_the_available_columns_when_a_mapping_misses(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """The refusal has to be actionable: say which columns the dataset does have."""
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=SqlAlchemyDatasetRepository(factory()),
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=NoOpDatasetUploadRunner(),
            staging_dir=tmp_path / ".staging",
        )

        with pytest.raises(DomainValidationError) as caught:
            await service.stage_and_submit(
                dataset_id,
                plan=_plan(),
                name="ds",
                validation_percentage=None,
                column_mapping={"input": "prompt", "output": "response"},
            )

    # `response` exists in _parquet_bytes' schema and `prompt` does not, so only
    # the genuinely-missing source is named -- with the real columns to fix it by.
    assert "prompt" in caught.value.detail
    assert "instruction, response" in caught.value.detail


async def test_staging_gives_the_download_a_read_budget_of_its_own(tmp_path: Path) -> None:
    """A GiB-scale shard must not inherit the timeout picked for an OIDC call.

    ``get_hf_http_client`` hands staging the shared ``app.state.http_client``,
    whose ``httpx.Timeout(10.0)`` ``main.py`` states is for the OIDC
    token-endpoint call. A shard that stalls longer than that mid-stream raises
    ``ReadTimeout``; ``_safe_staging_detail`` does not recognize it as a
    ``DomainValidationError``, so the row lands in ``error`` with the generic
    message and the whole partial download is discarded with no retry. The stream
    therefore states its own read budget.
    """
    from autotunex.services.hf_import_service import stage_parquet

    seen: list[object] = []
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=httpx.Timeout(10.0)
    ) as client:
        await stage_parquet(
            client,
            plan=_plan(),
            staging_dir=tmp_path / ".staging",
            dataset_id=uuid4(),
            name="ds",
            max_rows=50_000,
            max_bytes=_ROOMY_BYTES,
            token=None,
        )

    assert seen, "the staging download issued no request"
    timeout = cast(dict[str, float | None], seen[0])
    assert timeout["read"] is not None
    assert timeout["read"] > 10.0


async def test_stage_and_submit_records_which_split_fed_validation(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """Provenance must identify the evaluation source, not only count its rows.

    ``hf_split`` stores the *training* split, and the blob carries only
    ``validation_*`` counts -- so a ``test``-fed import was indistinguishable from
    a ``validation``-fed one whenever the two splits happen to be the same size,
    and the split's name was unrecoverable from the row either way.
    """
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    repository = SqlAlchemyDatasetRepository(factory())
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=repository,
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=NoOpDatasetUploadRunner(),
            staging_dir=tmp_path / ".staging",
        )

        await service.stage_and_submit(
            dataset_id,
            plan=_plan(validation_split="test", validation_urls=["https://cdn/v0.parquet"]),
            name="ds",
            validation_percentage=None,
            column_mapping={"input": "instruction"},
        )

    refreshed = await repository.get(dataset_id)
    assert refreshed is not None
    assert refreshed.hf_provenance is not None
    assert refreshed.hf_split == "train"
    assert refreshed.hf_provenance["validation_split"] == "test"


async def test_a_huge_column_mapping_cannot_overflow_status_detail(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    """The refusal message becomes ``status_detail``, which is bounded storage.

    ``status_detail`` is ``Text`` -- 64 KiB on MySQL -- and the write happens
    *inside* the staging error handler, so an oversized detail makes
    ``set_status`` raise there and leaves the row in ``importing`` with no
    terminal state to recover from: the upload path refuses a non-terminal row, so
    the only remedy is delete and re-import. The joined column list is
    caller-controlled and ``HfImportRequest`` caps neither its entries nor their
    length, so the message is capped where it is built.
    """
    from autotunex.services.hf_import_service import HfImportService

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    payload = _parquet_bytes(5)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = HfImportService(
            client=client,
            viewer_client=client,
            settings=make_settings(hf_import_enabled=True),
            repository=SqlAlchemyDatasetRepository(factory()),
            principal=Principal(email="u@example.com", provider="session", user_id=uuid4()),
            runner=NoOpDatasetUploadRunner(),
            staging_dir=tmp_path / ".staging",
        )

        with pytest.raises(DomainValidationError) as caught:
            await service.stage_and_submit(
                dataset_id,
                plan=_plan(),
                name="ds",
                validation_percentage=None,
                column_mapping={f"t{n}": f"absent_source_{n}" for n in range(5_000)},
            )

    assert len(caught.value.detail) < 2_000
