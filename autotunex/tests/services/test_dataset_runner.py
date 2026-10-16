"""InProcessDatasetUploadRunner terminal transitions, isolated with a fake backend.

The runner opens its OWN session (the request session is gone by the time it
runs), so these tests build a sessionmaker on the shared test engine and assert
by reading the row back through a fresh session.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.db.repositories.sqlalchemy import SqlAlchemyDatasetRepository
from autotunex.db.tables import UserTable
from autotunex.models.dataset import DatasetPreview
from autotunex.models.status import DatasetStatus
from autotunex.services.dataset_runner import (
    InProcessDatasetUploadRunner,
    NoOpDatasetUploadRunner,
)


class FakeStorageBackend:
    """A no-op backend: persist leaves files where they are and returns no refs."""

    def __init__(self) -> None:
        self.persisted: list[UUID] = []

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
        self.persisted.append(dataset_id)
        return None, None

    async def preview(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        artifact_url: str | None,
        rows: int,
    ) -> DatasetPreview:
        return DatasetPreview(train=[], validation=[])

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        return None


class _BlockingStorageBackend:
    """A backend whose persist blocks until released, to observe concurrency."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.concurrent = 0
        self.max_concurrent = 0

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        await self.release.wait()
        self.concurrent -= 1
        return None, None

    async def preview(
        self, *, dataset_id: UUID, name: str, data_format: str, artifact_url: str | None, rows: int
    ) -> DatasetPreview:
        return DatasetPreview(train=[], validation=[])

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        return None


async def _seed_dataset(factory: async_sessionmaker[AsyncSession]) -> UUID:
    async with factory() as session:
        repo = SqlAlchemyDatasetRepository(session)
        user = UserTable(id=uuid4(), email=f"{uuid4()}@example.com", role="user")
        session.add(user)
        await session.commit()
        dataset = await repo.create(
            user_id=str(user.id), name="ds", description="d", data_format="jsonl"
        )
        return dataset.id


def _stage_jsonl(staging: Path, dataset_id: UUID, rows: int) -> Path:
    directory = staging / str(dataset_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train.jsonl"
    path.write_text("\n".join(json.dumps({"i": n}) for n in range(rows)) + "\n")
    return path


def _stage_json_array(staging: Path, dataset_id: UUID, rows: list[dict[str, Any]]) -> Path:
    """Stage a pretty-printed JSON *array* under the `.jsonl` name the uploader uses.

    Mirrors a real `.json` upload: the frontend streams the raw file bytes and
    the service maps `.json` to the `jsonl` format, so a standard JSON array
    lands on disk as `train.jsonl` — the case that broke line-by-line parsing.
    """
    directory = staging / str(dataset_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train.jsonl"
    path.write_text(json.dumps(rows, indent=4))
    return path


async def test_process_marks_ready_and_records_counts(engine: AsyncEngine, tmp_path: Path) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    staging = tmp_path / ".staging"
    train = _stage_jsonl(staging, dataset_id, rows=3)
    runner = InProcessDatasetUploadRunner(
        session_factory=factory, storage=FakeStorageBackend(), staging_dir=staging
    )

    await runner.process(
        dataset_id,
        name="ds",
        data_format="jsonl",
        train=train,
        validation=None,
        validation_percentage=None,
        column_mapping=None,
    )

    async with factory() as session:
        refreshed = await SqlAlchemyDatasetRepository(session).get(dataset_id)
    assert refreshed is not None
    assert refreshed.status == DatasetStatus.READY
    assert refreshed.train_records == 3


async def test_process_normalizes_a_json_array_before_counting_and_remapping(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    # Reproduces the reported failure: a standard `.json` array uploaded with a
    # column mapping used to crash remap with `json.loads('[\n')` -> JSONDecodeError
    # and land the dataset in `error`. It must now normalize to JSONL and succeed.
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    staging = tmp_path / ".staging"
    train = _stage_json_array(
        staging,
        dataset_id,
        [{"instruction": "hi", "extra": "x"}, {"instruction": "bye", "extra": "y"}],
    )
    runner = InProcessDatasetUploadRunner(
        session_factory=factory, storage=FakeStorageBackend(), staging_dir=staging
    )

    await runner.process(
        dataset_id,
        name="ds",
        data_format="jsonl",
        train=train,
        validation=None,
        validation_percentage=None,
        column_mapping={"prompt": "instruction"},
    )

    async with factory() as session:
        refreshed = await SqlAlchemyDatasetRepository(session).get(dataset_id)
    assert refreshed is not None
    assert refreshed.status == DatasetStatus.READY
    assert refreshed.train_records == 2


async def test_process_marks_error_with_a_safe_detail_on_empty_split(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    staging = tmp_path / ".staging"
    train = _stage_jsonl(staging, dataset_id, rows=3)
    runner = InProcessDatasetUploadRunner(
        session_factory=factory, storage=FakeStorageBackend(), staging_dir=staging
    )

    await runner.process(
        dataset_id,
        name="ds",
        data_format="jsonl",
        train=train,
        validation=None,
        validation_percentage=0,
        column_mapping=None,
    )

    async with factory() as session:
        refreshed = await SqlAlchemyDatasetRepository(session).get(dataset_id)
    assert refreshed is not None
    assert refreshed.status == DatasetStatus.ERROR
    assert refreshed.status_detail is not None and "split" in refreshed.status_detail


async def test_process_cleans_up_staging(engine: AsyncEngine, tmp_path: Path) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    dataset_id = await _seed_dataset(factory)
    staging = tmp_path / ".staging"
    _stage_jsonl(staging, dataset_id, rows=2)
    runner = InProcessDatasetUploadRunner(
        session_factory=factory, storage=FakeStorageBackend(), staging_dir=staging
    )

    await runner.process(
        dataset_id,
        name="ds",
        data_format="jsonl",
        train=staging / str(dataset_id) / "train.jsonl",
        validation=None,
        validation_percentage=None,
        column_mapping=None,
    )

    assert not (staging / str(dataset_id)).exists()


async def test_noop_runner_records_the_submission() -> None:
    runner = NoOpDatasetUploadRunner()
    dataset_id = uuid4()

    await runner.submit(
        dataset_id,
        name="ds",
        data_format="jsonl",
        train=Path("t"),
        validation=None,
        validation_percentage=None,
        column_mapping=None,
    )

    assert runner.submitted == [dataset_id]


async def test_process_bounds_concurrency_to_the_configured_limit(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    staging = tmp_path / ".staging"
    ids = [await _seed_dataset(factory) for _ in range(3)]
    trains = [_stage_jsonl(staging, dsid, rows=2) for dsid in ids]
    storage = _BlockingStorageBackend()
    runner = InProcessDatasetUploadRunner(
        session_factory=factory, storage=storage, staging_dir=staging, max_concurrent=2
    )

    tasks = [
        asyncio.create_task(
            runner.process(
                dsid,
                name="ds",
                data_format="jsonl",
                train=train,
                validation=None,
                validation_percentage=None,
                column_mapping=None,
            )
        )
        for dsid, train in zip(ids, trains, strict=True)
    ]
    # Let the first batch reach the blocked persist, then release and drain.
    # Polling a plain counter (not an Event) is intentional: `concurrent` is
    # incremented by whichever tasks get scheduled first, and `sleep(0)` just
    # yields so they can run; there is no event to await instead.
    while storage.concurrent < 2:  # noqa: ASYNC110
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    storage.release.set()
    await asyncio.gather(*tasks)

    assert storage.max_concurrent == 2  # third upload waited for a slot


async def test_process_marks_error_when_processing_times_out(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    staging = tmp_path / ".staging"
    dataset_id = await _seed_dataset(factory)
    train = _stage_jsonl(staging, dataset_id, rows=2)

    class _SlowStorage(FakeStorageBackend):
        async def persist(
            self,
            *,
            dataset_id: UUID,
            name: str,
            data_format: str,
            train: Path,
            validation: Path | None,
        ) -> tuple[UUID | None, str | None]:
            await asyncio.sleep(1)
            return None, None

    runner = InProcessDatasetUploadRunner(
        session_factory=factory,
        storage=_SlowStorage(),
        staging_dir=staging,
        processing_timeout_seconds=0.05,
    )

    await runner.process(
        dataset_id,
        name="ds",
        data_format="jsonl",
        train=train,
        validation=None,
        validation_percentage=None,
        column_mapping=None,
    )

    async with factory() as session:
        refreshed = await SqlAlchemyDatasetRepository(session).get(dataset_id)
    assert refreshed is not None
    assert refreshed.status == DatasetStatus.ERROR
    assert "limit" in (refreshed.status_detail or "")


async def test_process_marks_error_when_disk_is_insufficient(
    engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    staging = tmp_path / ".staging"
    dataset_id = await _seed_dataset(factory)
    train = _stage_jsonl(staging, dataset_id, rows=2)
    runner = InProcessDatasetUploadRunner(
        session_factory=factory, storage=FakeStorageBackend(), staging_dir=staging
    )

    class _Usage:
        free = 0

    monkeypatch.setattr("autotunex.services.dataset_runner.shutil.disk_usage", lambda _p: _Usage())

    await runner.process(
        dataset_id,
        name="ds",
        data_format="jsonl",
        train=train,
        validation=None,
        validation_percentage=None,
        column_mapping=None,
    )

    async with factory() as session:
        refreshed = await SqlAlchemyDatasetRepository(session).get(dataset_id)
    assert refreshed is not None
    assert refreshed.status == DatasetStatus.ERROR
    assert "disk" in (refreshed.status_detail or "").lower()
