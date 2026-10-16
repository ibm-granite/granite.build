"""Unit tests for DatasetService, isolated from the database and real storage.

Hand-written Protocol fakes (a type-level assertion catches drift). Covers
scoping (admin all / user own / unprovisioned empty+404+403-on-create), each
domain error, and the upload hand-off (status → uploading, runner submitted).
"""

from __future__ import annotations

import builtins
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from starlette.datastructures import Headers, UploadFile

from autotunex.core.exceptions import (
    CallerNotProvisionedError,
    DatasetNotFoundError,
    DatasetNotReadyError,
    DomainValidationError,
    EmptyDatasetError,
    InvalidDatasetFormatError,
    ScopeNotPermittedError,
    UnsupportedDatasetFormatError,
)
from autotunex.db.repositories.protocols import DatasetRepository
from autotunex.db.tables import DatasetTable, JobTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.dataset import DatasetCreate, DatasetPreview
from autotunex.models.status import DatasetStatus, RunStatus
from autotunex.services.dataset_runner import DatasetUploadRunner, NoOpDatasetUploadRunner
from autotunex.services.datasets import DatasetService
from autotunex.services.storage.base import StorageBackend
from tests.conftest import make_settings

ADMIN_ID = uuid4()
ADMIN = Principal(email="admin@example.com", provider="session", user_id=ADMIN_ID, is_admin=True)


class FakeDatasetRepository:
    """In-memory dataset store enforcing UNIQUE(user_id, name)."""

    def __init__(self) -> None:
        self.datasets: dict[UUID, DatasetTable] = {}
        self.jobs: dict[UUID, list[JobTable]] = {}

    def seed(
        self,
        *,
        owner_id: str,
        name: str = "ds",
        status: DatasetStatus = DatasetStatus.EMPTY,
        artifact_url: str | None = None,
    ) -> DatasetTable:
        now = datetime.now(UTC)
        dataset = DatasetTable(
            id=uuid4(),
            user_id=owner_id,
            name=name,
            description="d",
            data_format="jsonl",
            status=status,
            status_detail=None,
            artifact_url=artifact_url,
            train_file=f"{name}_train",
            validation_file=f"{name}_validation",
            created_at=now,
            updated_at=now,
        )
        self.datasets[dataset.id] = dataset
        return dataset

    def _name_taken(self, *, user_id: str, name: str, excluding: UUID | None) -> bool:
        return any(
            d.id != excluding and d.user_id == user_id and d.name == name
            for d in self.datasets.values()
        )

    async def get(
        self,
        dataset_id: UUID,
        *,
        owner_id: UUID | None = None,
        include_shared: bool = False,
    ) -> DatasetTable | None:
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            return None
        if owner_id is not None and dataset.user_id != str(owner_id):
            return None
        return dataset

    async def list(
        self,
        *,
        limit: int,
        offset: int,
        owner_id: UUID | None = None,
        q: str | None = None,
        include_shared: bool = False,
    ) -> tuple[Sequence[DatasetTable], int]:
        matching = [
            d for d in self.datasets.values() if owner_id is None or d.user_id == str(owner_id)
        ]
        if q:
            needle = q.lower()
            matching = [d for d in matching if needle in d.name.lower()]
        ordered = sorted(matching, key=lambda d: d.created_at, reverse=True)
        return ordered[offset : offset + limit], len(ordered)

    async def create(
        self, *, user_id: str, name: str, description: str | None, data_format: str
    ) -> DatasetTable:
        from autotunex.core.exceptions import DatasetNameConflictError

        if self._name_taken(user_id=user_id, name=name, excluding=None):
            raise DatasetNameConflictError(name)
        now = datetime.now(UTC)
        dataset = DatasetTable(
            id=uuid4(),
            user_id=user_id,
            name=name,
            description=description,
            data_format=data_format,
            status=DatasetStatus.EMPTY,
            status_detail=None,
            train_file=f"{name}_train",
            validation_file=f"{name}_validation",
            created_at=now,
            updated_at=now,
        )
        self.datasets[dataset.id] = dataset
        return dataset

    async def update(
        self,
        dataset_id: UUID,
        *,
        owner_id: UUID | None = None,
        name: str,
        description: str | None,
        data_format: str,
    ) -> DatasetTable | None:
        dataset = await self.get(dataset_id, owner_id=owner_id)
        if dataset is None:
            return None
        dataset.name, dataset.description, dataset.data_format = name, description, data_format
        return dataset

    async def delete(self, dataset_id: UUID, *, owner_id: UUID | None = None) -> bool:
        dataset = await self.get(dataset_id, owner_id=owner_id)
        if dataset is None:
            return False
        del self.datasets[dataset_id]
        return True

    async def set_status(
        self, dataset_id: UUID, status: DatasetStatus, *, status_detail: str | None = None
    ) -> None:
        dataset = self.datasets.get(dataset_id)
        if dataset is not None:
            dataset.status = status
            dataset.status_detail = status_detail

    async def set_upload_result(
        self,
        dataset_id: UUID,
        *,
        train_records: int,
        train_file_size: int,
        validation_records: int | None,
        validation_file_size: int | None,
        data_format: str,
        artifact_id: UUID | None,
        artifact_url: str | None,
        status: DatasetStatus = DatasetStatus.READY,
    ) -> None:
        dataset = self.datasets.get(dataset_id)
        if dataset is not None:
            dataset.status = status

    async def jobs_for_dataset(
        self, dataset_ids: Sequence[UUID], *, owner_id: UUID | None = None
    ) -> dict[UUID, builtins.list[JobTable]]:
        result: dict[UUID, builtins.list[JobTable]] = {}
        for dataset_id in dataset_ids:
            jobs = self.jobs.get(dataset_id, [])
            if owner_id is not None:
                jobs = [j for j in jobs if j.user_id == str(owner_id)]
            if jobs:
                result[dataset_id] = jobs
        return result


class FakeStorageBackend:
    def __init__(self) -> None:
        self.preview_result = DatasetPreview(train=[{"a": 1}], validation=[])
        self.deleted: list[UUID] = []

    async def persist(
        self,
        *,
        dataset_id: UUID,
        name: str,
        data_format: str,
        train: Path,
        validation: Path | None,
    ) -> tuple[UUID | None, str | None]:
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
        return self.preview_result

    async def delete(self, *, dataset_id: UUID, name: str, artifact_url: str | None) -> None:
        self.deleted.append(dataset_id)


def _service(
    repository: FakeDatasetRepository,
    *,
    principal: Principal = ADMIN,
    storage: StorageBackend | None = None,
    runner: DatasetUploadRunner | None = None,
    storage_dir: Path | None = None,
) -> DatasetService:
    return DatasetService(
        repository=repository,
        principal=principal,
        storage=storage or FakeStorageBackend(),
        runner=runner or NoOpDatasetUploadRunner(),
        settings=make_settings(dataset_storage_dir=storage_dir),
    )


def _upload_file(content: bytes, filename: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": "application/octet-stream"}),
    )


def _body(*, name: str = "ds", data_format: str = "jsonl") -> DatasetCreate:
    return DatasetCreate(name=name, description="desc", data_format=data_format)


def test_doubles_satisfy_their_protocols() -> None:
    repository: DatasetRepository = FakeDatasetRepository()
    storage: StorageBackend = FakeStorageBackend()

    assert repository is not None and storage is not None


# Create / scoping.


async def test_create_stores_a_dataset_owned_by_the_caller() -> None:
    repository = FakeDatasetRepository()

    created = await _service(repository).create(_body(name="ds"))

    assert created.name == "ds"
    assert created.user_id == str(ADMIN_ID)
    assert created.status is DatasetStatus.EMPTY


async def test_create_rejects_an_unknown_format() -> None:
    repository = FakeDatasetRepository()

    with pytest.raises(InvalidDatasetFormatError):
        await _service(repository).create(_body(data_format="xml"))


async def test_create_is_refused_for_an_unprovisioned_caller() -> None:
    repository = FakeDatasetRepository()
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )

    with pytest.raises(CallerNotProvisionedError):
        await _service(repository, principal=principal).create(_body())


async def test_unprovisioned_caller_sees_an_empty_page() -> None:
    repository = FakeDatasetRepository()
    repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )

    page = await _service(repository, principal=principal).list(limit=20, offset=0)

    assert page.total == 0 and page.items == []


async def test_unprovisioned_caller_gets_not_found() -> None:
    repository = FakeDatasetRepository()
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="ghost@example.com", provider="session", user_id=None, is_admin=False
    )

    with pytest.raises(DatasetNotFoundError):
        await _service(repository, principal=principal).get(existing.id)


async def test_provisioned_user_sees_only_their_own() -> None:
    repository = FakeDatasetRepository()
    owner = uuid4()
    mine = repository.seed(owner_id=str(owner), name="mine")
    repository.seed(owner_id=str(uuid4()), name="theirs")
    principal = Principal(email="u@example.com", provider="session", user_id=owner, is_admin=False)

    page = await _service(repository, principal=principal).list(limit=20, offset=0)

    assert [d.id for d in page.items] == [mine.id]


async def test_get_of_another_users_dataset_is_not_found() -> None:
    repository = FakeDatasetRepository()
    other = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )

    with pytest.raises(DatasetNotFoundError):
        await _service(repository, principal=principal).get(other.id)


async def test_get_includes_caller_scoped_associated_jobs() -> None:
    repository = FakeDatasetRepository()
    dataset = repository.seed(owner_id=str(ADMIN_ID))
    repository.jobs[dataset.id] = [
        JobTable(
            id=uuid4(),
            user_id=str(ADMIN_ID),
            status=RunStatus.RUNNING,
            config_id=uuid4(),
            dataset_id=dataset.id,
            model="m",
            model_source="huggingface",
            experiment_name="exp",
        ),
    ]

    read = await _service(repository).get(dataset.id)

    assert [j.experiment_name for j in read.associated_jobs] == ["exp"]


async def test_get_scopes_associated_jobs_to_a_non_admin_caller() -> None:
    repository = FakeDatasetRepository()
    owner = uuid4()
    dataset = repository.seed(owner_id=str(owner))
    repository.jobs[dataset.id] = [
        JobTable(
            id=uuid4(),
            user_id=str(owner),
            status=RunStatus.RUNNING,
            config_id=uuid4(),
            dataset_id=dataset.id,
            model="m",
            model_source="huggingface",
            experiment_name="mine",
        ),
        JobTable(
            id=uuid4(),
            user_id=str(uuid4()),
            status=RunStatus.RUNNING,
            config_id=uuid4(),
            dataset_id=dataset.id,
            model="m",
            model_source="huggingface",
            experiment_name="theirs",
        ),
    ]
    principal = Principal(email="u@example.com", provider="session", user_id=owner, is_admin=False)

    read = await _service(repository, principal=principal).get(dataset.id)

    assert [j.experiment_name for j in read.associated_jobs] == ["mine"]


# Scope: every caller — admin included — sees only their own rows by default;
# an admin widens to all rows with scope=all; a non-admin asking for all is 403.


async def test_admin_sees_only_their_own_datasets_by_default() -> None:
    repository = FakeDatasetRepository()
    repository.seed(owner_id=str(uuid4()))  # someone else's, not the admin's

    page = await _service(repository).list(limit=20, offset=0)

    assert page.total == 0


async def test_admin_lists_all_datasets_with_scope_all() -> None:
    repository = FakeDatasetRepository()
    other = repository.seed(owner_id=str(uuid4()))

    page = await _service(repository).list(limit=20, offset=0, scope=DataScope.ALL)

    assert page.total == 1
    assert page.items[0].id == other.id


async def test_admin_cannot_get_a_foreign_dataset_by_default() -> None:
    repository = FakeDatasetRepository()
    other = repository.seed(owner_id=str(uuid4()))

    with pytest.raises(DatasetNotFoundError):
        await _service(repository).get(other.id)


async def test_admin_can_get_a_foreign_dataset_with_scope_all() -> None:
    repository = FakeDatasetRepository()
    other = repository.seed(owner_id=str(uuid4()))

    read = await _service(repository).get(other.id, scope=DataScope.ALL)

    assert read.id == other.id


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_list() -> None:
    repository = FakeDatasetRepository()
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )
    service = _service(repository, principal=principal)

    with pytest.raises(ScopeNotPermittedError):
        await service.list(limit=20, offset=0, scope=DataScope.ALL)


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_get() -> None:
    repository = FakeDatasetRepository()
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )

    with pytest.raises(ScopeNotPermittedError):
        await _service(repository, principal=principal).get(existing.id, scope=DataScope.ALL)


# Update / delete.


async def test_update_replaces_metadata() -> None:
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID), name="before")

    updated = await _service(repository).update(
        seeded.id, DatasetCreate(name="after", description="d2", data_format="csv")
    )

    assert updated.name == "after" and updated.data_format == "csv"


async def test_admin_cannot_update_a_foreign_dataset_by_default() -> None:
    repository = FakeDatasetRepository()
    other = repository.seed(owner_id=str(uuid4()), name="theirs")

    with pytest.raises(DatasetNotFoundError):
        await _service(repository).update(
            other.id, DatasetCreate(name="renamed", description="d", data_format="csv")
        )


async def test_admin_can_update_a_foreign_dataset_with_scope_all() -> None:
    repository = FakeDatasetRepository()
    other = repository.seed(owner_id=str(uuid4()), name="theirs")

    updated = await _service(repository).update(
        other.id,
        DatasetCreate(name="renamed", description="d", data_format="csv"),
        scope=DataScope.ALL,
    )

    assert updated.name == "renamed"


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_update() -> None:
    repository = FakeDatasetRepository()
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )

    with pytest.raises(ScopeNotPermittedError):
        await _service(repository, principal=principal).update(
            existing.id,
            DatasetCreate(name="x", description="y", data_format="csv"),
            scope=DataScope.ALL,
        )


async def test_delete_removes_and_calls_storage() -> None:
    repository = FakeDatasetRepository()
    storage = FakeStorageBackend()
    seeded = repository.seed(owner_id=str(ADMIN_ID))

    await _service(repository, storage=storage).delete(seeded.id)

    assert seeded.id not in repository.datasets
    assert storage.deleted == [seeded.id]


async def test_admin_cannot_delete_a_foreign_dataset_by_default() -> None:
    repository = FakeDatasetRepository()
    other = repository.seed(owner_id=str(uuid4()))

    with pytest.raises(DatasetNotFoundError):
        await _service(repository).delete(other.id)

    assert other.id in repository.datasets


async def test_admin_can_delete_a_foreign_dataset_with_scope_all() -> None:
    repository = FakeDatasetRepository()
    storage = FakeStorageBackend()
    other = repository.seed(owner_id=str(uuid4()))

    await _service(repository, storage=storage).delete(other.id, scope=DataScope.ALL)

    assert other.id not in repository.datasets


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_delete() -> None:
    repository = FakeDatasetRepository()
    existing = repository.seed(owner_id=str(uuid4()))
    principal = Principal(
        email="u@example.com", provider="session", user_id=uuid4(), is_admin=False
    )

    with pytest.raises(ScopeNotPermittedError):
        await _service(repository, principal=principal).delete(existing.id, scope=DataScope.ALL)


# Upload.


async def test_upload_streams_sets_uploading_and_hands_off(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    runner = NoOpDatasetUploadRunner()
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    service = _service(repository, runner=runner, storage_dir=tmp_path / "store")

    read = await service.upload(
        seeded.id,
        train=_upload_file(json.dumps({"a": 1}).encode() + b"\n", "train.jsonl"),
        validation=None,
        validation_percentage=None,
        column_mapping_json=None,
        gzip_encoded=False,
    )

    assert read.status is DatasetStatus.UPLOADING
    assert runner.submitted == [seeded.id]


async def test_upload_while_uploading_is_rejected(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID), status=DatasetStatus.UPLOADING)
    service = _service(repository, storage_dir=tmp_path / "store")

    with pytest.raises(DatasetNotReadyError):
        await service.upload(
            seeded.id,
            train=_upload_file(b"{}\n", "train.jsonl"),
            validation=None,
            validation_percentage=None,
            column_mapping_json=None,
            gzip_encoded=False,
        )


async def test_upload_of_an_empty_file_is_422(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    service = _service(repository, storage_dir=tmp_path / "store")

    with pytest.raises(EmptyDatasetError):
        await service.upload(
            seeded.id,
            train=_upload_file(b"", "train.jsonl"),
            validation=None,
            validation_percentage=None,
            column_mapping_json=None,
            gzip_encoded=False,
        )


async def test_upload_of_an_empty_file_leaves_no_staging_directory(tmp_path: Path) -> None:
    """An early raise (before the runner hand-off) must not orphan staging.

    ``EmptyDatasetError`` fires only after the (empty) train file has already
    been streamed to `staging/<dataset_id>/`. Nothing else ever reclaims that
    directory — the dataset row never transitions to `uploading` — so `upload`
    itself must clean it up before re-raising.
    """
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    storage_dir = tmp_path / "store"
    service = _service(repository, storage_dir=storage_dir)

    with pytest.raises(EmptyDatasetError):
        await service.upload(
            seeded.id,
            train=_upload_file(b"", "train.jsonl"),
            validation=None,
            validation_percentage=None,
            column_mapping_json=None,
            gzip_encoded=False,
        )

    staging = make_settings(dataset_storage_dir=storage_dir).dataset_staging_dir / str(seeded.id)
    assert not staging.exists()


async def test_upload_of_an_unsupported_extension_is_415(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    service = _service(repository, storage_dir=tmp_path / "store")

    with pytest.raises(UnsupportedDatasetFormatError):
        await service.upload(
            seeded.id,
            train=_upload_file(b"x", "train.txt"),
            validation=None,
            validation_percentage=None,
            column_mapping_json=None,
            gzip_encoded=False,
        )


async def test_upload_with_a_validation_file_format_mismatch_is_422(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    service = _service(repository, storage_dir=tmp_path / "store")

    with pytest.raises(DomainValidationError):
        await service.upload(
            seeded.id,
            train=_upload_file(b"{}\n", "train.jsonl"),
            validation=_upload_file(b"a,b\n1,2\n", "validation.csv"),
            validation_percentage=None,
            column_mapping_json=None,
            gzip_encoded=False,
        )


async def test_upload_with_both_validation_file_and_percentage_is_422(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    service = _service(repository, storage_dir=tmp_path / "store")

    with pytest.raises(DomainValidationError):
        await service.upload(
            seeded.id,
            train=_upload_file(b"{}\n", "train.jsonl"),
            validation=_upload_file(b"{}\n", "val.jsonl"),
            validation_percentage=20,
            column_mapping_json=None,
            gzip_encoded=False,
        )


async def test_upload_with_bad_column_mapping_json_is_422(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    seeded = repository.seed(owner_id=str(ADMIN_ID))
    service = _service(repository, storage_dir=tmp_path / "store")

    with pytest.raises(DomainValidationError):
        await service.upload(
            seeded.id,
            train=_upload_file(b"{}\n", "train.jsonl"),
            validation=None,
            validation_percentage=None,
            column_mapping_json="{not json",
            gzip_encoded=False,
        )


# Preview.


async def test_preview_is_read_only_for_ready_datasets(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    ready = repository.seed(owner_id=str(ADMIN_ID), status=DatasetStatus.READY)
    empty = repository.seed(owner_id=str(ADMIN_ID), name="empty", status=DatasetStatus.EMPTY)
    service = _service(repository)

    ready_read = await service.get(ready.id, preview=True, preview_rows=5)
    empty_read = await service.get(empty.id, preview=True, preview_rows=5)

    assert ready_read.preview is not None and ready_read.preview.train == [{"a": 1}]
    assert empty_read.preview is None


async def test_preview_is_read_for_artifact_backed_dataset_not_yet_ready(tmp_path: Path) -> None:
    # Datasets registered out-of-band by the tuning pipeline (via api-bridge) carry
    # a live ``artifact_url`` but were never flipped to ``ready`` — the pipeline's
    # write path does not set status. The preview gate must still read them, or the
    # UX shows "Unable to load dataset" for a dataset that plainly has data.
    repository = FakeDatasetRepository()
    registered = repository.seed(
        owner_id=str(ADMIN_ID),
        status=DatasetStatus.EMPTY,
        artifact_url="hf://huggingface.co/datasets/org/ds_abcd1234",
    )
    service = _service(repository)

    read = await service.get(registered.id, preview=True, preview_rows=5)

    assert read.preview is not None and read.preview.train == [{"a": 1}]


async def test_preview_failure_degrades_to_none(tmp_path: Path) -> None:
    repository = FakeDatasetRepository()
    ready = repository.seed(owner_id=str(ADMIN_ID), status=DatasetStatus.READY)

    class ExplodingStorage(FakeStorageBackend):
        async def preview(
            self,
            *,
            dataset_id: UUID,
            name: str,
            data_format: str,
            artifact_url: str | None,
            rows: int,
        ) -> DatasetPreview:
            raise RuntimeError("backend down")

    read = await _service(repository, storage=ExplodingStorage()).get(
        ready.id, preview=True, preview_rows=5
    )

    assert read.preview is None
