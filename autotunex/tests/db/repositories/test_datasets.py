"""SQLAlchemy dataset repository against a real in-memory database.

The two DB behaviours a fake cannot prove — the ``UNIQUE (user_id, name)``
collision and the ``ON DELETE RESTRICT`` from ``jobs.dataset_id`` — are pinned
here, plus the status/upload-result writes and the ``associated_jobs`` lookup.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.constants import SYSTEM_USER_ID
from autotunex.core.exceptions import DatasetInUseError, DatasetNameConflictError
from autotunex.db.repositories.sqlalchemy import SqlAlchemyDatasetRepository
from autotunex.db.tables import ConfigurationTable, JobTable, UserTable
from autotunex.models.status import DatasetStatus, RunStatus


async def _make_user(session: AsyncSession, *, email: str = "a@example.com") -> UserTable:
    user = UserTable(id=uuid4(), email=email, role="user")
    session.add(user)
    await session.commit()
    return user


async def test_create_persists_defaults_status_empty_and_generates_filenames(
    session: AsyncSession,
) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)

    created = await repository.create(
        user_id=str(user.id), name="alpaca", description="d", data_format="jsonl"
    )

    assert isinstance(created.id, UUID)
    assert created.status == DatasetStatus.EMPTY
    assert created.train_file == "alpaca_train"
    assert created.validation_file == "alpaca_validation"


async def test_create_with_a_duplicate_name_for_one_owner_conflicts(
    session: AsyncSession,
) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)
    await repository.create(user_id=str(user.id), name="dup", description="d", data_format="jsonl")

    with pytest.raises(DatasetNameConflictError):
        await repository.create(
            user_id=str(user.id), name="dup", description="d", data_format="jsonl"
        )


async def test_the_same_name_under_two_owners_is_allowed(session: AsyncSession) -> None:
    one = await _make_user(session, email="one@example.com")
    two = await _make_user(session, email="two@example.com")
    repository = SqlAlchemyDatasetRepository(session)
    await repository.create(
        user_id=str(one.id), name="shared", description="d", data_format="jsonl"
    )

    created = await repository.create(
        user_id=str(two.id), name="shared", description="d", data_format="jsonl"
    )

    assert created.name == "shared"


async def test_list_with_an_owner_id_filters_both_page_and_total(session: AsyncSession) -> None:
    mine = await _make_user(session, email="mine@example.com")
    theirs = await _make_user(session, email="theirs@example.com")
    repository = SqlAlchemyDatasetRepository(session)
    await repository.create(user_id=str(mine.id), name="a", description="d", data_format="jsonl")
    await repository.create(user_id=str(theirs.id), name="b", description="d", data_format="jsonl")

    datasets, total = await repository.list(limit=20, offset=0, owner_id=mine.id)

    assert total == 1
    assert [d.user_id for d in datasets] == [str(mine.id)]


async def test_list_filters_datasets_by_name_substring(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)
    await repository.create(
        user_id=str(user.id), name="squad-train", description="d", data_format="jsonl"
    )
    await repository.create(
        user_id=str(user.id), name="gsm8k", description="d", data_format="jsonl"
    )

    datasets, total = await repository.list(limit=20, offset=0, q="SQUAD")

    assert total == 1
    assert [d.name for d in datasets] == ["squad-train"]


async def test_update_replaces_fields_and_keeps_the_owner(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)
    created = await repository.create(
        user_id=str(user.id), name="before", description="d", data_format="jsonl"
    )

    updated = await repository.update(created.id, name="after", description="d2", data_format="csv")

    assert updated is not None
    assert updated.name == "after"
    assert updated.data_format == "csv"
    assert updated.user_id == str(user.id)


async def test_update_scoped_to_a_non_owner_returns_none(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)
    created = await repository.create(
        user_id=str(user.id), name="x", description="d", data_format="jsonl"
    )

    result = await repository.update(
        created.id, owner_id=uuid4(), name="y", description="d", data_format="jsonl"
    )

    assert result is None


async def test_set_status_and_set_upload_result_persist(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)
    created = await repository.create(
        user_id=str(user.id), name="x", description="d", data_format="jsonl"
    )

    await repository.set_status(created.id, DatasetStatus.UPLOADING)
    await repository.set_upload_result(
        created.id,
        train_records=100,
        train_file_size=2048,
        validation_records=10,
        validation_file_size=256,
        data_format="jsonl",
        artifact_id=None,
        artifact_url=None,
    )
    refreshed = await repository.get(created.id)

    assert refreshed is not None
    assert refreshed.status == DatasetStatus.READY
    assert refreshed.train_records == 100
    assert refreshed.validation_records == 10


async def test_set_status_error_records_a_detail(session: AsyncSession) -> None:
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)
    created = await repository.create(
        user_id=str(user.id), name="x", description="d", data_format="jsonl"
    )

    await repository.set_status(created.id, DatasetStatus.ERROR, status_detail="split was empty")
    refreshed = await repository.get(created.id)

    assert refreshed is not None
    assert refreshed.status == DatasetStatus.ERROR
    assert refreshed.status_detail == "split was empty"


async def test_delete_of_a_dataset_a_job_references_conflicts(session: AsyncSession) -> None:
    """ON DELETE RESTRICT from jobs.dataset_id, surfaced as a clean 409."""
    user = await _make_user(session)
    repository = SqlAlchemyDatasetRepository(session)
    dataset = await repository.create(
        user_id=str(user.id), name="ds", description="d", data_format="jsonl"
    )
    configuration = ConfigurationTable(
        id=uuid4(), user_id=str(user.id), name="cfg", config_data={"k": 1}
    )
    job = JobTable(
        id=uuid4(),
        user_id=str(user.id),
        status=RunStatus.PENDING,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="exp",
    )
    session.add_all([configuration, job])
    await session.commit()

    with pytest.raises(DatasetInUseError):
        await repository.delete(dataset.id)


async def test_jobs_for_dataset_returns_referencing_jobs_scoped_to_owner(
    session: AsyncSession,
) -> None:
    owner = await _make_user(session, email="owner@example.com")
    other = await _make_user(session, email="other@example.com")
    repository = SqlAlchemyDatasetRepository(session)
    dataset = await repository.create(
        user_id=str(owner.id), name="ds", description="d", data_format="jsonl"
    )
    configuration = ConfigurationTable(
        id=uuid4(), user_id=str(owner.id), name="cfg", config_data={"k": 1}
    )
    my_job = JobTable(
        id=uuid4(),
        user_id=str(owner.id),
        status=RunStatus.RUNNING,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="mine",
    )
    their_job = JobTable(
        id=uuid4(),
        user_id=str(other.id),
        status=RunStatus.RUNNING,
        config_id=configuration.id,
        dataset_id=dataset.id,
        model="m",
        model_source="huggingface",
        experiment_name="theirs",
    )
    session.add_all([configuration, my_job, their_job])
    await session.commit()

    scoped = await repository.jobs_for_dataset([dataset.id], owner_id=owner.id)
    unscoped = await repository.jobs_for_dataset([dataset.id])

    assert {j.experiment_name for j in scoped[dataset.id]} == {"mine"}
    assert {j.experiment_name for j in unscoped[dataset.id]} == {"mine", "theirs"}


async def _make_system_user(session: AsyncSession) -> UserTable:
    user = UserTable(id=SYSTEM_USER_ID, email="system@autotunex.local", role="user")
    session.add(user)
    await session.commit()
    return user


async def test_get_includes_a_system_owned_dataset_when_shared(session: AsyncSession) -> None:
    await _make_system_user(session)
    owner = await _make_user(session, email="owner@example.com")
    repository = SqlAlchemyDatasetRepository(session)
    system_dataset = await repository.create(
        user_id=str(SYSTEM_USER_ID), name="starter", description="d", data_format="jsonl"
    )

    found = await repository.get(system_dataset.id, owner_id=owner.id, include_shared=True)

    assert found is not None
    assert found.id == system_dataset.id


async def test_get_excludes_a_system_owned_dataset_by_default(session: AsyncSession) -> None:
    await _make_system_user(session)
    owner = await _make_user(session, email="owner@example.com")
    repository = SqlAlchemyDatasetRepository(session)
    system_dataset = await repository.create(
        user_id=str(SYSTEM_USER_ID), name="starter", description="d", data_format="jsonl"
    )

    found = await repository.get(system_dataset.id, owner_id=owner.id)

    assert found is None


async def test_list_includes_own_and_system_datasets_when_shared(session: AsyncSession) -> None:
    await _make_system_user(session)
    owner = await _make_user(session, email="owner@example.com")
    repository = SqlAlchemyDatasetRepository(session)
    await repository.create(
        user_id=str(owner.id), name="mine", description="d", data_format="jsonl"
    )
    await repository.create(
        user_id=str(SYSTEM_USER_ID), name="starter", description="d", data_format="jsonl"
    )

    rows, total = await repository.list(limit=20, offset=0, owner_id=owner.id, include_shared=True)

    assert {row.name for row in rows} == {"mine", "starter"}
    assert total == 2


async def test_list_excludes_system_datasets_by_default(session: AsyncSession) -> None:
    await _make_system_user(session)
    owner = await _make_user(session, email="owner@example.com")
    repository = SqlAlchemyDatasetRepository(session)
    await repository.create(
        user_id=str(owner.id), name="mine", description="d", data_format="jsonl"
    )
    await repository.create(
        user_id=str(SYSTEM_USER_ID), name="starter", description="d", data_format="jsonl"
    )

    rows, total = await repository.list(limit=20, offset=0, owner_id=owner.id)

    assert {row.name for row in rows} == {"mine"}
    assert total == 1
