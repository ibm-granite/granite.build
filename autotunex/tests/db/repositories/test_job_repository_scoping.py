"""owner_id scoping at the SQL layer.

``list`` must apply the same filter to its total-count statement as to the
page — an unfiltered total next to a filtered page is the same bug the
existing ``_total_statement`` docstring already warns about, with a new cause.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.models.status import RunStatus


async def _make_job(session: AsyncSession, *, owner_email: str) -> JobTable:
    """Persist a user, configuration, dataset and job in one go.

    ``datasets.description`` is ``NOT NULL`` (and ``train_file`` /
    ``validation_file`` are generated from ``name``), so a dataset built without
    a description fails at flush time rather than at construction — which reads
    as a repository bug when it is really a fixture one.

    The user is committed on its own, before its dependents. ``JobTable``,
    ``ConfigurationTable`` and ``DatasetTable`` all declare their ``user``
    relationship ``viewonly=True`` (the ``user_id`` type mismatch in item C1
    makes write-side management ambiguous), so the unit of work has no
    relationship to sort inserts by and falls back to table-name order —
    which puts ``configurations`` ahead of ``users`` and trips the real FK
    constraint. Flushing the owner first sidesteps that ordering quirk rather
    than fighting it.
    """
    user = UserTable(id=uuid4(), email=owner_email, role="user")
    session.add(user)
    await session.commit()

    configuration = ConfigurationTable(
        id=uuid4(),
        user_id=str(user.id),
        name="cfg",
        config_data={"lr": {"kind": "float", "low": 1e-6, "high": 1e-3}},
    )
    dataset = DatasetTable(
        id=uuid4(), user_id=str(user.id), name="ds", description="Scoping fixture."
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
    session.add_all([configuration, dataset, job])
    await session.commit()
    return job


async def test_get_with_no_owner_id_returns_any_job(session: AsyncSession) -> None:
    job = await _make_job(session, owner_email="a@example.com")
    repository = SqlAlchemyJobRepository(session)

    found = await repository.get(job.id)

    assert found is not None
    assert found.id == job.id


async def test_get_with_a_matching_owner_id_returns_the_job(session: AsyncSession) -> None:
    job = await _make_job(session, owner_email="a@example.com")
    repository = SqlAlchemyJobRepository(session)

    found = await repository.get(job.id, owner_id=UUID(job.user_id))

    assert found is not None


async def test_get_with_a_non_matching_owner_id_returns_none(session: AsyncSession) -> None:
    job = await _make_job(session, owner_email="a@example.com")
    repository = SqlAlchemyJobRepository(session)

    found = await repository.get(job.id, owner_id=uuid4())

    assert found is None


async def test_list_with_an_owner_id_filters_both_the_page_and_the_total(
    session: AsyncSession,
) -> None:
    mine = await _make_job(session, owner_email="a@example.com")
    await _make_job(session, owner_email="b@example.com")
    repository = SqlAlchemyJobRepository(session)

    rows, total = await repository.list(limit=20, offset=0, owner_id=UUID(mine.user_id))

    assert total == 1
    assert [job.id for job, _ in rows] == [mine.id]


async def test_list_with_no_owner_id_sees_every_job(session: AsyncSession) -> None:
    await _make_job(session, owner_email="a@example.com")
    await _make_job(session, owner_email="b@example.com")
    repository = SqlAlchemyJobRepository(session)

    rows, total = await repository.list(limit=20, offset=0)

    assert total == 2
    assert len(rows) == 2


async def test_delete_with_a_non_matching_owner_id_returns_false_and_keeps_the_row(
    session: AsyncSession,
) -> None:
    job = await _make_job(session, owner_email="a@example.com")
    repository = SqlAlchemyJobRepository(session)

    deleted = await repository.delete(job.id, owner_id=uuid4())

    assert deleted is False
    assert await repository.get(job.id) is not None


async def test_delete_with_the_matching_owner_id_removes_the_job(session: AsyncSession) -> None:
    job = await _make_job(session, owner_email="a@example.com")
    repository = SqlAlchemyJobRepository(session)

    deleted = await repository.delete(job.id, owner_id=UUID(job.user_id))

    assert deleted is True
    assert await repository.get(job.id) is None


async def test_is_visible_is_true_only_under_the_owning_scope(session: AsyncSession) -> None:
    job = await _make_job(session, owner_email="a@example.com")
    repository = SqlAlchemyJobRepository(session)
    owner_id = UUID(job.user_id)

    assert await repository.is_visible(job.id, owner_id=owner_id) is True
    assert await repository.is_visible(job.id, owner_id=uuid4()) is False
    assert await repository.is_visible(job.id, owner_id=None) is True
