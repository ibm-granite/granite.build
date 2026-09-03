"""The ORM mirrors ``resources/autotunex_schema.sql``.

The MySQL database is live, so fidelity to that file matters more than
tidiness — including its defects, which ``docs/schema-review.md`` catalogues but
this layer deliberately reproduces. These tests pin the properties that would
break the live database if someone "fixed" them.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    JobTable,
    ResultTable,
    TrialTable,
    UserTable,
)
from autotunex.models.status import DatasetStatus, RunStatus


def test_jobs_table_has_no_precision_column() -> None:
    """The one deliberate deviation from the schema file.

    ``precision`` was ``VARCHAR(50) NOT NULL`` with no default, so requiring it
    forced every writer to supply a value. Its values live in
    ``config_snapshot['precision']`` now.
    """
    assert "precision" not in inspect(JobTable).columns


def test_job_id_is_a_36_character_string_not_char_32() -> None:
    rendered = str(inspect(JobTable).columns["id"].type)

    assert "36" in rendered


def test_trial_id_is_a_short_string_not_a_uuid() -> None:
    """``trials.id`` is a short ``VARCHAR`` opaque code, not a UUID.

    Widened from ``VARCHAR(10)`` to ``VARCHAR(16)`` so Ray trial ids fit without
    lossy truncation (see the local/bash-runners design and migration
    ``0a2caef2a185``); the point of this test is that it is a short bounded
    string, not a 36-char UUID.
    """
    assert "16" in str(inspect(TrialTable).columns["id"].type)


async def test_status_is_stored_uppercase_and_read_back_as_the_enum(
    session: AsyncSession, job: JobTable
) -> None:
    stored = await session.scalar(select(JobTable.status).where(JobTable.id == job.id))

    assert stored is RunStatus.PENDING


async def test_status_is_persisted_as_the_uppercase_member_name(
    session: AsyncSession, job: JobTable
) -> None:
    """SQLAlchemy Enum stores ``.name``, which is what the MySQL ENUM declares."""
    raw = await session.execute(text("SELECT status FROM jobs WHERE id = :id"), {"id": str(job.id)})

    assert raw.scalar_one() == "PENDING"


async def test_dataset_train_file_is_generated_from_the_name(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="gen@example.com")
    session.add(user)
    await session.flush()
    dataset = DatasetTable(id=uuid4(), user_id=str(user.id), name="alpaca", description="d")
    session.add(dataset)
    await session.commit()

    refreshed = await session.get(DatasetTable, dataset.id, populate_existing=True)

    assert refreshed is not None
    assert refreshed.train_file == "alpaca_train"
    assert refreshed.validation_file == "alpaca_validation"


async def test_a_new_dataset_defaults_to_empty_status(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="ds-owner@example.com", role="user")
    session.add(user)
    await session.commit()
    dataset = DatasetTable(id=uuid4(), user_id=str(user.id), name="ds", description="desc")
    session.add(dataset)

    await session.commit()
    await session.refresh(dataset, ["status", "train_file", "validation_file"])

    assert dataset.status == DatasetStatus.EMPTY
    assert dataset.status_detail is None
    assert dataset.train_file == "ds_train"
    assert dataset.validation_file == "ds_validation"


async def test_dataset_user_relationship_reaches_the_owner(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="rel@example.com", role="user")
    session.add(user)
    await session.commit()
    dataset = DatasetTable(id=uuid4(), user_id=str(user.id), name="ds2", description="d")
    session.add(dataset)
    await session.commit()

    await session.refresh(dataset, ["user"])

    assert dataset.user.email == "rel@example.com"


async def test_deleting_a_user_cascades_to_their_jobs(session: AsyncSession, job: JobTable) -> None:
    """``ON DELETE CASCADE`` — the database removes the children, not the ORM.

    ``populate_existing=True`` is load-bearing. The cascade happens in SQLite, so
    the session knows nothing about it; with ``expire_on_commit=False`` a plain
    ``get`` would hand back the stale identity-map object and the assertion would
    be about the session cache rather than the database.
    """
    user = await session.get(UserTable, UUID(job.user_id))

    assert user is not None
    await session.delete(user)
    await session.commit()

    assert await session.get(JobTable, job.id, populate_existing=True) is None


async def test_deleting_a_referenced_configuration_is_rejected(
    session: AsyncSession, job: JobTable
) -> None:
    """``ON DELETE RESTRICT`` — a configuration in use cannot vanish."""
    configuration = await session.get(ConfigurationTable, job.config_id)

    assert configuration is not None
    await session.delete(configuration)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_a_trial_has_at_most_one_result(session: AsyncSession, job: JobTable) -> None:
    """``results.trial_id`` is UNIQUE, so the relationship is one-to-one."""
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    session.add(ResultTable(id=uuid4(), job_id=job.id, trial_id="t1", metric="eval_loss"))
    await session.commit()

    session.add(ResultTable(id=uuid4(), job_id=job.id, trial_id="t1", metric="eval_loss"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_metrics_json_round_trips_as_parsed_data(
    session: AsyncSession, job: JobTable
) -> None:
    """Compare parsed JSON — MySQL reorders object keys, SQLite does not."""
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    session.add(
        ResultTable(
            id=uuid4(),
            job_id=job.id,
            trial_id="t1",
            metric="eval_loss",
            metrics={"eval_loss": 0.42, "steps": 100},
        )
    )
    await session.commit()

    loaded = await session.scalar(select(ResultTable).where(ResultTable.trial_id == "t1"))

    assert loaded is not None
    assert loaded.metrics == {"eval_loss": 0.42, "steps": 100}
