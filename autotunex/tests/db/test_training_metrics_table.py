"""Table-level tests for ``training_metrics``."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.db.base import Base
from autotunex.db.tables import JobTable, TrainingMetricTable, UserTable


async def test_training_metrics_row_roundtrips(
    session: AsyncSession, user: UserTable, job: JobTable
) -> None:
    session.add(
        TrainingMetricTable(
            job_id=job.id,
            trial_id="t1",
            global_step=10,
            epoch=0.04,
            loss=15.3477,
            grad_norm=2.85,
            learning_rate=6.5e-07,
            split="train",
            extra={},
        )
    )
    await session.commit()

    row = (await session.execute(select(TrainingMetricTable))).scalar_one()

    assert row.loss == 15.3477
    assert row.trial_id == "t1"
    assert row.created_at is not None


def test_the_indexes_trail_id_so_the_keyset_page_needs_no_sort() -> None:
    # Guards the keyset indexing in revision f09bd54b61b7: every read is
    # `WHERE job_id = ? [AND trial_id = ?] AND id > ? ORDER BY id`, so an index
    # trailing global_step would leave the ordering to a sort.
    # Read through the metadata rather than ``__table__``, which is typed as the
    # generic ``FromClause`` and so exposes no ``.indexes`` to mypy.
    table = Base.metadata.tables[TrainingMetricTable.__tablename__]
    indexes = {idx.name: [c.name for c in idx.columns] for idx in table.indexes}

    assert indexes == {
        "ix_training_metrics_job_keyset": ["job_id", "id"],
        "ix_training_metrics_job_trial_keyset": ["job_id", "trial_id", "id"],
    }
