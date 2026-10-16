"""Add training_metrics table.

Revision ID: f09bd54b61b7
Revises: 0a2caef2a185
Create Date: 2026-08-26 00:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

import autotunex.db.types

revision = "f09bd54b61b7"
down_revision = "0a2caef2a185"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the per-step training_metrics time-series table.

    Both indexes trail ``id``, not ``global_step``, because every read is the
    keyset page ``WHERE job_id = ? [AND trial_id = ?] AND id > ? ORDER BY id``
    (see ``SqlAlchemyTrainingMetricsRepository.metrics_page``). Trailing
    ``global_step`` instead would let the engine filter but still force a sort
    for the ordering — and on an append-only per-step table that grows without
    bound, the sort is the part that hurts. ``trial_id`` needs no index of its
    own: it is covered by the ``(job_id, trial_id, id)`` composite.
    """
    op.create_table(
        "training_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("trial_id", sa.String(length=16), nullable=True),
        sa.Column("global_step", sa.Integer(), nullable=False),
        sa.Column("epoch", sa.Float(), nullable=True),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("grad_norm", sa.Float(), nullable=True),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_metrics_job_keyset", "training_metrics", ["job_id", "id"])
    op.create_index(
        "ix_training_metrics_job_trial_keyset",
        "training_metrics",
        ["job_id", "trial_id", "id"],
    )


def downgrade() -> None:
    """Drop the training_metrics table (its indexes are dropped with it).

    Dropping the table removes its indexes and the ``job_id`` foreign key
    atomically, so there is no need to drop the FK-backing index first — which
    on MySQL would fail with error 1553 while the FK still exists.
    """
    op.drop_table("training_metrics")
