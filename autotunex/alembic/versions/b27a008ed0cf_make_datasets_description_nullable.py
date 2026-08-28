"""Make ``datasets.description`` nullable.

The description is optional metadata, but the column was ``NOT NULL``, forcing
every caller to supply one. Relaxing it to nullable lets a dataset be created
without a description (stored as ``NULL``). ``NOT NULL`` -> ``NULL`` is a
non-destructive change on the live MySQL database, so this revision runs
normally there (it is not part of the stamped baseline).

Postgres and MySQL do this as a plain native ``ALTER`` (no table rewrite).
SQLite cannot ``ALTER`` a column's nullability, so it needs batch mode to
recreate the table — but ``datasets`` has generated
``train_file``/``validation_file`` columns and SQLite forbids writing those in
the table-copy ``INSERT`` (Alembic 1.18 copies every column). On SQLite we
therefore drop and re-add the generated columns inside the batch: dropping
excludes them from the data copy, and re-adding recreates them from their
expression. That expression mirrors ``db/tables/datasets.py`` so it renders
correctly on every dialect.

Revision ID: b27a008ed0cf
Revises: 7f175ebf55ad
Create Date: 2026-08-05 14:47:41.198451
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

import autotunex.db.types

revision = "b27a008ed0cf"
down_revision = "7f175ebf55ad"
branch_labels = None
depends_on = None


def _generated(suffix: str) -> sa.Computed:
    """The schema's dialect-portable ``CONCAT(name, suffix)`` generated column."""
    return sa.Computed(
        sa.column("name", sa.String).concat(sa.literal_column(f"'{suffix}'")),
        persisted=True,
    )


def _datasets_table(*, description_nullable: bool) -> sa.Table:
    """The ``datasets`` table as it exists at this revision (post-7f175ebf55ad).

    Used as ``copy_from`` so SQLite's batch recreate knows the true shape —
    including which columns are generated — rather than guessing from reflection.
    """
    return sa.Table(
        "datasets",
        sa.MetaData(),
        sa.Column("id", autotunex.db.types.Uuid36(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=description_nullable),
        sa.Column("train_file", sa.String(length=255), _generated("_train"), nullable=False),
        sa.Column("train_records", sa.Integer(), nullable=True),
        sa.Column("train_file_size", sa.Integer(), nullable=True),
        sa.Column(
            "validation_file", sa.String(length=255), _generated("_validation"), nullable=False
        ),
        sa.Column("validation_records", sa.Integer(), nullable=True),
        sa.Column("validation_file_size", sa.Integer(), nullable=True),
        sa.Column("data_format", sa.String(length=10), server_default="jsonl", nullable=False),
        sa.Column("artifact_id", autotunex.db.types.Uuid36(length=36), nullable=True),
        sa.Column("artifact_url", sa.Text(), nullable=True),
        sa.Column("created_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", autotunex.db.types.UtcDateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="empty", nullable=False),
        sa.Column("status_detail", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_datasets_user_name"),
    )


def _set_description_nullable(*, nullable: bool) -> None:
    """Flip ``datasets.description`` nullability, portably across dialects."""
    if op.get_bind().dialect.name != "sqlite":
        # Postgres/MySQL support this as a native ALTER — no table rewrite.
        op.alter_column("datasets", "description", existing_type=sa.Text(), nullable=nullable)
        return

    # SQLite: recreate the table. Drop and re-add the generated columns so they
    # are excluded from the data-copy INSERT (SQLite rejects writes to them),
    # then recreated from their expression.
    with op.batch_alter_table(
        "datasets", copy_from=_datasets_table(description_nullable=not nullable)
    ) as batch:
        batch.alter_column("description", existing_type=sa.Text(), nullable=nullable)
        batch.drop_column("train_file")
        batch.drop_column("validation_file")
        batch.add_column(
            sa.Column("train_file", sa.String(length=255), _generated("_train"), nullable=False)
        )
        batch.add_column(
            sa.Column(
                "validation_file",
                sa.String(length=255),
                _generated("_validation"),
                nullable=False,
            )
        )


def upgrade() -> None:
    """Relax ``datasets.description`` to nullable."""
    _set_description_nullable(nullable=True)


def downgrade() -> None:
    """Restore the ``NOT NULL`` constraint on ``datasets.description``."""
    _set_description_nullable(nullable=False)
