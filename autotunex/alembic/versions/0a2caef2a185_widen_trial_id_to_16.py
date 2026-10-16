"""Widen ``trials.id`` and its FK ``results.trial_id`` to ``VARCHAR(16)``.

Ray's trial ids are longer than the original ``VARCHAR(10)``, which would force
lossy truncation (and risk collisions). Both columns are widened together and
kept type-matched, since ``results.trial_id`` is a foreign key onto ``trials.id``.
``log_entries.trial_id`` is a separate ``CHAR(36)`` column with no FK and is left
untouched.

Written in batch mode so it applies on SQLite too (SQLite lacks
``ALTER COLUMN TYPE`` and recreates the table). The ``results -> trials`` FK
dictates the recreate order: widen ``trials.id`` first on upgrade, and reverse
that order on downgrade.

Revision ID: 0a2caef2a185
Revises: c628b830e8a3
Create Date: 2026-08-11 14:35:19.127832
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0a2caef2a185"
down_revision = "c628b830e8a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Widen both trial-id columns to ``VARCHAR(16)`` (parent before child)."""
    with op.batch_alter_table("trials") as batch:
        batch.alter_column(
            "id",
            type_=sa.String(length=16),
            existing_type=sa.String(length=10),
            existing_nullable=False,
        )
    with op.batch_alter_table("results") as batch:
        batch.alter_column(
            "trial_id",
            type_=sa.String(length=16),
            existing_type=sa.String(length=10),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Narrow both trial-id columns back to ``VARCHAR(10)`` (child before parent)."""
    with op.batch_alter_table("results") as batch:
        batch.alter_column(
            "trial_id",
            type_=sa.String(length=10),
            existing_type=sa.String(length=16),
            existing_nullable=False,
        )
    with op.batch_alter_table("trials") as batch:
        batch.alter_column(
            "id",
            type_=sa.String(length=10),
            existing_type=sa.String(length=16),
            existing_nullable=False,
        )
