"""Add jobs reward function columns.

Revision ID: c628b830e8a3
Revises: b27a008ed0cf
Create Date: 2026-08-06 00:10:49.289566

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c628b830e8a3"
down_revision = "b27a008ed0cf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable reward-function columns to ``jobs``.

    A job-level input (not part of ``config_snapshot``, which freezes the
    configuration). Nullable because only online-RL jobs carry one.
    """
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("reward_function_code", sa.Text(), nullable=True))
        batch.add_column(sa.Column("reward_function_name", sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Drop the reward-function columns."""
    with op.batch_alter_table("jobs") as batch:
        batch.drop_column("reward_function_name")
        batch.drop_column("reward_function_code")
