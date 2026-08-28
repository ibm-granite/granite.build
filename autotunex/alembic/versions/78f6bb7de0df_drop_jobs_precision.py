"""Backfill ``jobs.precision`` into ``config_snapshot``, then drop it.

``precision`` was ``VARCHAR(50) NOT NULL`` with no server default, so dropping it
also removes a required field from every writer — which is why it was dropped at
all. The live database has values in the column, so they are preserved inside
``config_snapshot['precision']`` before it goes, and ``downgrade()`` puts them
back. Nothing is lost in either direction.

**This revision does run against production**, unlike the baseline it follows.

Revision ID: 78f6bb7de0df
Revises: 1fb645a87b48
Create Date: 2026-07-29 17:09:16.901860
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "78f6bb7de0df"
down_revision = "1fb645a87b48"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Move every ``precision`` value into ``config_snapshot``, then drop it."""
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        op.execute(
            sa.text(
                "UPDATE jobs SET config_snapshot = "
                "COALESCE(config_snapshot::jsonb, '{}'::jsonb) "
                "|| jsonb_build_object('precision', \"precision\") "
                'WHERE "precision" IS NOT NULL'
            )
        )
    else:
        # MySQL and SQLite share json_set semantics. The COALESCE is load-bearing:
        # json_set(NULL, ...) returns NULL on BOTH MySQL and SQLite, so without
        # the COALESCE every precision value is silently discarded for rows that
        # have no snapshot yet. Verified against SQLite 3.50.2 and MySQL 8.4.
        op.execute(
            sa.text(
                "UPDATE jobs SET config_snapshot = "
                "json_set(COALESCE(config_snapshot, JSON_OBJECT()), "
                "'$.precision', `precision`) "
                "WHERE `precision` IS NOT NULL"
                if bind.dialect.name == "mysql"
                else "UPDATE jobs SET config_snapshot = "
                "json_set(COALESCE(config_snapshot, JSON_OBJECT()), "
                "'$.precision', precision) "
                "WHERE precision IS NOT NULL"
            )
        )

    with op.batch_alter_table("jobs") as batch:
        batch.drop_column("precision")


def downgrade() -> None:
    """Re-add ``precision`` and restore values out of ``config_snapshot``.

    Re-added as nullable rather than ``NOT NULL``: a row whose snapshot never
    carried a precision has nothing to restore, and failing the downgrade on that
    would make the migration one-way in practice.
    """
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("precision", sa.String(50), nullable=True))

    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        op.execute(
            sa.text(
                "UPDATE jobs SET \"precision\" = config_snapshot::jsonb ->> 'precision' "
                "WHERE config_snapshot IS NOT NULL"
            )
        )
    elif bind.dialect.name == "mysql":
        # json_unquote is MySQL-only: without it, precision would come back as
        # the quoted JSON string '"bf16"' instead of the scalar bf16.
        op.execute(
            sa.text(
                "UPDATE jobs SET `precision` = "
                "json_unquote(json_extract(config_snapshot, '$.precision')) "
                "WHERE config_snapshot IS NOT NULL"
            )
        )
    else:
        # SQLite's json_extract already returns an unquoted scalar; json_unquote
        # does not exist here.
        op.execute(
            sa.text(
                "UPDATE jobs SET precision = "
                "json_extract(config_snapshot, '$.precision') "
                "WHERE config_snapshot IS NOT NULL"
            )
        )
