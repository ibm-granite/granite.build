# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Add the server defaults the api-bridge's inserts rely on.

Revision ID: 5e1d2c7a9b40
Revises: dd4511a46fc5
Create Date: 2026-10-27 12:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "5e1d2c7a9b40"
down_revision = "dd4511a46fc5"
branch_labels = None
depends_on = None

_RUN_STATUS = sa.Enum(
    "PENDING", "RUNNING", "PAUSED", "TERMINATED", "ERROR", "COMPLETED", name="run_status"
)

# (table, column, existing type, existing nullable, server default). The defaults are
# the ones resources/autotunex_schema.sql and the api-bridge's tables.py declare.
_DEFAULTS = (
    ("gb_tasks", "status", _RUN_STATUS, False, "PENDING"),
    ("jobs", "status", _RUN_STATUS, False, "PENDING"),
    ("jobs", "seed", sa.Integer(), True, sa.text("42")),
    ("trials", "status", _RUN_STATUS, False, "PENDING"),
    ("users", "role", sa.String(length=50), True, "user"),
)


def upgrade() -> None:
    """Give each column the DDL default the raw schema file declares.

    The baseline created them with no server default, only the ORM's Python-side
    one, which the api-bridge's SQLAlchemy Core inserts never see. Its bootstrap
    omits ``gb_tasks.status`` and failed ``NOT NULL`` on a migrated database, and its
    ``insert_user`` omits ``users.role``, which stayed NULL. ``status`` stores the
    enum member name, which is what ``Enum(RunStatus)`` writes.
    """
    for table, column, type_, nullable, default in _DEFAULTS:
        with op.batch_alter_table(table) as batch:
            batch.alter_column(
                column, existing_type=type_, existing_nullable=nullable, server_default=default
            )


def downgrade() -> None:
    """Drop the defaults again.

    On a database adopted from ``resources/autotunex_schema.sql`` (stamped at the
    baseline), these defaults existed before this revision, and downgrading drops them
    anyway: the api-bridge's bootstrap still writes ``gb_tasks.status`` itself, but
    its ``insert_user`` then leaves ``users.role`` NULL. Re-run ``upgrade`` to restore
    them.
    """
    for table, column, type_, nullable, _default in reversed(_DEFAULTS):
        with op.batch_alter_table(table) as batch:
            batch.alter_column(
                column, existing_type=type_, existing_nullable=nullable, server_default=None
            )
