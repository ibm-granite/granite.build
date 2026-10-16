# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Add dataset HuggingFace provenance columns.

Revision ID: dd4511a46fc5
Revises: a3c71d94e5b2
Create Date: 2026-09-17 14:35:09.389778

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "dd4511a46fc5"
down_revision = "a3c71d94e5b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable HF provenance columns to ``datasets``.

    ``hf_repo_id``/``hf_revision``/``hf_config``/``hf_split`` are separate
    columns because they are what a query filters and dedupes on; everything
    display-only (applied mapping, original and retained row counts) lands in
    ``hf_provenance`` instead. All five are nullable — every existing row, and
    every dataset created by upload rather than HF import, carries no HF origin.

    Plain ``add_column``, not ``batch_alter_table``: ``datasets`` has two
    ``GENERATED ALWAYS AS (...) STORED`` columns (``train_file``,
    ``validation_file``), and SQLite batch mode's copy-and-recreate strategy
    cannot round-trip those — it either mis-renders the copied DDL or tries to
    ``INSERT`` into a generated column, both fatal. A plain ``ADD COLUMN``
    compiles to SQLite's native ``ALTER TABLE ... ADD COLUMN`` (support since
    3.35) instead of a table rebuild, so it does not touch the generated
    columns at all. Confirmed the same failure pre-exists on this table's
    earlier ``7f175ebf55ad`` batch-mode downgrade, independent of this change.
    """
    op.add_column("datasets", sa.Column("hf_repo_id", sa.String(length=255), nullable=True))
    op.add_column("datasets", sa.Column("hf_revision", sa.String(length=64), nullable=True))
    op.add_column("datasets", sa.Column("hf_config", sa.String(length=255), nullable=True))
    op.add_column("datasets", sa.Column("hf_split", sa.String(length=255), nullable=True))
    op.add_column("datasets", sa.Column("hf_provenance", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Drop the HF provenance columns, losing any recorded import origin.

    Plain ``drop_column`` for the same reason as ``upgrade`` — SQLite's native
    ``ALTER TABLE ... DROP COLUMN`` avoids the copy-and-recreate that batch
    mode would force on this table's generated columns.
    """
    for column in ("hf_provenance", "hf_split", "hf_config", "hf_revision", "hf_repo_id"):
        op.drop_column("datasets", column)
