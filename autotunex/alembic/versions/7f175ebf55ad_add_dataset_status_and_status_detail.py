"""Add ``datasets.status`` and ``datasets.status_detail``.

Phase 1 dataset upload introduces a status lifecycle
(``empty|uploading|ready|error``) and a safe human-readable ``status_detail``.
``status`` is ``NOT NULL DEFAULT 'empty'`` so existing rows (written before the
API had a write path) become ``empty``, which is correct: they carry no uploaded
file this service produced. Batch mode keeps the column adds portable to SQLite.

Revision ID: 7f175ebf55ad
Revises: 78f6bb7de0df
Create Date: 2026-08-04 12:43:33.565395
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "7f175ebf55ad"
down_revision = "78f6bb7de0df"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the two columns; ``status`` defaults existing rows to ``empty``."""
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(
            sa.Column(
                "status",
                sa.String(length=20),
                nullable=False,
                server_default="empty",
            )
        )
        batch.add_column(sa.Column("status_detail", sa.Text(), nullable=True))


def downgrade() -> None:
    """Drop the two columns."""
    with op.batch_alter_table("datasets") as batch:
        batch.drop_column("status_detail")
        batch.drop_column("status")
