# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
