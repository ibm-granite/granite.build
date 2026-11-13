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
