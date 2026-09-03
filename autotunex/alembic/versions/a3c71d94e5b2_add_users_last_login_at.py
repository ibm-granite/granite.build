# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Add users.last_login_at.

Revision ID: a3c71d94e5b2
Revises: f09bd54b61b7
Create Date: 2026-09-02 09:50:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from autotunex.db.types import UtcDateTime

revision = "a3c71d94e5b2"
down_revision = "f09bd54b61b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add a nullable ``last_login_at`` to ``users``.

    Backs the Users table's "Last login on" column, which until now read
    ``updated_at`` — a column that moves whenever the row changes, so an admin
    editing a role showed up as that user logging in. A dedicated column is
    written only on authentication, so nothing else can imitate a login.

    Nullable, and backfilled from ``updated_at`` — which really does hold the old
    login times, so this migrates existing data rather than inventing it. The
    pre-refactor service wrote logins into that column directly (``UPDATE users SET
    updated_at=NOW()`` on every OIDC callback), so on the live database it is the
    best and only record of when each user last signed in. Landing the column empty
    would discard all of it and show every existing user an unknown last login.

    The backfill is conditional on ``updated_at > created_at``, and that guard is
    the honest part. A row nothing ever wrote after creation has the two equal, and
    such a row has no login behind it — the tuning pipeline creates users it never
    authenticates as. Copying unconditionally would give every one of them a
    fabricated login equal to their creation time, which is exactly the
    ``created_at``-as-login claim this column exists to stop making. Those rows stay
    ``NULL``: "unknown", rendered as an em dash, corrected by the user's next request.

    The guard is imperfect in one direction and knowingly so: a user who logged in
    under the old code and *then* had their role changed carries the role-change
    time here, because that write moved ``updated_at`` too. It is still an upper
    bound on a real login, and it is the only evidence the database holds. Nothing
    after this revision has that ambiguity — ``last_login_at`` is written on
    authentication and by nothing else.

    ``updated_at`` is re-assigned its own value so it survives this migration. On
    MySQL the live schema declares it ``ON UPDATE CURRENT_TIMESTAMP``, so an
    ``UPDATE`` that did not mention it would silently rewrite every backfilled row's
    modification time to the moment the migration ran — losing that audit trail
    across the whole table. Explicitly assigning a column suppresses MySQL's
    automatic update; on SQLite and Postgres the assignment is a harmless no-op. The
    value read on the right-hand side is the pre-update one either way, so
    ``last_login_at`` still receives the original timestamp.

    Batch mode for SQLite, which cannot ``ALTER TABLE ADD COLUMN`` with every
    constraint form; harmless on MySQL and Postgres, where it is a plain
    ``ALTER``. ``UtcDateTime`` rather than ``sa.DateTime`` so the column matches
    the ORM's own type and the three-dialect migration matrix agrees with
    ``alembic check``; ``timezone=True`` is redundant (the decorator's ``impl``
    already carries it) and passed only to read the same as revision
    ``f09bd54b61b7`` next door.
    """
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("last_login_at", UtcDateTime(timezone=True), nullable=True))

    # Declared locally rather than imported from ``db/tables/`` on purpose: a
    # migration must keep describing the schema as it was at this revision, and an
    # ORM model that later gains or loses a column would silently change what this
    # historical UPDATE does. Runs after the batch context exits — on SQLite, batch
    # mode recreates the table, so an UPDATE inside it would target the copy.
    users = sa.table(
        "users",
        sa.column("created_at", UtcDateTime(timezone=True)),
        sa.column("updated_at", UtcDateTime(timezone=True)),
        sa.column("last_login_at", UtcDateTime(timezone=True)),
    )
    op.execute(
        users.update()
        .where(users.c.updated_at > users.c.created_at)
        .values(last_login_at=users.c.updated_at, updated_at=users.c.updated_at)
    )


def downgrade() -> None:
    """Drop ``last_login_at``, losing the recorded login times with it."""
    with op.batch_alter_table("users") as batch:
        batch.drop_column("last_login_at")
