"""The precision drop must not lose data.

``jobs.precision`` was ``VARCHAR(50) NOT NULL`` and the live database has values
in it, so the migration backfills ``config_snapshot['precision']`` before
dropping the column. These tests pin the round trip.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

REVISIONS = Path("alembic/versions")


def test_the_drop_revision_backfills_before_dropping() -> None:
    """Order matters: a DROP before the UPDATE would discard every value."""
    source = _drop_revision_source()

    backfill = source.index("config_snapshot")
    drop = source.index("drop_column")

    assert backfill < drop


def test_the_drop_revision_coalesces_a_null_snapshot() -> None:
    """``json_set(NULL, ...)`` returns NULL, silently discarding every value.

    Verified on both MySQL and SQLite (3.50.2) — this is not a MySQL-only quirk.
    """
    assert "COALESCE" in _drop_revision_source().upper()


def test_the_drop_revision_has_a_downgrade_that_restores_the_column() -> None:
    source = _drop_revision_source()

    assert "def downgrade" in source
    assert "add_column" in source


def test_the_drop_revision_handles_postgres_separately() -> None:
    """Postgres has no ``json_set``; it needs ``jsonb_build_object``."""
    source = _drop_revision_source()

    assert "jsonb" in source


def test_the_status_revision_adds_both_columns_and_can_downgrade() -> None:
    source = _status_revision_source()

    assert 'down_revision = "78f6bb7de0df"' in source
    assert '"status"' in source and '"status_detail"' in source
    assert 'server_default="empty"' in source
    assert "def downgrade" in source
    assert "drop_column" in source


def _status_revision_source() -> str:
    """Return the source of the revision that adds ``datasets.status``.

    Keyed on the revision's unique ``down_revision`` rather than the column name:
    later revisions (e.g. making ``datasets.description`` nullable) legitimately
    reference ``status_detail`` in a ``copy_from`` table definition, so a
    column-name match is not unique.
    """
    matches = [
        path.read_text()
        for path in REVISIONS.glob("*.py")
        if "status_detail" in path.read_text()
        and 'down_revision = "78f6bb7de0df"' in path.read_text()
    ]

    assert len(matches) == 1, f"expected exactly one status revision, found {len(matches)}"
    return matches[0]


def _drop_revision_source() -> str:
    """Return the source of the revision that drops ``precision``."""
    matches = [
        path.read_text()
        for path in REVISIONS.glob("*.py")
        if "precision" in path.read_text() and "drop_column" in path.read_text()
    ]

    assert len(matches) == 1, f"expected exactly one precision revision, found {len(matches)}"
    return matches[0]


_LAST_LOGIN_REVISION = "a3c71d94e5b2"
_REVISION_BEFORE_LAST_LOGIN = "f09bd54b61b7"


def _alembic(database: Path, *args: str) -> None:
    """Run an alembic command against ``database`` in a subprocess.

    A subprocess rather than ``alembic.command``: ``alembic/env.py`` drives the
    async engine with ``asyncio.run``, which raises if called from inside the
    event loop pytest-asyncio's auto mode already has running. It also keeps the
    developer's own ``.env`` out of it — ``AUTOTUNEX_JOB_BACKEND`` is pinned here
    for the same reason ``make_settings`` pins its fields, so a local
    ``job_backend=llmb`` cannot fail ``Settings`` construction inside the child.
    """
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        env={
            **os.environ,
            "AUTOTUNEX_DATABASE_URL": f"sqlite+aiosqlite:///{database}",
            "AUTOTUNEX_JOB_BACKEND": "none",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"alembic {' '.join(args)} failed:\n{result.stderr}"


def test_the_last_login_revision_backfills_from_updated_at(tmp_path: Path) -> None:
    """The pre-refactor code wrote logins into ``updated_at``; that data must survive.

    Without the backfill, every existing user's last login reads as unknown the
    moment this revision lands — which is the whole reason the column carries a
    value worth migrating rather than starting empty.
    """
    database = tmp_path / "backfill.db"
    _alembic(database, "upgrade", _REVISION_BEFORE_LAST_LOGIN)
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, email, role, created_at, updated_at) VALUES "
                "('11111111-1111-1111-1111-111111111111', 'veteran@example.com', 'user', "
                "'2026-01-01 00:00:00.000000', '2026-08-30 12:34:56.000000')"
            )
        )

    _alembic(database, "upgrade", _LAST_LOGIN_REVISION)

    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT updated_at, last_login_at FROM users WHERE email = 'veteran@example.com'")
        ).one()
    assert row.last_login_at == row.updated_at


def test_the_last_login_revision_leaves_a_never_touched_row_null(tmp_path: Path) -> None:
    """``updated_at == created_at`` means nothing ever wrote the row, so no login happened.

    The tuning pipeline creates users it never logs in as. Copying ``updated_at``
    unconditionally would give every one of them a fabricated login equal to their
    creation time, which is the ``created_at``-as-login claim this column exists to
    avoid making.
    """
    database = tmp_path / "untouched.db"
    _alembic(database, "upgrade", _REVISION_BEFORE_LAST_LOGIN)
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, email, role, created_at, updated_at) VALUES "
                "('22222222-2222-2222-2222-222222222222', 'pipeline@example.com', 'user', "
                "'2026-02-02 00:00:00.000000', '2026-02-02 00:00:00.000000')"
            )
        )

    _alembic(database, "upgrade", _LAST_LOGIN_REVISION)

    with engine.connect() as connection:
        last_login = connection.execute(
            text("SELECT last_login_at FROM users WHERE email = 'pipeline@example.com'")
        ).scalar_one()
    assert last_login is None


def test_the_last_login_revision_pins_updated_at_in_the_backfill() -> None:
    """MySQL's ``ON UPDATE CURRENT_TIMESTAMP`` would otherwise rewrite every row.

    A source assertion, not an executed one, and deliberately: the behaviour this
    guards exists only on MySQL, whose live ``users.updated_at`` carries ``ON
    UPDATE CURRENT_TIMESTAMP``. SQLite has no such clause, so a round-trip test
    there would pass whether or not the assignment is present and would prove
    nothing. Same reasoning as ``test_the_drop_revision_handles_postgres_separately``
    above. Dropping the assignment would set every backfilled row's ``updated_at``
    to the migration's own run time, flattening the table's modification history.
    """
    source = _last_login_revision_source()

    assert "updated_at=users.c.updated_at" in source


def test_the_last_login_revision_backfills_after_adding_the_column() -> None:
    """An UPDATE inside the batch context would target SQLite's throwaway copy."""
    source = _last_login_revision_source()

    assert source.index("add_column") < source.index("op.execute")


def test_the_last_login_revision_has_a_downgrade_that_drops_the_column() -> None:
    source = _last_login_revision_source()

    assert "def downgrade" in source
    assert "drop_column" in source


def _last_login_revision_source() -> str:
    """Return the source of the revision that adds ``users.last_login_at``."""
    matches = [
        path.read_text()
        for path in REVISIONS.glob("*.py")
        if f'revision = "{_LAST_LOGIN_REVISION}"' in path.read_text()
    ]

    assert len(matches) == 1, f"expected exactly one last-login revision, found {len(matches)}"
    return matches[0]
