"""The precision drop must not lose data.

``jobs.precision`` was ``VARCHAR(50) NOT NULL`` and the live database has values
in it, so the migration backfills ``config_snapshot['precision']`` before
dropping the column. These tests pin the round trip.
"""

from __future__ import annotations

import os
import re
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
    """Return the source of the revision that adds ``users.last_login_at``.

    Matched with a line-start anchor, not a bare substring: an unanchored
    ``revision = "..."`` search also matches any later revision's
    ``down_revision = "..."`` line, since that line ends with the same text.
    Without the anchor, the very next migration built on top of this one
    would make this helper find two "matches" instead of one.
    """
    pattern = re.compile(rf'^revision = "{_LAST_LOGIN_REVISION}"$', re.MULTILINE)
    matches = [
        path.read_text() for path in REVISIONS.glob("*.py") if pattern.search(path.read_text())
    ]

    assert len(matches) == 1, f"expected exactly one last-login revision, found {len(matches)}"
    return matches[0]


_HF_PROVENANCE_REVISION = "dd4511a46fc5"
_REVISION_BEFORE_HF_PROVENANCE = "a3c71d94e5b2"


def test_the_raw_schema_file_does_not_declare_the_hf_provenance_columns() -> None:
    """A post-baseline revision's columns must not also be in the schema file.

    CLAUDE.md's bring-up path for a database built from
    ``resources/autotunex_schema.sql`` is ``alembic stamp 1fb645a87b48`` then
    ``alembic upgrade head``: the baseline's ``upgrade()`` would fail on the
    already-existing tables, so it is stamped, and every later revision runs for
    real. A column declared in *both* places therefore makes that upgrade die on
    MySQL error 1060 (duplicate column) -- unlike the known ``trials.id`` re-widen
    divergence, which is merely redundant.

    This is why ``datasets.status``, ``datasets.status_detail``,
    ``jobs.reward_function_code``/``_name``, ``users.last_login_at`` and the whole
    ``training_metrics`` table are all ORM-and-migration only (see *Open decisions*
    6). These five belong to that set.
    """
    schema = Path("resources/autotunex_schema.sql").read_text()

    declared = {
        column
        for column in ("hf_repo_id", "hf_revision", "hf_config", "hf_split", "hf_provenance")
        if column in schema
    }

    assert not declared, (
        f"{sorted(declared)} are in both the schema file and revision "
        f"{_HF_PROVENANCE_REVISION}; a stamped baseline then `upgrade head` would "
        "fail on a duplicate column."
    )


def test_the_hf_provenance_revision_adds_columns_and_downgrade_drops_them(
    tmp_path: Path,
) -> None:
    """The one revision on this table that is not ``batch_alter_table``.

    ``datasets`` has two ``GENERATED ALWAYS AS (...) STORED`` columns
    (``train_file``, ``validation_file``) that SQLite batch mode's
    copy-and-recreate cannot round-trip, so this revision uses plain
    ``add_column``/``drop_column`` instead -- the one migration deviating from
    every other revision's precedent in this file. A round trip through both
    directions is what actually exercises that choice, rather than a
    source-only assertion.
    """
    database = tmp_path / "hf_provenance.db"
    _alembic(database, "upgrade", _REVISION_BEFORE_HF_PROVENANCE)
    engine = create_engine(f"sqlite:///{database}")
    hf_columns = {"hf_repo_id", "hf_revision", "hf_config", "hf_split", "hf_provenance"}

    with engine.connect() as connection:
        before = {row[1] for row in connection.execute(text("PRAGMA table_info(datasets)"))}
    assert not (hf_columns & before)

    _alembic(database, "upgrade", _HF_PROVENANCE_REVISION)
    with engine.connect() as connection:
        after_upgrade = {row[1] for row in connection.execute(text("PRAGMA table_info(datasets)"))}
    assert hf_columns <= after_upgrade

    _alembic(database, "downgrade", _REVISION_BEFORE_HF_PROVENANCE)
    with engine.connect() as connection:
        after_downgrade = {
            row[1] for row in connection.execute(text("PRAGMA table_info(datasets)"))
        }
    assert not (hf_columns & after_downgrade)


_BRIDGE_DEFAULTS_REVISION = "5e1d2c7a9b40"
_REVISION_BEFORE_BRIDGE_DEFAULTS = "dd4511a46fc5"
# (table, column) -> the DDL default resources/autotunex_schema.sql declares.
_BRIDGE_DEFAULTS = {
    ("gb_tasks", "status"): "'PENDING'",
    ("jobs", "status"): "'PENDING'",
    ("jobs", "seed"): "42",
    ("trials", "status"): "'PENDING'",
    ("users", "role"): "'user'",
}


def _column_defaults(database: Path) -> dict[tuple[str, str], str | None]:
    engine = create_engine(f"sqlite:///{database}")
    with engine.connect() as connection:
        return {
            (table, row[1]): row[4]
            for table in {table for table, _ in _BRIDGE_DEFAULTS}
            for row in connection.execute(text(f"PRAGMA table_info({table})"))
        }


def test_the_bridge_defaults_revision_adds_the_schema_file_defaults(tmp_path: Path) -> None:
    """The api-bridge's Core inserts omit these columns and rely on a DDL default.

    The baseline gave them none, so its bootstrap failed ``NOT NULL`` on
    ``gb_tasks.status``. ``status`` holds the enum member *name*: the ORM's
    ``Enum(RunStatus)`` stores ``PENDING``, not the ``pending`` value.
    """
    database = tmp_path / "bridge_defaults.db"
    _alembic(database, "upgrade", _REVISION_BEFORE_BRIDGE_DEFAULTS)
    before = _column_defaults(database)
    assert all(before[key] is None for key in _BRIDGE_DEFAULTS)

    _alembic(database, "upgrade", _BRIDGE_DEFAULTS_REVISION)
    after = _column_defaults(database)
    assert {key: after[key] for key in _BRIDGE_DEFAULTS} == _BRIDGE_DEFAULTS

    _alembic(database, "downgrade", _REVISION_BEFORE_BRIDGE_DEFAULTS)
    assert _column_defaults(database) == before


def test_a_gb_task_inserted_without_a_status_is_pending(tmp_path: Path) -> None:
    """The bootstrap-shaped insert: every NOT NULL column but ``status``."""
    database = tmp_path / "gb_task_insert.db"
    _alembic(database, "upgrade", _BRIDGE_DEFAULTS_REVISION)
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO gb_tasks (id, job_id, type) VALUES "
                "('33333333-3333-3333-3333-333333333333', "
                "'44444444-4444-4444-4444-444444444444', 'TUNING')"
            )
        )
        status = connection.execute(text("SELECT status FROM gb_tasks")).scalar_one()
    assert status == "PENDING"


def test_alembic_check_finds_no_drift_at_head(tmp_path: Path) -> None:
    """The ORM and the migrations build the same schema, server defaults included.

    ``alembic/env.py`` sets ``compare_server_default``, so a ``server_default``
    added to the ORM without a migration fails here. ``make migrations-check``
    runs the same command but is not part of ``make check``.
    """
    database = tmp_path / "check.db"
    _alembic(database, "upgrade", "head")
    _alembic(database, "check")
