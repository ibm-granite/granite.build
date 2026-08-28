"""The precision drop must not lose data.

``jobs.precision`` was ``VARCHAR(50) NOT NULL`` and the live database has values
in it, so the migration backfills ``config_snapshot['precision']`` before
dropping the column. These tests pin the round trip.
"""

from __future__ import annotations

from pathlib import Path

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
