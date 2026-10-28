"""The api-bridge's ``tables.py`` agrees with the ORM on what the database must provide.

The bridge writes the same tables through SQLAlchemy Core and its own copy of their
definitions, so the ORM's Python-side defaults never apply to it. Where the bridge
declares a server default, an insert there may omit the column, and the database must
then supply it. The live database is built from this ORM plus Alembic, so the bridge
tests passing against its own ``metadata.create_all`` proved nothing about that. A
``gb_tasks.status`` the bridge defaulted and the ORM did not was the bootstrap's
``NOT NULL`` 500.

The module is loaded by file path: the bridge is a separate distribution that is not
installed for this suite, and its ``tables.py`` imports nothing but SQLAlchemy.

Each default is checked against both the ORM and the schema ``alembic upgrade head``
builds, because a ``server_default`` added to the ORM without a migration passes the
ORM comparison, and ``alembic check`` does not compare server defaults
(``alembic/env.py`` sets ``compare_type`` only).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import MetaData, Table, create_engine, inspect

from autotunex.db import tables  # noqa: F401  (registers tables on Base.metadata)
from autotunex.db.base import Base

BRIDGE_TABLES = Path(__file__).resolve().parents[2] / "src/api-bridge/src/api_bridge/tables.py"


def _bridge_tables() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_bridge_tables", BRIDGE_TABLES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bridge_metadata() -> MetaData:
    metadata = _bridge_tables().metadata
    assert isinstance(metadata, MetaData)
    return metadata


def _bridge_defaulted_columns() -> list[tuple[str, str]]:
    return [
        (table.name, column.name)
        for table in _bridge_metadata().tables.values()
        for column in table.columns
        if column.server_default is not None
    ]


def test_every_bridge_table_is_an_orm_table() -> None:
    assert set(_bridge_metadata().tables) <= set(Base.metadata.tables)


@pytest.mark.parametrize(("table", "column"), _bridge_defaulted_columns())
def test_a_column_the_bridge_defaults_has_a_server_default_in_the_orm(
    table: str, column: str
) -> None:
    orm_column = Base.metadata.tables[table].c[column]

    assert orm_column.server_default is not None, (
        f"{table}.{column} has a server default in api_bridge/tables.py but not in the "
        "ORM, so a bridge insert that omits it fails or stores NULL on a database "
        "built by Alembic. Add server_default= to the ORM column and a migration."
    )


def _sqlite_default(metadata: MetaData, table: str, column: str) -> str | None:
    """The column's default as SQLite renders it, unquoted: ``'42'`` and ``42`` read alike.

    An ``INT`` column stores either as 42, so the quoting is not a difference that
    matters (the bridge writes ``seed`` as a string, the ORM as ``text("42")``).
    """
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    rendered = {c["name"]: c["default"] for c in inspect(engine).get_columns(table)}[column]
    return None if rendered is None else str(rendered).strip("'")


@pytest.mark.parametrize(("table", "column"), _bridge_defaulted_columns())
def test_the_orm_and_bridge_server_defaults_agree(table: str, column: str) -> None:
    bridge_table = _bridge_metadata().tables[table]
    assert isinstance(bridge_table, Table)

    assert _sqlite_default(Base.metadata, table, column) == _sqlite_default(
        bridge_table.metadata, table, column
    )


@pytest.fixture(scope="module")
def migrated_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A SQLite database at ``alembic upgrade head``, built once for this module.

    A subprocess, as in ``test_migrations._alembic``: ``alembic/env.py`` runs its own
    event loop, and the developer's ``.env`` must not reach ``Settings``.
    """
    database = tmp_path_factory.mktemp("bridge_parity") / "head.db"
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env={
            **os.environ,
            "AUTOTUNEX_DATABASE_URL": f"sqlite+aiosqlite:///{database}",
            "AUTOTUNEX_JOB_BACKEND": "none",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"alembic upgrade head failed:\n{result.stderr}"
    return database


def _migrated_default(database: Path, table: str, column: str) -> str | None:
    columns = inspect(create_engine(f"sqlite:///{database}")).get_columns(table)
    rendered = {c["name"]: c["default"] for c in columns}[column]
    return None if rendered is None else str(rendered).strip("'")


@pytest.mark.parametrize(("table", "column"), _bridge_defaulted_columns())
def test_the_migrated_schema_has_the_bridge_server_default(
    migrated_database: Path, table: str, column: str
) -> None:
    bridge_table = _bridge_metadata().tables[table]
    assert isinstance(bridge_table, Table)

    assert _migrated_default(migrated_database, table, column) == _sqlite_default(
        bridge_table.metadata, table, column
    ), (
        f"{table}.{column}: the bridge's server default is missing from, or differs "
        "in, the schema Alembic builds. Add or fix a migration."
    )


def test_the_bridge_status_defaults_are_found() -> None:
    """Guards the parametrize source: an empty list would make the tests above vacuous."""
    assert {("gb_tasks", "status"), ("users", "role")} <= set(_bridge_defaulted_columns())
