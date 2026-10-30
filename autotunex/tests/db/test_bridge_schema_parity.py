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

Each default is checked against the ORM only. That the migrations build the ORM's
defaults is ``test_migrations.test_alembic_check_finds_no_drift_at_head``'s job:
``alembic/env.py`` sets ``compare_server_default``, and Alembic cannot see the bridge's
own ``MetaData``.
"""

from __future__ import annotations

import importlib.util
from functools import cache
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


@cache
def _bridge_metadata() -> MetaData:
    """Loaded once: every test and parametrized case reads the same ``MetaData``."""
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


@cache
def _sqlite_defaults(metadata: MetaData) -> dict[tuple[str, str], str | None]:
    """Every column's default as SQLite renders it, unquoted: ``'42'`` and ``42`` read alike.

    Built once per ``MetaData``. An ``INT`` column stores either as 42, so the quoting
    is not a difference that matters (the bridge writes ``seed`` as a string, the ORM
    as ``text("42")``).
    """
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    inspector = inspect(engine)
    return {
        (table, c["name"]): None if c["default"] is None else str(c["default"]).strip("'")
        for table in inspector.get_table_names()
        for c in inspector.get_columns(table)
    }


def _sqlite_default(metadata: MetaData, table: str, column: str) -> str | None:
    return _sqlite_defaults(metadata)[(table, column)]


@pytest.mark.parametrize(("table", "column"), _bridge_defaulted_columns())
def test_the_orm_and_bridge_server_defaults_agree(table: str, column: str) -> None:
    bridge_table = _bridge_metadata().tables[table]
    assert isinstance(bridge_table, Table)

    assert _sqlite_default(Base.metadata, table, column) == _sqlite_default(
        bridge_table.metadata, table, column
    )


def test_the_bridge_status_defaults_are_found() -> None:
    """Guards the parametrize source: an empty list would make the tests above vacuous."""
    assert {("gb_tasks", "status"), ("users", "role")} <= set(_bridge_defaulted_columns())
