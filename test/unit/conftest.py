#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Isolate the unit tests from the developer's real home directory.

The standalone SQLite db lives under ``GB_HOME_DIR`` (default ~/.granite.build).
Many unit tests create randomly named tables through ``SqliteStorageFactory()``;
without this fixture they would pile up in the developer's real db (#574).
Point GB_HOME_DIR at a per-test temp dir instead.
"""

import sqlite3
from pathlib import Path

import pytest

from gbcommon.types.constants import get_gb_home_dir
from gbserver.storage import singleton_storage
from gbserver.storage.sqlite.sqlite_storage import SQLITE_DB_FILE_NAME
from gbserver.storage.sql.engine_cache import get_singleton_engine_cache

# Cached module globals in singleton_storage that a test can populate lazily.
_SINGLETON_GLOBALS = ("__STORAGE", "__STORAGE_FACTORY")


def _real_db_tables(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    # Read-only, so the guard itself never touches the developer's db.
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {name for (name,) in rows}


@pytest.fixture(scope="session", autouse=True)
def _guard_real_gb_home_db():
    """Fail the run if any unit test left new tables in the real db (#574).

    Resolved before any per-test override, so it is the db the developer's
    standalone server actually uses.
    """
    db_path = Path(get_gb_home_dir()) / SQLITE_DB_FILE_NAME
    existed = db_path.exists()
    before = _real_db_tables(db_path)
    yield
    leaked = sorted(_real_db_tables(db_path) - before)
    created = not existed and db_path.exists()
    assert not leaked and not created, (
        f"Unit tests wrote to the real SQLite db {db_path}: "
        f"{'created the file; ' if created else ''}new tables {leaked}. "
        "Some test bypasses the GB_HOME_DIR isolation in test/unit/conftest.py."
    )


@pytest.fixture(autouse=True)
def _isolate_gb_home_dir(tmp_path, monkeypatch):
    # Not "gb_home": some tests create that dir themselves to check it gets created.
    home = tmp_path / "gb_home_sqlite"
    home.mkdir()
    # sqlite_storage resolves the home dir via get_gb_home_dir() at call time, so
    # setting the env var is enough to redirect it to a per-test temp dir.
    monkeypatch.setenv("GB_HOME_DIR", str(home))
    # A test that lazily builds the admin storage singleton would pin it to its own
    # temp db for every later test; restore whatever was there before (None, or a
    # value a module-scoped fixture set on purpose).
    saved = {name: getattr(singleton_storage, name) for name in _SINGLETON_GLOBALS}
    yield
    for name, value in saved.items():
        setattr(singleton_storage, name, value)
    # Each test gets its own db path and so its own cached engine; dispose them so
    # pooled connections don't accumulate until atexit.
    get_singleton_engine_cache().dispose_all()
