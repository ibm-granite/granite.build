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
without this fixture they would pile up in the developer's real db.
Point GB_HOME_DIR at a per-test temp dir instead.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from gbcommon.types.constants import get_gb_home_dir
from gbserver.storage import singleton_storage
from gbserver.storage.sql.engine_cache import get_singleton_engine_cache
from gbserver.storage.sqlite.sqlite_storage import SQLITE_DB_FILE_NAME

# Cached module globals in singleton_storage that a test can populate lazily.
_SINGLETON_GLOBALS = ("__STORAGE", "__STORAGE_FACTORY")


def _real_db_tables(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    # Read-only, so the guard itself never touches the developer's db.
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {name for (name,) in rows}


@pytest.fixture(scope="session", autouse=True)
def _guard_real_gb_home_db(request):
    """Fail the run if any unit test left new tables in the real db.

    Resolved before any per-test override, so it is the db the developer's
    standalone server actually uses. Only active when every collected test is a
    unit test: integration/e2e tests are not isolated by this conftest and may
    legitimately use the real db, which would show up here as a false positive.
    """
    unit_dir = Path(__file__).parent.resolve()
    if not all(item.path.resolve().is_relative_to(unit_dir) for item in request.session.items):
        yield
        return
    db_path = Path(get_gb_home_dir()) / SQLITE_DB_FILE_NAME
    existed = db_path.exists()
    try:
        before = _real_db_tables(db_path)
    except sqlite3.Error:
        # Locked, WAL without -shm, or corrupt: can't snapshot, so don't guard.
        yield
        return
    yield
    try:
        after = _real_db_tables(db_path)
    except sqlite3.Error:
        return
    leaked = sorted(after - before)
    created = not existed and db_path.exists()
    assert not leaked and not created, (
        f"Unit tests wrote to the real SQLite db {db_path}: "
        f"{'created the file; ' if created else ''}new tables {leaked}. "
        "Some test bypasses the GB_HOME_DIR isolation in test/unit/conftest.py."
    )


@pytest.fixture(autouse=True)
def _isolate_gb_home_dir(tmp_path_factory, monkeypatch):
    # Outside the test's own tmp_path: some tests assert exactly what is in it, and
    # some create a "gb_home" there themselves to check it gets created.
    home = tmp_path_factory.mktemp("gb_home_sqlite")
    # sqlite_storage resolves the home dir via get_gb_home_dir() when a storage is
    # constructed, so setting the env var is enough to redirect storage built during
    # the test. Storage built earlier (setup_class, module/session fixtures) still
    # points at the real db.
    monkeypatch.setenv("GB_HOME_DIR", str(home))
    # A test that lazily builds the admin storage singleton would pin it to its own
    # temp db for every later test; restore whatever was there before (None, or a
    # value a module-scoped fixture set on purpose).
    saved = {name: getattr(singleton_storage, name) for name in _SINGLETON_GLOBALS}
    yield
    for name, value in saved.items():
        setattr(singleton_storage, name, value)
    # Each test gets its own db path and so its own cached engine; dispose them so
    # pooled connections don't accumulate until atexit. This also disposes Postgres
    # engines under live-storage tests (and logs INFO lines per engine); that is
    # expected and harmless, they are recreated on demand.
    get_singleton_engine_cache().dispose_all()
