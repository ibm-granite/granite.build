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
import sys
from contextlib import closing
from pathlib import Path

import pytest

from gbcommon.types.constants import get_gb_home_dir
from gbserver.storage import singleton_storage
from gbserver.storage.sql.engine_cache import get_singleton_engine_cache
from gbserver.storage.sqlite.sqlite_storage import SQLITE_DB_FILE_NAME

# Cached module globals in singleton_storage that a test can populate lazily.
_SINGLETON_GLOBALS = ("__STORAGE", "__STORAGE_FACTORY")

# Other lazily built process-wide singletons that hold (or reach) storage, as
# (module, global, unset value). A test that builds one binds it to its own temp
# db for every later test. They are looked up in sys.modules rather than
# imported, so this fixture never pulls a module (and its import-time work) in.
_OTHER_SINGLETONS = (
    ("gbserver.resilience.nht_singleton", "_node_health_tracker", None),
    ("gbserver.resilience.nht_singleton", "_tracker_initialized", False),
    ("gbserver.spaces.space_access_manager", "_storage_manager", None),
    ("gbserver.api.lineage", "_openlineage_service", None),
    ("gbserver.api.lineage", "_index_service", None),
    ("gbserver.metrics.metrics_client", "_SINGLETON_METRICS_CLIENT", None),
)


def _save_other_singletons() -> list:
    """Each singleton's current value, or its unset value if its module isn't loaded yet."""
    saved = []
    for module_name, name, unset in _OTHER_SINGLETONS:
        module = sys.modules.get(module_name)
        if module is None:
            saved.append((module_name, name, unset))
            continue
        # Fail loudly if a global was renamed: restoring it would silently do nothing.
        assert hasattr(
            module, name
        ), f"{module_name}.{name} no longer exists; update _OTHER_SINGLETONS in {__file__}"
        saved.append((module_name, name, getattr(module, name)))
    return saved


def _restore_other_singletons(saved: list) -> None:
    for module_name, name, value in saved:
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, name, value)


_home_counter = 0


def _real_db_tables(db_path: Path) -> set[str]:
    """Return the table names in the SQLite db at ``db_path``, or an empty set if it doesn't exist."""
    if not db_path.exists():
        return set()
    # Read-only, so the guard itself never touches the developer's db.
    with closing(
        sqlite3.connect(db_path.absolute().as_uri() + "?mode=ro", uri=True)
    ) as conn:
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
    if not all(
        item.path.resolve().is_relative_to(unit_dir) for item in request.session.items
    ):
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
    # Rows written into existing gb_* tables don't show up as new tables, so also
    # fail if anything in this process opened an engine on the real db at all.
    # Engines on temp homes are disposed per test; one left on the real db means
    # some storage was built before the GB_HOME_DIR override (setup_class, module
    # or session fixtures) and may have written to it.
    opened = [
        str(engine.url)
        for engine in list(get_singleton_engine_cache().engines.values())
        if engine.url.get_backend_name() == "sqlite"
        and _is_same_file(engine.url.database, db_path)
    ]
    assert not opened, (
        f"Unit tests opened the real SQLite db {db_path} via {opened}. "
        "Some test bypasses the GB_HOME_DIR isolation in test/unit/conftest.py."
    )
    try:
        after = _real_db_tables(db_path)
    except sqlite3.Error:
        return
    # gb_* are the real admin tables; a standalone server sharing this db may
    # create them lazily during the run, which is not a test leak.
    leaked = sorted(t for t in after - before if not t.startswith("gb_"))
    created = not existed and db_path.exists()
    assert not leaked and not created, (
        f"Unit tests wrote to the real SQLite db {db_path}: "
        f"{'created the file; ' if created else ''}new tables {leaked}. "
        "Some test bypasses the GB_HOME_DIR isolation in test/unit/conftest.py."
    )


@pytest.fixture(scope="session")
def _gb_home_base(tmp_path_factory) -> Path:
    # Outside each test's own tmp_path: some tests assert exactly what is in it, and
    # some create a "gb_home" there themselves to check it gets created.
    return tmp_path_factory.mktemp("gb_home_sqlite")


@pytest.fixture(autouse=True)
def _isolate_gb_home_dir(_gb_home_base, monkeypatch):
    """Point GB_HOME_DIR at a fresh temp dir for each test and restore the singletons after it."""
    # A unique path that is not created here: sqlite_storage creates it on first use,
    # so the many tests that never touch storage cost no directory.
    global _home_counter
    _home_counter += 1
    home = _gb_home_base / f"t{_home_counter}"
    # sqlite_storage resolves the home dir via get_gb_home_dir() when a storage is
    # constructed, so setting the env var is enough to redirect storage built during
    # the test. Storage built earlier (setup_class, module/session fixtures) still
    # points at the real db.
    monkeypatch.setenv("GB_HOME_DIR", str(home))
    # A test that lazily builds the admin storage singleton would pin it to its own
    # temp db for every later test; restore whatever was there before (None, or a
    # value a module-scoped fixture set on purpose).
    saved = {name: getattr(singleton_storage, name) for name in _SINGLETON_GLOBALS}
    saved_others = _save_other_singletons()
    yield
    for name, value in saved.items():
        setattr(singleton_storage, name, value)
    _restore_other_singletons(saved_others)
    # Each test gets its own db path and so its own cached engine; dispose those so
    # pooled connections don't accumulate until atexit. Other engines (e.g. Postgres
    # pools under live-storage tests) are kept, so they don't reconnect every test.
    if home.exists():
        get_singleton_engine_cache().dispose_matching(
            lambda engine: _is_under(engine.url.database, home)
        )


def _is_same_file(database: str | None, path: Path) -> bool:
    if not database:
        return False
    try:
        return Path(database).resolve() == path.resolve()
    except (OSError, ValueError):
        return False


def _is_under(database: str | None, home: Path) -> bool:
    if not database:
        return False
    try:
        return Path(database).resolve().is_relative_to(home.resolve())
    except (OSError, ValueError):
        return False
