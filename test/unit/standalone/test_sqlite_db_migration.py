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

"""Tests for the one-time ~/.llmb -> GB_HOME_DIR standalone SQLite db migration."""

import importlib

import pytest

from gbserver.storage.sqlite.sqlite_storage import (
    LEGACY_LLMB_DIR_NAME,
    SQLITE_DB_FILE_NAME,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake home with a legacy ~/.llmb dir and a fresh GB_HOME_DIR, both under tmp."""
    fake_home = tmp_path / "home"
    legacy_dir = fake_home / LEGACY_LLMB_DIR_NAME
    legacy_dir.mkdir(parents=True)
    gb_home = tmp_path / "granite_build"

    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("GB_HOME_DIR", str(gb_home))

    # GB_HOME_DIR is read at import time; reload so the migration helper sees tmp paths.
    import gbcommon.types.constants

    importlib.reload(gbcommon.types.constants)
    import gbserver.commands.command_standalone as cmd

    importlib.reload(cmd)

    legacy_db = legacy_dir / SQLITE_DB_FILE_NAME
    new_db = gb_home / SQLITE_DB_FILE_NAME
    return cmd, legacy_db, new_db


def test_migrates_when_new_absent(env):
    cmd, legacy_db, new_db = env
    legacy_db.write_text("legacy-db-contents")

    cmd._migrate_legacy_sqlite_db()

    assert new_db.exists()
    assert new_db.read_text() == "legacy-db-contents"
    # Legacy file preserved as backup.
    assert legacy_db.exists()


def test_does_not_overwrite_existing_new_db(env):
    cmd, legacy_db, new_db = env
    legacy_db.write_text("legacy")
    new_db.parent.mkdir(parents=True, exist_ok=True)
    new_db.write_text("current")

    cmd._migrate_legacy_sqlite_db()

    # New db is the source of truth; must not be clobbered.
    assert new_db.read_text() == "current"


def test_noop_when_no_legacy_db(env):
    cmd, legacy_db, new_db = env
    assert not legacy_db.exists()

    cmd._migrate_legacy_sqlite_db()

    assert not new_db.exists()


def test_idempotent(env):
    cmd, legacy_db, new_db = env
    legacy_db.write_text("legacy")

    cmd._migrate_legacy_sqlite_db()
    cmd._migrate_legacy_sqlite_db()

    assert new_db.read_text() == "legacy"
