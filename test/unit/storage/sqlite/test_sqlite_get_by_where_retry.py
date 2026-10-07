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

"""get_by_where must stay retried.

The tenacity @retry lives on the private BaseSQLItemStorage._get_by_where_with_retry
rather than on the public get_by_where (so mypy accepts the SQLite storages'
mixin, see that method's docstring). These tests pin down that a call through the
public method -- including the SQLite mixin's locked override -- still retries a
transient failure, so a refactor that drops the retry is caught.
"""

from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError
from tenacity import wait_none

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_space import StoredSpace

_RETRYING = BaseSQLItemStorage.__dict__["_get_by_where_with_retry"].retry


@pytest.fixture
def space_storage():
    storage = SqliteStorageFactory().create_space_storage(
        table_name="test_get_by_where_retry"
    )
    storage.add(StoredSpace(name="s1", git_repo_uri="file:///s1"))
    return storage


def _fail_then_succeed(storage, failures):
    """Patch the row lookup to raise ``failures`` times before working."""
    real = storage._get_by_where_row_dicts
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise OperationalError("SELECT", {}, Exception("database is locked"))
        return real(*args, **kwargs)

    return patch.object(storage, "_get_by_where_row_dicts", side_effect=flaky), calls


def test_get_by_where_retries_transient_failure(space_storage):
    patcher, calls = _fail_then_succeed(space_storage, failures=2)
    with patcher, patch.object(_RETRYING, "wait", wait_none()):
        items = space_storage.get_by_where({"name": "s1"})

    assert [s.name for s in items] == ["s1"]
    assert calls["n"] == 3  # two failures, then success


def test_get_by_where_gives_up_after_the_attempt_limit(space_storage):
    patcher, calls = _fail_then_succeed(space_storage, failures=100)
    with patcher, patch.object(_RETRYING, "wait", wait_none()):
        with pytest.raises(OperationalError):
            space_storage.get_by_where({"name": "s1"})

    assert calls["n"] == 10  # stop_after_attempt(10), then re-raised
