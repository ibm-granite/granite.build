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

The tenacity @retry lives on a private helper rather than on the public
get_by_where (so mypy accepts the SQLite storages' mixin, see get_by_where's
docstring). These tests pin down that a call through the public method --
including the SQLite mixin's locked override -- still retries a transient
failure, so a refactor that drops the retry is caught. They also pin the retry
boundary (a deterministic ValueError is not retried) and that the
exact_liked_list_columns post-filter still applies on a retried call.

Backoff is skipped by replacing ``tenacity.nap.time``, the module reference
through which tenacity's default sleep (``tenacity.nap.sleep``, documented as the
one to mock in tests) looks up ``time.sleep``. This keeps the patch inside
tenacity: the process-wide ``time.sleep`` -- used by other threads, filelock, and
the autouse ``_mock_time`` fixture in mock mode -- is untouched, so the sleep
counts are exact. It also avoids reaching into the decorated function's Retrying
object. (Patching ``tenacity.nap.sleep`` itself would not work: Retrying binds it
as a default argument at import time.)
"""

from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.artifact import ArtifactType

# Patch only tenacity's reference to the time module, so the global time.sleep
# (and any other thread using it) is untouched.
_TENACITY_TIME = "tenacity.nap.time"


@pytest.fixture
def space_storage():
    storage = SqliteStorageFactory().create_space_storage(
        table_name="test_get_by_where_retry"
    )
    storage.add(StoredSpace(name="s1", git_repo_uri="file:///s1"))
    return storage


@pytest.fixture
def artifact_registry():
    """Registry with tags ["a"] and ["ab"]; a %a% LIKE on tags matches both."""
    registry = SqliteStorageFactory().create_artifact_registry(
        table_name="test_get_by_where_retry_artifacts"
    )
    for name, tags in (("exact", ["a"]), ("superset", ["ab"])):
        registry.add(
            ArtifactRegistration(
                type=ArtifactType.MODEL,
                uri=f"https://example.com/{name}",
                username="me",
                space_name="space",
                tags=tags,
            )
        )
    return registry


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
    with patcher, patch(_TENACITY_TIME) as fake_time:
        items = space_storage.get_by_where({"name": "s1"})

    assert [s.name for s in items] == ["s1"]
    assert calls["n"] == 3  # two failures, then success
    assert fake_time.sleep.call_count == 2  # one backoff per failure


def test_get_by_where_gives_up_after_the_attempt_limit(space_storage):
    patcher, calls = _fail_then_succeed(space_storage, failures=100)
    with patcher, patch(_TENACITY_TIME) as fake_time:
        with pytest.raises(OperationalError):
            space_storage.get_by_where({"name": "s1"})

    assert calls["n"] == 10  # stop_after_attempt(10), then re-raised
    assert fake_time.sleep.call_count == 9  # no backoff after the final attempt


def test_get_by_where_does_not_retry_value_error(space_storage):
    with patch(_TENACITY_TIME) as fake_time:
        with pytest.raises(ValueError):
            space_storage.get_by_where(42)  # type: ignore[arg-type]

    assert fake_time.sleep.call_count == 0  # surfaced on the first attempt, no backoff


def test_get_by_where_post_filters_exact_tags_after_retry(artifact_registry):
    patcher, calls = _fail_then_succeed(artifact_registry, failures=1)
    with patcher, patch(_TENACITY_TIME) as fake_time:
        items = artifact_registry.get_by_where({"tags": ["a"]})

    assert [a.uri for a in items] == ["https://example.com/exact"]
    assert calls["n"] == 2  # one failure, then success
    assert fake_time.sleep.call_count == 1
