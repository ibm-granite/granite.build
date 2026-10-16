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

"""Unit tests for StorageSpaceAccessManager.get_user_spaces_with_access().

Storage is mocked; no DB required. Covers two properties raised in review of
PR #302: a storage error must propagate (not resolve to an empty list, which
scope_space_name_filter() would otherwise turn into a silent deny-all), and
the per-membership space lookup must be a single batched query rather than
one get_by_name() round trip per membership.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbserver.spaces.storage_space_access_manager import StorageSpaceAccessManager
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.constants import PUBLIC_SPACE_NAME

_EMAIL = "alice@example.com"


def _memberships(*names_and_roles: tuple[str, str]) -> list[StoredSpaceUser]:
    return [
        StoredSpaceUser(space_name=name, username=_EMAIL, role=role)
        for name, role in names_and_roles
    ]


def _patched_storage(space_user_storage, space_storage):
    fake_storage = SimpleNamespace(
        space_user_storage=space_user_storage, space_storage=space_storage
    )
    return patch(
        "gbserver.spaces.storage_space_access_manager.get_admin_storage",
        return_value=fake_storage,
    )


def test_get_user_spaces_with_access_propagates_storage_error():
    """A real DB failure must surface as an exception, not resolve to []
    (which the list-endpoint scoping in api/utils.py would otherwise treat
    as "deny all", masking an outage as an empty/zero-count result)."""
    space_user_storage = SimpleNamespace(
        get_by_username=MagicMock(side_effect=RuntimeError("db unavailable"))
    )
    space_storage = SimpleNamespace(get_by_where=MagicMock(), get_by_name=MagicMock())
    with _patched_storage(space_user_storage, space_storage):
        with pytest.raises(RuntimeError, match="db unavailable"):
            StorageSpaceAccessManager().get_user_spaces_with_access(_EMAIL)


def test_get_user_spaces_with_access_batches_space_lookup():
    """One get_by_where() call for all memberships' spaces, not one
    get_by_name() call per membership."""
    space_a = StoredSpace(name="space-a", git_repo_uri="")
    space_b = StoredSpace(name="space-b", git_repo_uri="")
    space_user_storage = SimpleNamespace(
        get_by_username=MagicMock(
            return_value=_memberships(("space-a", "admin"), ("space-b", "member"))
        )
    )
    get_by_where = MagicMock(return_value=[space_a, space_b])
    # No public space in storage -- the fallback get_by_name() call should
    # still happen exactly once, for PUBLIC_SPACE_NAME only.
    get_by_name = MagicMock(return_value=None)
    space_storage = SimpleNamespace(get_by_where=get_by_where, get_by_name=get_by_name)

    with _patched_storage(space_user_storage, space_storage):
        result = StorageSpaceAccessManager().get_user_spaces_with_access(_EMAIL)

    get_by_where.assert_called_once_with(where={"name": ["space-a", "space-b"]})
    get_by_name.assert_called_once_with(PUBLIC_SPACE_NAME)
    by_name = {r.space.name: r for r in result}
    assert by_name["space-a"].is_admin is True
    assert by_name["space-b"].is_admin is False


def test_get_user_spaces_with_access_no_memberships_skips_batch_query():
    """Zero memberships -> the batched lookup is skipped entirely rather than
    run with an empty name list (a wasted column.in_([]) round trip on this
    now-hot-path method for every list/count/tags request); only the
    public-space fallback can still add a row."""
    space_user_storage = SimpleNamespace(get_by_username=MagicMock(return_value=[]))
    get_by_where = MagicMock(return_value=[])
    get_by_name = MagicMock(return_value=None)
    space_storage = SimpleNamespace(get_by_where=get_by_where, get_by_name=get_by_name)

    with _patched_storage(space_user_storage, space_storage):
        result = StorageSpaceAccessManager().get_user_spaces_with_access(_EMAIL)

    get_by_where.assert_not_called()
    get_by_name.assert_called_once_with(PUBLIC_SPACE_NAME)
    assert result == []
