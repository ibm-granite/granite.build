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

"""Unit tests for GET /spaces/ list scoping.

list_spaces built its row_filter straight from the optional `name` query
param and called storage.space_storage.get_by_where() with no authorization
check at all, so any authenticated user could enumerate every space in the
deployment -- names, git remotes, Lakehouse namespaces -- regardless of their
own membership. scope_space_name_filter() (api/utils.py) closes the gap.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from unit.api._space_scoping_test_helpers import (
    ALICE,
    ALICE_SPACE,
    BOB_SPACE,
)
from unit.api._space_scoping_test_helpers import fake_request as _fake_request
from unit.api._space_scoping_test_helpers import row_matches as _row_matches
from unit.api._space_scoping_test_helpers import set_alice_access as _set_alice_access

from gbserver.api import spaces as spaces_module
from gbserver.api.spaces import list_spaces, spaces_for_user
from gbserver.storage.stored_space import StoredSpace


def _patched_list_storage(spaces: list):
    """A get_admin_storage() stand-in whose space_storage.get_by_where()
    actually applies the row_filter, so the tests prove the name value
    computed by scope_space_name_filter() is what narrows the result set."""
    fake_storage = SimpleNamespace(
        space_storage=SimpleNamespace(
            get_by_where=lambda where=None: [
                s for s in spaces if _row_matches(s, where or {})
            ]
        )
    )
    return patch.object(spaces_module, "get_admin_storage", return_value=fake_storage)


def test_list_spaces_excludes_other_spaces_for_non_admin():
    alice_space = StoredSpace(name=ALICE_SPACE, git_repo_uri="git://internal/a.git")
    bob_space = StoredSpace(name=BOB_SPACE, git_repo_uri="git://internal/b.git")
    with (
        _patched_list_storage([alice_space, bob_space]),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        _set_alice_access(manager)
        resp = list_spaces(_fake_request(ALICE, f"{ALICE}@example.com"))
    assert {s.name for s in resp.spaces} == {ALICE_SPACE}


def test_list_spaces_rejects_explicit_cross_space_request():
    """An explicit name=space-B (bob's space, not alice's) must return
    nothing -- not bob's space, including its git remote."""
    bob_space = StoredSpace(name=BOB_SPACE, git_repo_uri="git://internal/b.git")
    with (
        _patched_list_storage([bob_space]),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        _set_alice_access(manager)
        resp = list_spaces(_fake_request(ALICE, f"{ALICE}@example.com"), name=BOB_SPACE)
    assert resp.spaces == []


def test_list_spaces_unrestricted_for_super_admin():
    alice_space = StoredSpace(name=ALICE_SPACE, git_repo_uri="git://internal/a.git")
    bob_space = StoredSpace(name=BOB_SPACE, git_repo_uri="git://internal/b.git")
    with (
        _patched_list_storage([alice_space, bob_space]),
        patch("gbserver.api.utils.is_super_admin", return_value=True),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        resp = list_spaces(_fake_request("admin_x", "admin_x@example.com"))
    assert {s.name for s in resp.spaces} == {ALICE_SPACE, BOB_SPACE}
    manager.assert_not_called()


def test_spaces_for_user_propagates_storage_error():
    """spaces_for_user() must not turn a real storage error into a 404 --
    there's no legitimate "not found" case for this route (a user with zero
    spaces just gets an empty list), so a broad except here can only ever
    mask a real failure behind a misleading status code."""
    with patch.object(
        spaces_module, "user_spaces_list", side_effect=RuntimeError("db unavailable")
    ):
        with pytest.raises(RuntimeError, match="db unavailable"):
            spaces_for_user(_fake_request(ALICE, f"{ALICE}@example.com"))
