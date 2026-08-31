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

"""Unit tests for scope_space_name_filter().

GET /builds/, /artifacts/, /spaces/ and their /count and /tags variants build
their row_filter straight from query params and call storage.get_by_where()
with no per-row authorization check at all -- unlike the single-object GET
routes, which load the row and then call confirm_space_member_access /
confirm_space_write_access on it. scope_space_name_filter() is what closes
that gap: its return value is used as the space_name filter passed into
get_row_filter() by every one of those list routes, except when it returns
NO_ACCESSIBLE_SPACE -- callers check for that with `is` and return an empty
result directly, without ever calling get_row_filter()/storage.get_by_where().

test/conftest.py's autouse `_mock_space_access` fixture stubs
gbserver.api.utils.is_super_admin to an unconditional True in mock mode,
which would make every non-admin case here trivially pass regardless of the
fix under test. Each test overrides it explicitly.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbserver.api import artifacts as artifacts_module
from gbserver.api import builds as builds_module
from gbserver.api import spaces as spaces_module
from gbserver.api.artifacts import list_artifacts
from gbserver.api.builds import count_builds, list_builds
from gbserver.api.spaces import list_spaces
from gbserver.api.utils import NO_ACCESSIBLE_SPACE, scope_space_name_filter
from gbserver.spaces.space_access_manager import SpaceAccessInfo
from gbserver.storage.stored_space import StoredSpace

SPACE_A = "space-A"
SPACE_B = "space-B"


def _fake_request(email: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(data={"user": SimpleNamespace(email=email)})
    )


def _access(*names: str) -> list[SpaceAccessInfo]:
    return [
        SpaceAccessInfo(space=StoredSpace(name=name, git_repo_uri=""), is_admin=False)
        for name in names
    ]


def test_super_admin_passes_through_requested_filter_unchanged():
    with (
        patch("gbserver.api.utils.is_super_admin", return_value=True),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        assert scope_space_name_filter(_fake_request("admin@example.com"), "") == ""
        assert (
            scope_space_name_filter(_fake_request("admin@example.com"), SPACE_B)
            == SPACE_B
        )
    # A super admin never needs the accessible-spaces lookup at all.
    manager.assert_not_called()


def test_non_admin_with_no_requested_space_gets_full_accessible_list():
    with (
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        manager.return_value.get_user_spaces_with_access.return_value = _access(
            SPACE_A, SPACE_B
        )
        result = scope_space_name_filter(_fake_request("alice@example.com"), "")
    assert sorted(result) == [SPACE_A, SPACE_B]


def test_non_admin_requesting_accessible_space_gets_it_back():
    with (
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        manager.return_value.get_user_spaces_with_access.return_value = _access(
            SPACE_A, SPACE_B
        )
        result = scope_space_name_filter(_fake_request("alice@example.com"), SPACE_A)
    assert result == [SPACE_A]


def test_non_admin_requesting_inaccessible_space_gets_sentinel():
    """This is the actual leak this function closes: alice is only a member of
    space-A, but the caller-supplied space_name=space-B must not be honored --
    it must resolve to the NO_ACCESSIBLE_SPACE marker, which callers check for
    with `is` and turn into an empty result themselves, rather than an empty
    list (which get_row_filter() treats as "no filter", i.e. everything)."""
    with (
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        manager.return_value.get_user_spaces_with_access.return_value = _access(SPACE_A)
        result = scope_space_name_filter(_fake_request("alice@example.com"), SPACE_B)
    assert result is NO_ACCESSIBLE_SPACE


def test_non_admin_with_no_accessible_spaces_gets_sentinel_not_empty_list():
    """An empty accessible set must not resolve to an empty list -- get_row_filter()
    drops falsy-length list values entirely, which would silently disable the
    space_name filter and return every row across every tenant."""
    with (
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        manager.return_value.get_user_spaces_with_access.return_value = []
        result = scope_space_name_filter(_fake_request("alice@example.com"), "")
    assert result is NO_ACCESSIBLE_SPACE


# Every list-style route that scopes via scope_space_name_filter(), and how to
# drive + verify it. `space_kwarg` is the route's own query-param name for the
# space filter (list_spaces calls it `name`, not `space_name`). `get_empty`
# extracts the collection/count from the response to compare against `empty`.
# Adding a NEW route here is the reason this table exists: forgetting the
# NO_ACCESSIBLE_SPACE check (see _NoAccessibleSpace's docstring in
# api/utils.py) would make this test fail loudly instead of the omission
# going unnoticed.
_SCOPED_ROUTES = [
    pytest.param(
        builds_module,
        list_builds,
        "space_name",
        lambda resp: resp.builds,
        [],
        id="list_builds",
    ),
    pytest.param(
        builds_module,
        count_builds,
        "space_name",
        lambda resp: resp.count,
        0,
        id="count_builds",
    ),
    pytest.param(
        artifacts_module,
        list_artifacts,
        "space_name",
        lambda resp: resp.artifacts,
        [],
        id="list_artifacts",
    ),
    pytest.param(
        spaces_module,
        list_spaces,
        "name",
        lambda resp: resp.spaces,
        [],
        id="list_spaces",
    ),
]


@pytest.mark.parametrize("module, route, space_kwarg, get_empty, empty", _SCOPED_ROUTES)
def test_all_scoped_routes_short_circuit_on_no_accessible_space(
    module, route, space_kwarg, get_empty, empty
):
    """Every route in _SCOPED_ROUTES must check for NO_ACCESSIBLE_SPACE and
    return its empty response WITHOUT ever calling get_admin_storage() --
    i.e. without reaching get_row_filter()/storage.get_by_where()/count()."""
    storage = MagicMock(name="get_admin_storage")
    with (
        patch.object(
            module, "scope_space_name_filter", return_value=NO_ACCESSIBLE_SPACE
        ),
        patch.object(module, "get_admin_storage", storage),
    ):
        resp = route(_fake_request("alice@example.com"), **{space_kwarg: "space-B"})
    assert get_empty(resp) == empty
    storage.assert_not_called()
