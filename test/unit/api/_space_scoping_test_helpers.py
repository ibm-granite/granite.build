#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the cross-space list-endpoint scoping tests split
across test_builds.py, test_artifacts.py, and test_spaces.py -- all three
exercise scope_space_name_filter() (api/utils.py) against a fake
row_filter-aware storage, so they share the same alice/bob space setup and
row-matching logic.
"""

from types import SimpleNamespace
from typing import Any

from gbserver.spaces.space_access_manager import SpaceAccessInfo
from gbserver.storage.stored_space import StoredSpace

ALICE = "alice"
ALICE_SPACE = "space-A"
BOB_SPACE = "space-B"


def fake_request(login: str, email: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(data={"user": SimpleNamespace(login=login, email=email)})
    )


def row_matches(item: Any, where: dict) -> bool:
    """Apply a row_filter dict the way get_by_where()'s real filter-building
    does: list/tuple/set values match-any (IN), scalar values match exactly."""
    for key, value in (where or {}).items():
        actual = getattr(item, key)
        if isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True


def set_alice_access(manager) -> None:
    """alice is a non-admin member of ALICE_SPACE only."""
    manager.return_value.get_user_spaces_with_access.return_value = [
        SpaceAccessInfo(
            space=StoredSpace(name=ALICE_SPACE, git_repo_uri=""), is_admin=False
        )
    ]
