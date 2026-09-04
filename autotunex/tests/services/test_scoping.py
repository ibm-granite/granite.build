"""Unit tests for the ownership-scope resolver shared by every service."""

from __future__ import annotations

from uuid import uuid4

import pytest

from autotunex.core.constants import SYSTEM_USER_ID
from autotunex.core.exceptions import ScopeNotPermittedError
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.services.scoping import (
    is_delete_protected,
    is_system_owned,
    resolve_owner_filter,
    sees_nothing,
)

_ADMIN = Principal(email="a@example.com", provider="session", user_id=uuid4(), is_admin=True)
_USER = Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)
_GHOST = Principal(email="g@example.com", provider="session", user_id=None, is_admin=False)
_SYSTEM = Principal(
    email="system@autotunex.local", provider="standalone", user_id=SYSTEM_USER_ID, is_admin=True
)
_ASSUMING_SYSTEM = Principal(
    email="system@autotunex.local",
    provider="session",
    user_id=SYSTEM_USER_ID,
    is_admin=True,
    impersonator="a@example.com",
)


def test_own_scope_returns_the_callers_own_id_for_a_provisioned_user() -> None:
    assert resolve_owner_filter(_USER, DataScope.OWN) == _USER.user_id


def test_own_scope_returns_the_callers_own_id_even_for_an_admin() -> None:
    assert resolve_owner_filter(_ADMIN, DataScope.OWN) == _ADMIN.user_id


def test_all_scope_returns_none_for_an_admin() -> None:
    assert resolve_owner_filter(_ADMIN, DataScope.ALL) is None


def test_all_scope_is_forbidden_for_a_non_admin() -> None:
    with pytest.raises(ScopeNotPermittedError):
        resolve_owner_filter(_USER, DataScope.ALL)


def test_all_scope_is_forbidden_for_an_unprovisioned_caller() -> None:
    with pytest.raises(ScopeNotPermittedError):
        resolve_owner_filter(_GHOST, DataScope.ALL)


def test_sees_nothing_is_true_only_for_own_scope_without_an_id() -> None:
    assert sees_nothing(_GHOST, DataScope.OWN) is True
    assert sees_nothing(_USER, DataScope.OWN) is False
    assert sees_nothing(_ADMIN, DataScope.OWN) is False


# Delete-protection of the shared system tier.


def test_is_system_owned_matches_the_reserved_id_as_a_string_or_uuid() -> None:
    assert is_system_owned(str(SYSTEM_USER_ID)) is True
    assert is_system_owned(SYSTEM_USER_ID) is True


def test_is_system_owned_is_false_for_another_owner_or_no_owner() -> None:
    assert is_system_owned(str(uuid4())) is False
    assert is_system_owned(None) is False


def test_a_system_owned_row_is_delete_protected_from_a_normal_user() -> None:
    assert is_delete_protected(_USER, str(SYSTEM_USER_ID)) is True


def test_a_system_owned_row_is_delete_protected_from_an_admin() -> None:
    assert is_delete_protected(_ADMIN, str(SYSTEM_USER_ID)) is True


def test_a_system_owned_row_is_delete_protected_from_the_system_user_itself() -> None:
    # The single-owner standalone deployment: being the system owner ambiently is
    # what made "any user can delete the starter configs" true, so it is refused.
    assert is_delete_protected(_SYSTEM, str(SYSTEM_USER_ID)) is True


def test_a_system_owned_row_is_deletable_while_impersonating_the_system_user() -> None:
    assert is_delete_protected(_ASSUMING_SYSTEM, str(SYSTEM_USER_ID)) is False


def test_an_ordinary_row_is_never_delete_protected() -> None:
    assert is_delete_protected(_USER, str(_USER.user_id)) is False
    assert is_delete_protected(_ASSUMING_SYSTEM, str(_USER.user_id)) is False
