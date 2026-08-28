"""Unit tests for the ownership-scope resolver shared by every service."""

from __future__ import annotations

from uuid import uuid4

import pytest

from autotunex.core.exceptions import ScopeNotPermittedError
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.services.scoping import resolve_owner_filter, sees_nothing

_ADMIN = Principal(email="a@example.com", provider="session", user_id=uuid4(), is_admin=True)
_USER = Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)
_GHOST = Principal(email="g@example.com", provider="session", user_id=None, is_admin=False)


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
