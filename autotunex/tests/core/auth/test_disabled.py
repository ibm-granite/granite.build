"""Standalone mode: a permissive implementation of the auth seam, not a bypass.

Two distinct rules, per the design spec: unset email means fully unrestricted
regardless of ``standalone_role``; a set email means the *role setting* wins
over whatever the database row says, once stage two resolves it.
"""

from __future__ import annotations

from autotunex.core.auth.disabled import (
    STANDALONE_PROVIDER,
    SYSTEM_STANDALONE_EMAIL,
    DisabledAuthenticator,
)
from autotunex.core.auth.protocols import Authenticator
from autotunex.core.config import Settings


async def test_unset_email_yields_the_default_system_owner() -> None:
    authenticator: Authenticator = DisabledAuthenticator(
        Settings(environment="test", standalone_email=None)
    )

    principal = await authenticator.authenticate(bearer=None, api_key=None, session=None)

    assert principal.email == SYSTEM_STANDALONE_EMAIL
    assert principal.is_admin is True  # default standalone_role is admin
    assert principal.provider == STANDALONE_PROVIDER


async def test_unset_email_with_user_role_yields_a_scoped_system_owner() -> None:
    authenticator = DisabledAuthenticator(
        Settings(environment="test", standalone_email=None, standalone_role="user")
    )

    principal = await authenticator.authenticate(bearer=None, api_key=None, session=None)

    assert principal.email == SYSTEM_STANDALONE_EMAIL
    assert principal.is_admin is False


async def test_set_email_with_admin_role_yields_an_admin_principal() -> None:
    authenticator = DisabledAuthenticator(
        Settings(environment="test", standalone_email="dev@example.com", standalone_role="admin")
    )

    principal = await authenticator.authenticate(bearer=None, api_key=None, session=None)

    assert principal.email == "dev@example.com"
    assert principal.is_admin is True


async def test_set_email_with_user_role_yields_a_non_admin_principal() -> None:
    authenticator = DisabledAuthenticator(
        Settings(environment="test", standalone_email="dev@example.com", standalone_role="user")
    )

    principal = await authenticator.authenticate(bearer=None, api_key=None, session=None)

    assert principal.is_admin is False


async def test_ignores_whatever_credentials_are_passed() -> None:
    authenticator = DisabledAuthenticator(Settings(environment="test"))

    principal = await authenticator.authenticate(
        bearer="anything", api_key="anything", session="anything"
    )

    assert principal.provider == STANDALONE_PROVIDER
