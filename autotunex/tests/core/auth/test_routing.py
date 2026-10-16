"""RoutingAuthenticator's four dispatch rules.

Each credential kind has its own transport (Authorization: Bearer, X-API-Key,
a session cookie), so routing is deterministic — no shape-sniffing, no
provider-order sensitivity.
"""

from __future__ import annotations

import logging

import pytest

from autotunex.core.auth.routing import RoutingAuthenticator
from autotunex.core.exceptions import (
    ConflictingCredentialsError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from autotunex.models.auth import Principal


class FakeVerifier:
    """A verifier that accepts exactly one credential string."""

    name = "fake"

    def __init__(self, accepts: str, principal: Principal) -> None:
        self._accepts = accepts
        self._principal = principal

    async def verify(self, credential: str) -> Principal:
        if credential != self._accepts:
            raise InvalidCredentialsError()
        return self._principal


BEARER_PRINCIPAL = Principal(email="bearer@example.com", provider="oidc")
API_KEY_PRINCIPAL = Principal(email="key@example.com", provider="api_key")
SESSION_PRINCIPAL = Principal(email="session@example.com", provider="session")


@pytest.fixture
def authenticator() -> RoutingAuthenticator:
    return RoutingAuthenticator(
        bearer_verifier=FakeVerifier("good-bearer", BEARER_PRINCIPAL),
        api_key_verifier=FakeVerifier("good-key", API_KEY_PRINCIPAL),
        session_verifier=FakeVerifier("good-session", SESSION_PRINCIPAL),
    )


async def test_bearer_and_api_key_together_is_a_conflict(
    authenticator: RoutingAuthenticator,
) -> None:
    with pytest.raises(ConflictingCredentialsError):
        await authenticator.authenticate(bearer="good-bearer", api_key="good-key", session=None)


async def test_bearer_alone_is_routed_to_the_bearer_verifier(
    authenticator: RoutingAuthenticator,
) -> None:
    principal = await authenticator.authenticate(bearer="good-bearer", api_key=None, session=None)

    assert principal == BEARER_PRINCIPAL


async def test_api_key_alone_is_routed_to_the_api_key_verifier(
    authenticator: RoutingAuthenticator,
) -> None:
    principal = await authenticator.authenticate(bearer=None, api_key="good-key", session=None)

    assert principal == API_KEY_PRINCIPAL


async def test_explicit_bearer_beats_an_ambient_session_cookie(
    authenticator: RoutingAuthenticator,
) -> None:
    """A browser attaches cookies with no intent behind them; a bearer is explicit."""
    principal = await authenticator.authenticate(
        bearer="good-bearer", api_key=None, session="good-session"
    )

    assert principal == BEARER_PRINCIPAL


async def test_explicit_api_key_beats_an_ambient_session_cookie(
    authenticator: RoutingAuthenticator,
) -> None:
    """Explicit API key takes priority over ambient session cookie."""
    principal = await authenticator.authenticate(
        bearer=None, api_key="good-key", session="good-session"
    )

    assert principal == API_KEY_PRINCIPAL


async def test_a_cookie_with_no_explicit_credential_is_routed_to_the_session_verifier(
    authenticator: RoutingAuthenticator,
) -> None:
    principal = await authenticator.authenticate(bearer=None, api_key=None, session="good-session")

    assert principal == SESSION_PRINCIPAL


async def test_nothing_presented_is_missing_not_invalid(
    authenticator: RoutingAuthenticator,
) -> None:
    with pytest.raises(MissingCredentialsError):
        await authenticator.authenticate(bearer=None, api_key=None, session=None)


async def test_a_credential_routed_to_an_unregistered_verifier_is_invalid_not_missing() -> None:
    """Same 401 as a wrong credential — naming which schemes are off is a leak."""
    authenticator = RoutingAuthenticator()  # no verifiers registered at all

    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer="anything", api_key=None, session=None)


async def test_a_wrong_bearer_is_invalid(authenticator: RoutingAuthenticator) -> None:
    with pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer="wrong", api_key=None, session=None)


async def test_a_missing_credential_is_logged_at_warning(
    authenticator: RoutingAuthenticator, caplog: pytest.LogCaptureFixture
) -> None:
    """Design spec §5: an opaque client-facing detail requires a logged reason."""
    with caplog.at_level("WARNING"), pytest.raises(MissingCredentialsError):
        await authenticator.authenticate(bearer=None, api_key=None, session=None)

    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "no credential" in caplog.text


async def test_conflicting_credentials_are_logged_at_warning(
    authenticator: RoutingAuthenticator, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"), pytest.raises(ConflictingCredentialsError):
        await authenticator.authenticate(bearer="good-bearer", api_key="good-key", session=None)

    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "both" in caplog.text


async def test_an_unregistered_provider_logs_which_credential_kind_was_presented(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The one rejection an operator cannot otherwise diagnose.

    The caller deliberately gets the same 401 as a wrong credential, so "my valid
    API key returns 401" has no visible cause. Naming the kind in the log is what
    points at ``AUTOTUNEX_AUTH_PROVIDERS``.
    """
    authenticator = RoutingAuthenticator()  # no verifiers registered at all

    with caplog.at_level("WARNING"), pytest.raises(InvalidCredentialsError):
        await authenticator.authenticate(bearer=None, api_key="anything", session=None)

    assert "API key" in caplog.text


async def test_no_fragment_of_a_credential_ever_reaches_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The absolute rule, guarded across every rejection that holds a credential.

    An operator's diagnostic must never become a credential disclosure in a log
    file. Only the credential *kind* is loggable, and every kind label is a
    literal chosen inside ``routing.py`` rather than read out of the request — so
    this stays true no matter what a caller sends.

    Captured at DEBUG, not WARNING: the rule is absolute across every level, and
    a WARNING ceiling would let a future ``logger.debug(..., credential)`` inside
    ``_verify`` slip a credential into the log file without this test noticing.
    """
    secret = "sup3r-s3cret-credential-value"
    authenticator = RoutingAuthenticator()

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConflictingCredentialsError):
            await authenticator.authenticate(bearer=secret, api_key=secret, session=None)
        with pytest.raises(InvalidCredentialsError):
            await authenticator.authenticate(bearer=secret, api_key=None, session=None)
        with pytest.raises(InvalidCredentialsError):
            await authenticator.authenticate(bearer=None, api_key=secret, session=None)
        with pytest.raises(InvalidCredentialsError):
            await authenticator.authenticate(bearer=None, api_key=None, session=secret)

    assert len(caplog.records) == 4
    assert secret not in caplog.text
