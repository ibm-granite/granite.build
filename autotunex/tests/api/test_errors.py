"""The single error-translation layer, including auth-specific header forwarding."""

from __future__ import annotations

from http import HTTPStatus

from httpx import ASGITransport, AsyncClient

from autotunex.core.exceptions import (
    AuthenticationError,
    ConflictingCredentialsError,
    ExpiredCredentialsError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from autotunex.main import create_app
from tests.conftest import make_settings


def test_missing_credentials_error_is_a_401_with_a_bearer_challenge() -> None:
    error = MissingCredentialsError()

    assert error.status_code == HTTPStatus.UNAUTHORIZED
    assert error.headers is not None
    assert error.headers["WWW-Authenticate"].startswith('Bearer realm="autotunex"')


def test_invalid_credentials_error_names_invalid_token() -> None:
    error = InvalidCredentialsError()

    assert 'error="invalid_token"' in (error.headers or {})["WWW-Authenticate"]


def test_expired_credentials_error_is_not_an_invalid_credentials_error() -> None:
    """Siblings, not parent/child — see this plan's divergence note 2.

    A subclass would make ``pytest.raises(InvalidCredentialsError)`` pass on an
    expired token, erasing the very distinction the spec introduces the split
    to preserve.
    """
    error = ExpiredCredentialsError()

    assert isinstance(error, AuthenticationError)
    assert not isinstance(error, InvalidCredentialsError)
    assert "expired" in error.detail


def test_expired_credentials_error_tells_a_client_to_refresh() -> None:
    error = ExpiredCredentialsError()

    challenge = (error.headers or {})["WWW-Authenticate"]

    assert 'error="invalid_token"' in challenge
    assert "expired" in challenge


def test_conflicting_credentials_error_is_a_400_with_no_header() -> None:
    error = ConflictingCredentialsError()

    assert error.status_code == HTTPStatus.BAD_REQUEST
    assert error.headers is None


async def test_a_domain_error_s_headers_reach_the_http_response() -> None:
    app = create_app(make_settings())

    @app.get("/__raises_missing_credentials")
    async def _raise() -> None:
        raise MissingCredentialsError()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/__raises_missing_credentials")

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.headers["www-authenticate"].startswith('Bearer realm="autotunex"')
    assert response.headers["content-type"].startswith("application/problem+json")
