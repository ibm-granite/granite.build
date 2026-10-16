"""Authentication over the real ASGI stack, with API keys as the live provider.

A module-local ``settings`` fixture overrides the conftest default (which is
``auth_providers=["disabled"]``, where nothing can be rejected) so these tests
exercise a real verifier end to end, including through ``JobService`` scoping.
Overriding ``settings`` is enough: ``conftest``'s ``app`` fixture calls
``create_app(settings)``, so ``app.state.authenticator`` is built from *these*
settings, not the default ones.

The ghost key is registered here from the start rather than swapped in mid-test.
``app.state.authenticator`` is built once in ``create_app`` — deliberately, so
``PyJWKClient`` can cache signing keys on the instance in the OIDC phase — which
means it is not a dependency ``dependency_overrides`` can intercept, and a
mid-test settings swap has to rebuild it by hand. Declaring every key up front
avoids needing to.
"""

from __future__ import annotations

import hashlib
import logging
from http import HTTPStatus
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.config import Settings
from autotunex.db.tables import JobTable, UserTable
from tests.conftest import API, make_settings

ADMIN_KEY = "admin-service-key"
USER_KEY = "user-service-key"
GHOST_KEY = "ghost-service-key"


def _digest(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@pytest.fixture
def settings() -> Settings:
    """Overrides the conftest default to enable the API key provider.

    Goes through ``make_settings`` rather than building ``Settings`` directly —
    that factory is the only thing keeping a developer's ``.env`` and exported
    ``AUTOTUNEX_`` variables out of the suite, and a fresh ``Settings(...)`` here
    would reintroduce exactly the drift it was written to remove.
    """
    return make_settings(
        auth_providers=["api_key"],
        api_keys={
            _digest(ADMIN_KEY): "admin@example.com",
            _digest(USER_KEY): "tester@example.com",
            _digest(GHOST_KEY): "ghost@example.com",
        },
    )


async def test_no_credential_at_all_is_a_401(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs")

    assert response.status_code == HTTPStatus.UNAUTHORIZED


async def test_a_401_is_a_problem_detail_with_a_bearer_challenge(client: AsyncClient) -> None:
    """Spec §6: auth failures use the same RFC 9457 shape as every other error."""
    response = await client.get(f"{API}/jobs")

    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.headers["www-authenticate"].startswith('Bearer realm="autotunex"')
    body = response.json()
    assert body["status"] == HTTPStatus.UNAUTHORIZED
    assert body["detail"] == "Authentication is required."


async def test_health_is_reachable_with_no_credentials(client: AsyncClient) -> None:
    """A liveness probe cannot present one, so ``/health`` is exempt structurally."""
    response = await client.get("/health")

    assert response.status_code == HTTPStatus.OK


async def test_an_unmapped_key_is_a_401(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs", headers={"X-API-Key": "not-a-configured-key"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "The credential is not valid."


async def test_a_bearer_token_gets_the_same_401_as_an_invalid_key(client: AsyncClient) -> None:
    """The bearer verifier is not registered, and the response must not say so."""
    unmapped = await client.get(f"{API}/jobs", headers={"X-API-Key": "nope"})
    routed_to_disabled = await client.get(
        f"{API}/jobs", headers={"Authorization": "Bearer some.jwt.here"}
    )

    assert routed_to_disabled.status_code == unmapped.status_code
    assert routed_to_disabled.json() == unmapped.json()


async def test_a_bearer_header_with_no_token_reads_as_no_credential(client: AsyncClient) -> None:
    """Rescues the guarantee the deleted stage-one unit tests used to pin.

    A bare ``Bearer`` with nothing after it must not reach the bearer verifier
    as an empty credential — that would earn "invalid token" instead of the
    missing-credential challenge that tells the caller what to do. ``HTTPBearer``
    now owns the parsing (see ``bearer_scheme`` in ``deps.py``), so this has to
    be proven through the real header rather than a hand-built object.
    """
    response = await client.get(f"{API}/jobs", headers={"Authorization": "Bearer"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "Authentication is required."


async def test_an_empty_api_key_header_reads_as_no_credential(client: AsyncClient) -> None:
    """A present-but-empty ``X-API-Key`` must not reach the verifier as ``""``.

    ``APIKeyHeader.check_api_key`` (see ``api_key_scheme`` in ``deps.py``)
    normalizes a falsy header value to ``None`` before this ever reaches
    ``get_authenticated_principal``, so an empty header and an absent header
    must land on the identical missing-credential 401 — not "invalid token",
    which would need a verifier call and would tell the caller a credential was
    at least received. Asserting the detail, not just the status code, is the
    point: both outcomes are a 401, and a status-only assertion would not catch
    a regression that routed the empty string to the verifier instead.
    """
    response = await client.get(f"{API}/jobs", headers={"X-API-Key": ""})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "Authentication is required."


async def test_an_empty_session_cookie_reads_as_no_credential(client: AsyncClient) -> None:
    """The cookie transport gets the identical empty-reads-as-absent treatment.

    ``APIKeyCookie.check_api_key`` (see ``session_scheme`` in ``deps.py``) applies
    the same falsy check as the header scheme above, so a ``session`` cookie
    present with an empty value must not be routed to the session verifier as
    ``""`` — it must read as no credential at all, same as the header case.
    """
    response = await client.get(f"{API}/jobs", headers={"Cookie": "session="})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "Authentication is required."


async def test_a_non_empty_session_cookie_is_routed_to_the_session_slot(
    client: AsyncClient,
) -> None:
    """Finding 3: pins that ``session_scheme`` reads the ``session`` cookie by name.

    An absent cookie and an empty ``session=`` cookie (the test above) both
    produce the identical missing-credential 401, so neither can observe
    whether the cookie is actually being parsed — a misspelled
    ``APIKeyCookie(name="sesion")``, or dropping the ``session`` parameter
    from ``get_authenticated_principal`` altogether, would leave both of
    those tests passing unchanged. A non-empty value is only reachable if it
    was read and routed to the session slot, which this module's ``settings``
    fixture leaves with no registered verifier (only ``"api_key"`` is
    enabled) — so the discriminating signal is the *invalid-credential* 401,
    not the missing-credential one.
    """
    response = await client.get(f"{API}/jobs", headers={"Cookie": "session=abc"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "The credential is not valid."


async def test_a_whitespace_only_bearer_token_reads_as_no_credential(client: AsyncClient) -> None:
    """A fourth member of the empty-credential class: whitespace is not a token.

    FastAPI's ``get_authorization_scheme_param`` strips the token
    (``"Bearer   ".partition(" ")`` then ``.strip()``), so three trailing
    spaces and nothing else reduce to ``''`` — falsy, exactly like the bare
    ``"Bearer"`` case above — and ``HTTPBearer.__call__`` reads that as no
    credential rather than handing an empty string to the bearer verifier.
    The old ``not token`` guard in the deleted ``_bearer_token`` helper did not
    catch this case (``'  '`` is truthy), so this is new coverage, not a
    restatement of the first test in this block.
    """
    response = await client.get(f"{API}/jobs", headers={"Authorization": "Bearer   "})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "Authentication is required."


async def test_a_non_bearer_authorization_scheme_reads_as_no_credential(
    client: AsyncClient,
) -> None:
    """A ``Basic`` header is not a bearer token and must not be routed as one."""
    response = await client.get(f"{API}/jobs", headers={"Authorization": "Basic c29tZXRoaW5n"})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["detail"] == "Authentication is required."


async def test_a_well_formed_bearer_token_is_distinguished_from_no_credential_at_all(
    client: AsyncClient,
) -> None:
    """Proves parsing is happening at all, not just that two tests happen to agree.

    A well-formed ``Bearer`` token has no registered verifier, so it must fail as
    an *invalid* credential — a different body than the missing-credential 401
    the no-token and non-bearer cases above get. If ``HTTPBearer`` were somehow
    swallowing every ``Authorization`` value into ``None``, this would collapse
    to the same body as those two and the difference would go unnoticed.
    """
    no_token = await client.get(f"{API}/jobs", headers={"Authorization": "Bearer"})
    well_formed = await client.get(f"{API}/jobs", headers={"Authorization": "Bearer some.jwt.here"})

    assert well_formed.json()["detail"] == "The credential is not valid."
    assert well_formed.json() != no_token.json()


async def test_two_explicit_credentials_are_a_400(client: AsyncClient) -> None:
    response = await client.get(
        f"{API}/jobs",
        headers={"Authorization": "Bearer some.jwt.here", "X-API-Key": ADMIN_KEY},
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] == "Provide exactly one credential."


async def test_a_key_mapped_to_an_admin_sees_every_job(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    session.add(UserTable(id=uuid4(), email="admin@example.com", role="admin"))
    await session.commit()

    response = await client.get(f"{API}/jobs?scope=all", headers={"X-API-Key": ADMIN_KEY})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["total"] == 1


async def test_a_key_mapped_to_the_job_s_owner_sees_it(
    client: AsyncClient, job: JobTable, user: UserTable
) -> None:
    """``USER_KEY`` maps to ``tester@example.com``, which is the ``user`` fixture's email."""
    response = await client.get(f"{API}/jobs", headers={"X-API-Key": USER_KEY})

    assert response.json()["total"] == 1
    assert response.json()["items"][0]["id"] == str(job.id)


async def test_a_key_mapped_to_an_email_with_no_users_row_sees_an_empty_page(
    client: AsyncClient, job: JobTable
) -> None:
    """Spec decision 5: ``users`` is attribution, not an allowlist.

    ``ghost@example.com`` has no row, so this authenticates and then resolves to
    nothing — fails closed with an empty page, never 401 and never 403.
    """
    response = await client.get(f"{API}/jobs", headers={"X-API-Key": GHOST_KEY})

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_a_key_mapped_to_an_email_with_no_users_row_gets_404_not_403(
    client: AsyncClient, job: JobTable
) -> None:
    response = await client.get(f"{API}/jobs/{job.id}", headers={"X-API-Key": GHOST_KEY})

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_a_wrong_api_key_is_never_logged(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Spec §5: no fragment of any credential reaches a log record.

    The ``autotunex``-logger assertion is the non-vacuity check: ``httpx``
    itself logs an INFO line for every request/response, so a bare
    ``assert caplog.records`` would pass whether or not
    ``ApiKeyVerifier.verify`` logs anything at all — that httpx noise alone
    makes ``caplog.records`` non-empty regardless of this fix. Requiring a
    record from an ``autotunex`` logger specifically is what actually fails
    if the WARNING this finding adds is removed.
    ``test_a_wrong_api_key_s_rejection_is_logged_at_warning`` below pins the
    level; this test pins that whatever is produced stays clean.
    """
    secret = "a-key-that-must-never-appear-in-any-log-record"

    with caplog.at_level(logging.DEBUG):
        response = await client.get(f"{API}/jobs", headers={"X-API-Key": secret})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert any(record.name.startswith("autotunex") for record in caplog.records)
    assert secret not in caplog.text
    assert secret[:8] not in caplog.text


async def test_a_wrong_api_key_s_rejection_is_logged_at_warning(
    client: AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Finding 2: a wrong API key used to be the one rejection with no trace.

    ``RoutingAuthenticator._verify`` only logs when no verifier is
    registered; once ``"api_key"`` is enabled, a wrong key is rejected
    silently inside ``ApiKeyVerifier`` itself. Measured before this fix: a
    wrong ``X-API-Key`` produced zero log records while every other
    rejection kind produced one WARNING — the exact "my key stopped working
    after rotation" symptom the module docstring in ``routing.py`` says must
    not happen. This pins that the rejection is now logged at WARNING,
    without ever including a fragment of the presented key.
    """
    secret = "a-key-that-must-never-appear-in-any-log-record"

    with caplog.at_level(logging.DEBUG):
        response = await client.get(f"{API}/jobs", headers={"X-API-Key": secret})

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert secret not in caplog.text
    assert secret[:8] not in caplog.text


async def test_the_published_schema_advertises_all_three_credential_schemes(
    client: AsyncClient,
) -> None:
    """The whole point of Task 6: the schema — not runtime behaviour — gains this.

    ``deps.py``'s three module-level scheme instances are declared unconditionally,
    so they show up here regardless of which providers this app's settings enable.
    """
    response = await client.get("/openapi.json")

    schemes = response.json()["components"]["securitySchemes"]
    bearer = schemes["BearerAuth"]
    api_key = schemes["ApiKeyAuth"]
    session_cookie = schemes["SessionCookieAuth"]
    assert bearer["type"] == "http"
    assert bearer["scheme"] == "bearer"
    assert api_key["type"] == "apiKey"
    assert api_key["in"] == "header"
    assert api_key["name"] == "X-API-Key"
    assert session_cookie["type"] == "apiKey"
    assert session_cookie["in"] == "cookie"
    assert session_cookie["name"] == "session"


async def test_a_job_operation_carries_security_but_health_does_not(client: AsyncClient) -> None:
    """The assertion that would catch the schemes being declared but never wired.

    Advertising the three schemes under ``components.securitySchemes`` is not by
    itself proof that any operation requires one — that only happens because
    ``get_authenticated_principal`` (which every job route depends on, per
    ``test_route_coverage.py``) itself depends on the three scheme instances.
    ``/health`` has no such dependency and must carry none.
    """
    response = await client.get("/openapi.json")

    schema = response.json()
    jobs_security = schema["paths"][f"{API}/jobs"]["get"].get("security")
    health_security = schema["paths"]["/health"]["get"].get("security")

    assert jobs_security
    assert not health_security
