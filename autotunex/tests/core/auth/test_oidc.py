"""OIDC bearer verification.

No network call anywhere in this file: signing uses an in-process RSA
keypair, and the one test that touches ``JwksSigningKeyResolver`` monkeypatches
PyJWT's own client method rather than reaching a real JWKS endpoint.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWKClient

from autotunex.core.auth.oidc import JwksSigningKeyResolver, OidcBearerVerifier
from autotunex.core.exceptions import ExpiredCredentialsError, InvalidCredentialsError

ISSUER = "https://idp.example.com/oauth2"
AUDIENCE = "test-client-id"


async def test_jwks_resolver_wraps_the_blocking_pyjwt_call_in_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the resolver offloads blocking PyJWT calls to a thread.

    Asserting only on the return value and the recorded call is not enough:
    a version that called ``get_signing_key_from_jwt`` directly on the event
    loop, dropping ``asyncio.to_thread`` entirely, would produce the exact
    same return value and the exact same recorded call. What actually proves
    the offload is that the stub runs on a *different* thread than the test
    body — the event-loop thread never blocks on it.
    """
    calls: list[str] = []
    call_thread_ids: list[int] = []
    test_thread_id = threading.get_ident()

    class FakeSigningKey:
        key = "the-resolved-key-object"

    def fake_get_signing_key_from_jwt(self: PyJWKClient, token: str) -> FakeSigningKey:
        calls.append(token)
        call_thread_ids.append(threading.get_ident())
        return FakeSigningKey()

    monkeypatch.setattr(PyJWKClient, "get_signing_key_from_jwt", fake_get_signing_key_from_jwt)
    resolver = JwksSigningKeyResolver("https://example.com/.well-known/jwks.json")

    key = await resolver.resolve_signing_key("a-token")

    assert key == "the-resolved-key-object"
    assert calls == ["a-token"]
    assert call_thread_ids != [test_thread_id]


def _rsa_keypair() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


class _FakeResolver:
    """Always resolves to one fixed key, regardless of the token presented."""

    def __init__(self, key: object) -> None:
        self._key = key

    async def resolve_signing_key(self, token: str) -> object:
        return self._key


class _RaisingResolver:
    """Fakes what the real ``JwksSigningKeyResolver`` does when resolution itself fails.

    Every other fake in this module resolves successfully, which is precisely
    why a bug that lets resolver exceptions escape ``verify`` uncaught could
    slip past this suite. This fake reproduces the two shapes PyJWT's real
    resolver raises on attacker-controlled input: a malformed token
    (``DecodeError``) and an unreachable IdP (``PyJWKClientConnectionError``).
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def resolve_signing_key(self, token: str) -> object:
        raise self._exc


def _claims(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "email": "dev@example.com",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    base.update(overrides)
    return base


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _forge_hs256_token(claims: dict[str, object], secret: bytes) -> str:
    """Hand-roll an HS256-signed JWT, bypassing ``jwt.encode``'s own guard.

    PyJWT 2.13's ``HMACAlgorithm.prepare_key`` refuses to sign with a key that
    looks like a PEM-encoded asymmetric key — ``jwt.encode(..., pem,
    algorithm="HS256")`` raises ``InvalidKeyError`` outright, so the classic
    RS256/HS256 key-confusion attack can no longer be constructed through
    PyJWT's own encoder. A real attacker isn't limited to that encoder, so the
    test must not be either: this builds the token exactly as one would by
    hand, HMAC-signing the header/payload with the RSA public key's PEM bytes
    as the secret. What is actually under test is unaffected by this either
    way — ``OidcBearerVerifier`` must reject it because ``algorithms=["RS256"]``
    never permits ``alg: HS256``, before the signature is ever inspected.
    """
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = (
        f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}."
        f"{_b64url(json.dumps(claims, separators=(',', ':')).encode())}"
    )
    signature = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _verifier(key: object) -> OidcBearerVerifier:
    return OidcBearerVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=["RS256"],
        email_claims=["email", "emailAddress"],
        leeway_seconds=30,
        key_resolver=_FakeResolver(key),
    )


@pytest.fixture(scope="module")
def keypair() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    return _rsa_keypair()


async def test_a_valid_token_authenticates(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    token = jwt.encode(_claims(), private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    principal = await verifier.verify(token)

    assert principal.email == "dev@example.com"
    assert principal.provider == "oidc"


async def test_emailaddress_is_used_when_email_is_absent(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    claims = _claims()
    del claims["email"]
    claims["emailAddress"] = "dev2@example.com"
    token = jwt.encode(claims, private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    principal = await verifier.verify(token)

    assert principal.email == "dev2@example.com"


async def test_a_token_with_no_email_claim_at_all_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    claims = _claims()
    del claims["email"]
    token = jwt.encode(claims, private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_an_expired_token_is_expired_not_merely_invalid(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    expired_at = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())
    token = jwt.encode(_claims(exp=expired_at), private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(ExpiredCredentialsError):
        await verifier.verify(token)


async def test_a_wrong_issuer_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    token = jwt.encode(
        _claims(iss="https://not-the-configured-issuer.example.com"), private_key, algorithm="RS256"
    )
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_wrong_audience_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    token = jwt.encode(_claims(aud="a-different-client"), private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_an_audience_array_containing_ours_is_accepted(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    token = jwt.encode(
        _claims(aud=[AUDIENCE, "some-other-audience"]), private_key, algorithm="RS256"
    )
    verifier = _verifier(public_key)

    principal = await verifier.verify(token)

    assert principal.email == "dev@example.com"


async def test_a_token_with_no_audience_claim_at_all_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    """``Audience is mandatory`` has to mean absent, not merely wrong.

    Unlike ``exp`` below, this is *not* guarded by ``options={"require": [...]}``
    alone. With a non-``None`` ``audience=`` argument — which
    ``OidcBearerVerifier`` always passes — PyJWT raises
    ``MissingRequiredClaimError("aud")`` for an absent *or* falsy ``aud`` before
    ``require`` is ever consulted, so ``audience=`` by itself already rejects
    this token. Listing ``"aud"`` in ``require`` is defence-in-depth on top of
    that, not the thing doing the work. A token that cannot be scoped to this
    API is a signal to distrust — spec §2.1.
    """
    private_key, public_key = keypair
    claims = _claims()
    del claims["aud"]
    token = jwt.encode(claims, private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_with_no_issuer_claim_at_all_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, public_key = keypair
    claims = _claims()
    del claims["iss"]
    token = jwt.encode(claims, private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_with_no_expiry_claim_at_all_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    """This is the one claim ``require`` actually guards, unlike ``iss``/``aud``.

    PyJWT only enforces expiry when the ``exp`` claim is present — there is no
    independent check that fires on its *absence*, the way there is for
    ``iss``/``aud`` via the non-``None`` ``issuer=``/``audience=`` arguments.
    Without ``"exp"`` in ``options={"require": [...]}``, a token minted with no
    expiry at all would be accepted and would never expire.
    """
    private_key, public_key = keypair
    claims = _claims()
    del claims["exp"]
    token = jwt.encode(claims, private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_that_is_not_yet_valid_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    """``nbf`` is half of spec §2.1's "exp/nbf with leeway" and is otherwise untested.

    The offset is well beyond ``leeway_seconds=30``, so this cannot pass on skew.
    """
    private_key, public_key = keypair
    not_before = int((datetime.now(UTC) + timedelta(minutes=10)).timestamp())
    token = jwt.encode(_claims(nbf=not_before), private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_inside_the_leeway_window_is_still_accepted(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    """``oidc_leeway_seconds`` is a real tolerance, not a decorative setting."""
    private_key, public_key = keypair
    just_expired = int((datetime.now(UTC) - timedelta(seconds=5)).timestamp())
    token = jwt.encode(_claims(exp=just_expired), private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    principal = await verifier.verify(token)

    assert principal.email == "dev@example.com"


async def test_a_signature_from_an_unrelated_key_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    _, public_key = keypair
    unrelated_private_key, _ = _rsa_keypair()
    forged = jwt.encode(_claims(), unrelated_private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(forged)


async def test_hs256_using_the_rsa_public_key_as_an_hmac_secret_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    """The classic RS256/HS256 key-confusion attack. algorithms= is the only guard."""
    _, public_key = keypair
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    forged = _forge_hs256_token(_claims(), pem)
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(forged)


async def test_alg_none_is_rejected(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    _, public_key = keypair
    forged = jwt.encode(_claims(), None, algorithm="none")  # type: ignore[arg-type]
    verifier = _verifier(public_key)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(forged)


async def test_no_library_exception_text_reaches_the_client(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real PyJWT reason is logged server-side; the client gets a fixed string.

    The configured issuer is checked against the ``WWW-Authenticate`` challenge,
    not against ``detail``: pinning ``detail`` to a literal already proves the
    issuer is absent from it, so asserting that separately could never fail. The
    challenge is the part that could plausibly grow a dynamic hint — an
    ``error_description`` naming the expected issuer would be an easy "helpful"
    addition, and would hand a prober the value they are searching for.
    """
    private_key, public_key = keypair
    token = jwt.encode(
        _claims(iss="https://not-the-configured-issuer.example.com"), private_key, algorithm="RS256"
    )
    verifier = _verifier(public_key)

    with caplog.at_level(logging.WARNING), pytest.raises(InvalidCredentialsError) as exc_info:
        await verifier.verify(token)

    assert exc_info.value.detail == "The credential is not valid."
    headers = exc_info.value.headers
    assert headers is not None, "a 401 must carry a WWW-Authenticate challenge"
    assert ISSUER not in headers["WWW-Authenticate"]
    assert any(
        record.name.startswith("autotunex") and record.levelno == logging.WARNING
        for record in caplog.records
    )


async def test_a_malformed_token_the_resolver_cannot_parse_is_rejected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression test: the resolver call must run inside the ``try`` block.

    Mirrors what the real ``JwksSigningKeyResolver`` does when handed a
    credential with too few segments: ``PyJWKClient.get_signing_key_from_jwt``
    raises ``jwt.DecodeError`` before ``jwt.decode`` is ever reached. If the
    resolver call sits outside ``verify``'s ``try`` block, this exception
    escapes uncaught and would hit the global 500 handler instead of
    producing a 401 — a real bug, not a hypothetical one.
    """
    verifier = OidcBearerVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=["RS256"],
        email_claims=["email"],
        leeway_seconds=30,
        key_resolver=_RaisingResolver(jwt.DecodeError("Not enough segments")),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(InvalidCredentialsError) as exc_info:
        await verifier.verify("this-is-not-a-jwt")

    assert exc_info.value.detail == "The credential is not valid."
    assert any(
        record.name.startswith("autotunex") and record.levelno == logging.WARNING
        for record in caplog.records
    )


async def test_an_unreachable_idp_during_key_resolution_is_rejected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression test: the resolver call must run inside the ``try`` block.

    Mirrors what the real ``JwksSigningKeyResolver`` does when the JWKS host
    cannot be reached: ``PyJWKClientConnectionError`` is a ``PyJWTError``
    subclass, so it is handled by the same ``except`` clause as decode-time
    failures — but only once the resolver call is inside the ``try``.
    """
    verifier = OidcBearerVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=["RS256"],
        email_claims=["email"],
        leeway_seconds=30,
        key_resolver=_RaisingResolver(jwt.PyJWKClientConnectionError("connection refused")),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(InvalidCredentialsError) as exc_info:
        await verifier.verify("well-formed-but-idp-is-unreachable")

    assert exc_info.value.detail == "The credential is not valid."
    assert any(
        record.name.startswith("autotunex") and record.levelno == logging.WARNING
        for record in caplog.records
    )


async def test_a_jwks_endpoint_returning_non_json_is_rejected_not_a_500(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ``ValueError`` from key resolution is IdP trouble, so it must be a 401.

    ``PyJWKClient.fetch_data`` wraps only ``(URLError, TimeoutError)`` around
    its ``json.load(response)`` (``jwt/jwks_client.py``), so a JWKS endpoint
    answering ``200 text/html`` — proxy interception, misrouted ingress, a
    captive portal — raises ``json.JSONDecodeError``. That is a ``ValueError``,
    **not** a ``PyJWTError``, so before the ``except`` clause was widened it
    escaped ``verify`` and reached the generic 500 handler. Malformed input and
    IdP trouble are a 401, never a 500.
    """
    verifier = OidcBearerVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=["RS256"],
        email_claims=["email"],
        leeway_seconds=30,
        key_resolver=_RaisingResolver(
            json.JSONDecodeError("Expecting value", "<html>not json</html>", 0)
        ),
    )

    with caplog.at_level(logging.WARNING), pytest.raises(InvalidCredentialsError) as exc_info:
        await verifier.verify("well-formed-but-jwks-returned-html")

    assert exc_info.value.detail == "The credential is not valid."
    assert any(
        record.name.startswith("autotunex") and record.levelno == logging.WARNING
        for record in caplog.records
    )


async def test_a_connection_reset_during_key_resolution_is_rejected_not_a_500() -> None:
    """The ``OSError`` half of the same widening.

    A reset arriving mid-body escapes PyJWT's own ``(URLError, TimeoutError)``
    handler — ``ConnectionResetError`` is an ``OSError`` and is not a
    ``URLError`` — so it too would otherwise reach the 500 handler.
    """
    verifier = OidcBearerVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=["RS256"],
        email_claims=["email"],
        leeway_seconds=30,
        key_resolver=_RaisingResolver(ConnectionResetError("Connection reset by peer")),
    )

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify("well-formed-but-the-idp-hung-up")


async def test_an_expired_token_is_logged_at_info_not_warning(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expiry must leave a server-side trace, but must not spend the WARNING channel.

    Two things are pinned. First, that expiry is logged at all: it was
    previously the *only* rejection with no trace anywhere, while both
    ``SECURITY.md`` and ``verify``'s docstring claimed every failure mode was
    logged. Second, that it is INFO — the negative assertion is the load-bearing
    half, since a regression to WARNING would satisfy a bare "something was
    logged" check while diluting the channel that otherwise carries only
    genuine attack signal.
    """
    private_key, public_key = keypair
    expired = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())
    token = jwt.encode(_claims(exp=expired), private_key, algorithm="RS256")
    verifier = _verifier(public_key)

    with caplog.at_level(logging.DEBUG), pytest.raises(ExpiredCredentialsError):
        await verifier.verify(token)

    ours = [record for record in caplog.records if record.name.startswith("autotunex")]
    assert [record.levelno for record in ours] == [logging.INFO]
    assert "expired" in ours[0].getMessage()


def _jwk_set_for(public_key: rsa.RSAPublicKey, *, kid: str) -> dict[str, object]:
    """Build a minimal JWKS document advertising ``public_key`` under ``kid``."""
    numbers = public_key.public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": kid,
                "n": _b64url(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
                "e": _b64url(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
            }
        ]
    }


async def test_a_newline_in_the_unverified_kid_cannot_forge_a_log_line(
    keypair: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CWE-117 regression, through the real ``PyJWKClient`` interpolation path.

    This does not fake the exception. ``PyJWKClient.get_signing_key`` really
    does raise ``PyJWKClientError(f'Unable to find a signing key that matches:
    "{kid}"')``, and that ``kid`` really is read from the token's *unverified*
    header by ``get_signing_key_from_jwt`` — so an unauthenticated caller
    chooses it freely. Only ``fetch_data`` is stubbed, which keeps the real
    lookup-miss-refresh-miss-raise path intact with no network.

    Because ``core.logging`` writes one unescaped line per record, an
    unsanitised ``%s`` on that exception lets a newline in ``kid`` forge log
    entries with **no credential required**. The assertions are therefore about
    the shape of what was logged, not merely that something was: no record may
    contain a line break, and the forged marker must not appear as a line of
    its own. The last assertion pins the other half of the requirement — the
    diagnostic text is *sanitised, not discarded*, because "which check failed"
    is what an operator needs.
    """
    private_key, public_key = keypair
    forged = 'no-such-kid"\n2026-08-01 00:00:00 CRITICAL autotunex: FORGED-ADMIN-GRANT'
    token = jwt.encode(_claims(), private_key, algorithm="RS256", headers={"kid": forged})
    monkeypatch.setattr(
        PyJWKClient, "fetch_data", lambda self: _jwk_set_for(public_key, kid="the-genuine-kid")
    )
    verifier = OidcBearerVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=["RS256"],
        email_claims=["email"],
        leeway_seconds=30,
        key_resolver=JwksSigningKeyResolver("https://unused.invalid/jwks"),
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)

    ours = [record for record in caplog.records if record.name.startswith("autotunex")]
    assert ours, "the rejection must still be logged for operators"
    assert not any("\n" in record.getMessage() or "\r" in record.getMessage() for record in ours)
    logged = "\n".join(record.getMessage() for record in ours)
    assert "FORGED-ADMIN-GRANT" not in logged.splitlines()
    assert "Unable to find a signing key" in logged


def test_verifier_name_is_oidc() -> None:
    assert (
        OidcBearerVerifier(
            issuer=ISSUER,
            audience=AUDIENCE,
            algorithms=["RS256"],
            email_claims=["email"],
            leeway_seconds=30,
            key_resolver=_FakeResolver(None),
        ).name
        == "oidc"
    )
