"""The session token this service mints and verifies itself — no server-side store.

Every rejection path is also checked for two things the brief's own draft was
silent on (see the controller rulings amending Task 2): that it is logged —
at INFO for expiry, WARNING for everything else — and that the logged text
never contains the credential itself.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from autotunex.core.auth.protocols import CredentialVerifier
from autotunex.core.auth.session import SessionCookieVerifier, mint_session_token
from autotunex.core.exceptions import ExpiredCredentialsError, InvalidCredentialsError

# 32+ characters, deliberately: PyJWT emits `InsecureKeyLengthWarning` below
# that for HS256 (RFC 7518 §3.2's own minimum), and a previously clean test
# suite growing a warnings block on every run is exactly the noise that
# trains reviewers to stop reading it. `Settings.session_secret` now enforces
# this same 32-character floor in production (see `core/config.py`); these
# constants keep the test doubles honest about the same bound.
SECRET = "test-session-secret-at-least-32-characters-long"
_A_DIFFERENT_SECRET = "a-different-secret-at-least-32-characters-long"


def _claims(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "email": "dev@example.com",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=8)).timestamp()),
    }
    base.update(overrides)
    return base


def _token(claims: dict[str, object], *, secret: str = SECRET) -> str:
    return jwt.encode(claims, secret, algorithm="HS256")


async def test_mint_then_verify_round_trips_the_email() -> None:
    token = mint_session_token(email="dev@example.com", secret=SECRET, ttl_hours=8)
    verifier = SessionCookieVerifier(SECRET)

    principal = await verifier.verify(token)

    assert principal.email == "dev@example.com"
    assert principal.provider == "session"


async def test_an_expired_session_token_raises_expired_not_invalid() -> None:
    # Built via the `_claims`/`_token` helpers directly, not decode-then-re-sign
    # through `mint_session_token`: that dance was dead weight once these
    # helpers existed, and it made this test's arrange phase depend on
    # `mint_session_token` succeeding for no reason the test cares about.
    token = _token(_claims(exp=0))
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(ExpiredCredentialsError):
        await verifier.verify(token)


async def test_a_token_signed_with_a_different_secret_is_rejected() -> None:
    token = mint_session_token(email="dev@example.com", secret=_A_DIFFERENT_SECRET, ttl_hours=8)
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_malformed_token_is_rejected() -> None:
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify("not-a-jwt-at-all")


async def test_a_token_with_no_expiry_claim_at_all_is_rejected() -> None:
    """Proactive coverage for the mutation the plan calls out explicitly.

    PyJWT only enforces expiry when ``exp`` is present at all — there is no
    independent check that fires on its *absence* the way there is for, say,
    ``iss``/``aud`` under a non-``None`` ``issuer=``/``audience=`` argument
    (see ``oidc.py``'s equivalent test). Without ``options={"require": ["exp"]}``
    in ``SessionCookieVerifier.verify``, a token minted with no expiry at all
    would be accepted and would never expire — a session that lives forever.
    """
    claims = _claims()
    del claims["exp"]
    token = _token(claims)
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_with_no_email_claim_at_all_is_rejected() -> None:
    claims = _claims()
    del claims["email"]
    token = _token(claims)
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_with_an_empty_email_claim_is_rejected() -> None:
    token = _token(_claims(email=""))
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_with_a_non_string_email_claim_is_rejected() -> None:
    token = _token(_claims(email=12345))
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_a_token_with_a_whitespace_only_email_claim_is_rejected() -> None:
    """``"   "`` is truthy, so a bare ``not email`` check would authenticate it.

    No real email is ever whitespace, so this closes the same class of gap
    ``_is_unset`` closes for settings — see ``core/config.py``.
    """
    token = _token(_claims(email="   "))
    verifier = SessionCookieVerifier(SECRET)

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)


async def test_an_email_claim_padded_with_whitespace_is_stripped() -> None:
    """The stripped value is what ends up on ``Principal``, not the raw claim."""
    token = _token(_claims(email="  dev@example.com  "))
    verifier = SessionCookieVerifier(SECRET)

    principal = await verifier.verify(token)

    assert principal.email == "dev@example.com"


def test_verifier_name_is_session() -> None:
    verifier: CredentialVerifier = SessionCookieVerifier(SECRET)

    assert verifier.name == "session"


async def test_mint_sets_exp_exactly_ttl_hours_after_iat() -> None:
    """The TTL is the module's one piece of arithmetic and its one security-relevant number.

    Every other test here mints with the default ``ttl_hours=8`` and never
    inspects the resulting ``exp``, so a mutation that silently ignored
    ``ttl_hours`` (hard-coding ``8``) or mis-scaled it (``timedelta(days=...)``
    instead of ``hours``) would pass every other test in this file. Decoding
    with the real secret and a real algorithm — not merely asserting on the
    opaque token string — is what actually pins the arithmetic.
    """
    token = mint_session_token(email="dev@example.com", secret=SECRET, ttl_hours=1)

    claims = jwt.decode(token, key=SECRET, algorithms=["HS256"], options={"require": ["exp"]})

    assert claims["exp"] - claims["iat"] == 3600


async def test_an_expired_session_token_is_logged_at_info_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expiry must leave a trace but must not spend the WARNING channel.

    The negative half (not WARNING) is load-bearing: a regression that logged
    expiry at WARNING would still satisfy a bare "something was logged"
    check while diluting the channel that otherwise carries genuine attack
    signal — the same reasoning ``oidc.py`` documents for its own expiry path.
    """
    token = _token(_claims(exp=0))
    verifier = SessionCookieVerifier(SECRET)

    with caplog.at_level(logging.DEBUG), pytest.raises(ExpiredCredentialsError):
        await verifier.verify(token)

    ours = [record for record in caplog.records if record.name.startswith("autotunex")]
    assert [record.levelno for record in ours] == [logging.INFO]


async def test_a_wrong_secret_rejection_is_logged_at_warning_and_never_leaks_the_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = mint_session_token(email="dev@example.com", secret=_A_DIFFERENT_SECRET, ttl_hours=8)
    verifier = SessionCookieVerifier(SECRET)

    with caplog.at_level(logging.DEBUG), pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)

    ours = [record for record in caplog.records if record.name.startswith("autotunex")]
    assert any(record.levelno == logging.WARNING for record in ours)
    logged = "\n".join(record.getMessage() for record in ours)
    assert token not in logged


async def test_a_malformed_token_rejection_is_logged_at_warning_and_never_leaks_the_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    credential = "not-a-jwt-at-all"
    verifier = SessionCookieVerifier(SECRET)

    with caplog.at_level(logging.DEBUG), pytest.raises(InvalidCredentialsError):
        await verifier.verify(credential)

    ours = [record for record in caplog.records if record.name.startswith("autotunex")]
    assert any(record.levelno == logging.WARNING for record in ours)
    logged = "\n".join(record.getMessage() for record in ours)
    assert credential not in logged


async def test_a_missing_email_claim_rejection_is_logged_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    claims = _claims()
    del claims["email"]
    token = _token(claims)
    verifier = SessionCookieVerifier(SECRET)

    with caplog.at_level(logging.DEBUG), pytest.raises(InvalidCredentialsError):
        await verifier.verify(token)

    ours = [record for record in caplog.records if record.name.startswith("autotunex")]
    assert any(record.levelno == logging.WARNING for record in ours)
