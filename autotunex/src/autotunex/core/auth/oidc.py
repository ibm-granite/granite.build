# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""OIDC bearer-token verification (W3ID access or ID tokens).

Verification asserts signature, issuer, mandatory audience, expiry with
leeway, and an explicit algorithm allowlist — the allowlist is what makes
``alg: none`` and HS256/RS256 key-confusion attacks structurally impossible,
not merely untested.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

import jwt as pyjwt
from jwt import PyJWKClient

from autotunex.core.auth.protocols import SigningKeyResolver
from autotunex.core.exceptions import ExpiredCredentialsError, InvalidCredentialsError
from autotunex.core.logging import get_logger
from autotunex.models.auth import Principal

logger = get_logger(__name__)

_MAX_LOGGED_LIBRARY_TEXT: Final = 256
"""Ceiling on the escaped length of library text reaching the log.

PyJWT's own longest message is well under 100 characters, and a genuine W3ID
``kid`` is a short base64url thumbprint, so 256 is generous headroom for every
legitimate case while denying an attacker an unbounded write amplified by
``kid`` interpolation (see :func:`_log_safe`).
"""

_TRUNCATION_MARKER: Final = "...[truncated]"
"""Appended when text is cut, so a reader never mistakes a clipped message for a whole one."""


def _log_safe(text: str) -> str:
    r"""Neutralise library-supplied text before it reaches the log.

    Exists because PyJWT interpolates *unverified* token data into exception
    text that we log verbatim. The concrete case: ``PyJWKClient.get_signing_key``
    raises ``PyJWKClientError(f'Unable to find a signing key that matches:
    "{kid}"')`` (``jwt/jwks_client.py``), and that ``kid`` is read from the
    token's **unverified** header by ``get_signing_key_from_jwt`` — arbitrary
    attacker-chosen JSON with no type check and no length bound. Since
    ``core.logging`` formats records as a single unescaped line, a ``kid``
    containing a newline would forge log entries with no credential required
    (CWE-117).

    This is deliberately applied to *all* library text at the moment of
    logging, not to that one exception type: the audit that concluded "PyJWT's
    messages interpolate nothing" held for ``InvalidAudienceError`` /
    ``InvalidIssuerError`` / ``ExpiredSignatureError`` and was still wrong,
    because it generalised from a sample. A future version interpolating token
    data into a different message is contained by this function without anyone
    re-auditing. **Do not delete this as defensive noise.**

    Every non-printable character is escaped, which by ``str.isprintable()``
    covers the whole forging set — C0/C1 controls, ``\n``, ``\r``, U+0085 NEL
    and the U+2028/U+2029 separators Python's own ``splitlines`` honours —
    while leaving legitimate non-ASCII printable text intact (unlike
    ``unicode_escape``, which would mangle it). Escaping runs before
    truncation so the *escaped* length is what the bound applies to; otherwise
    a run of newlines would expand past it.
    """
    escaped = "".join(char if char.isprintable() else _escape_character(char) for char in text)
    if len(escaped) > _MAX_LOGGED_LIBRARY_TEXT:
        return escaped[:_MAX_LOGGED_LIBRARY_TEXT] + _TRUNCATION_MARKER
    return escaped


def _escape_character(char: str) -> str:
    """Render one non-printable character as a visible escape sequence."""
    code = ord(char)
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"


class JwksSigningKeyResolver:
    """Satisfies :class:`SigningKeyResolver`, backed by a JWKS endpoint.

    Must be constructed once and reused — ``PyJWKClient`` caches fetched keys
    on the instance, so per-request construction would make every request a
    JWKS fetch.
    """

    def __init__(self, jwks_uri: str) -> None:
        self._client = PyJWKClient(jwks_uri)

    async def resolve_signing_key(self, token: str) -> Any:  # noqa: ANN401
        """Return the key material for ``token``, off the event loop.

        ``get_signing_key_from_jwt`` is blocking ``urllib``; cached keys make
        this a cold-start and rotation cost only.
        """
        signing_key = await asyncio.to_thread(self._client.get_signing_key_from_jwt, token)
        return signing_key.key


class OidcBearerVerifier:
    """Satisfies :class:`autotunex.core.auth.protocols.CredentialVerifier` for W3ID bearer tokens.

    ``algorithms`` is always passed explicitly to ``jwt.decode`` — never
    inferred from the token's own header — which is what makes ``alg: none``
    and HS256/RS256 key-confusion attacks structurally impossible. ``audience``
    is required at construction: there is no way to build this verifier
    without one.
    """

    name = "oidc"

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        algorithms: list[str],
        email_claims: list[str],
        leeway_seconds: int,
        key_resolver: SigningKeyResolver,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._algorithms = algorithms
        self._email_claims = email_claims
        self._leeway_seconds = leeway_seconds
        self._key_resolver = key_resolver

    async def verify(self, credential: str) -> Principal:
        """Verify signature, issuer, audience and expiry, then read the email.

        No unverified value from ``credential`` reaches the log in a form it
        can control — until the signature checks out every claim is
        attacker-supplied. That holds *because of* :func:`_log_safe`, not
        because the library text happens to be safe: PyJWT interpolates the
        unverified ``kid`` into one of its own messages, so the text below is
        sanitised rather than trusted.

        Key resolution runs inside the same ``try`` as the decode call: the
        real ``JwksSigningKeyResolver`` fails on input an attacker fully
        controls — a malformed token, an unreachable IdP, and also a JWKS
        endpoint answering with non-JSON — and every one of those must become
        a 401 like any other rejection, not escape as an unhandled 500.

        After a failed verification, PyJWT's own exception text is logged
        (sanitised) for operators, at a level that differs by path: expiry at
        INFO, every other rejection at WARNING (see the handlers below for
        why). The caller only ever sees a fixed detail string.
        """
        try:
            key = await self._key_resolver.resolve_signing_key(credential)
            claims = pyjwt.decode(
                credential,
                key=key,
                algorithms=self._algorithms,
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway_seconds,
                # "exp" is the one entry `require` actually guards: PyJWT only
                # checks expiry when the claim is present, so without it here a
                # token minted with no `exp` at all would be accepted forever.
                # "iss" and "aud" are already enforced independently — PyJWT
                # rejects them as missing whenever `issuer=`/`audience=` are
                # passed, which they always are above — so listing them here is
                # deliberate defence-in-depth: it keeps "absent is rejected"
                # true even if either argument ever became optional.
                options={"require": ["exp", "iss", "aud"]},
            )
        # Caught before the general clause below, and kept a distinct exception:
        # `ExpiredCredentialsError` is a sibling of `InvalidCredentialsError` so
        # the WWW-Authenticate challenge tells a client to refresh.
        except pyjwt.ExpiredSignatureError as exc:
            # INFO, not WARNING, and deliberately so. Expiry is routine and
            # high-volume; a WARNING per expired token would dilute a channel
            # that otherwise carries only genuine attack signal (forged
            # signatures, audience probes). It is also the one *non-opaque*
            # rejection — the caller is told to refresh — so `routing.py`'s
            # "the caller learns nothing, so the operator must learn
            # everything" rationale does not apply here. It is still logged:
            # silence would make expiry the only rejection with no
            # server-side trace at all.
            logger.info("OIDC bearer token rejected as expired: %s", _log_safe(str(exc)))
            raise ExpiredCredentialsError() from None
        # `OSError` and `ValueError` are not redundant with `PyJWTError`, which
        # subclasses neither. `PyJWKClient.fetch_data` wraps only
        # `(URLError, TimeoutError)` around its `json.load(response)`
        # (`jwt/jwks_client.py`), so a JWKS endpoint answering `200 text/html`
        # — proxy interception, misrouted ingress, a captive portal — raises
        # `json.JSONDecodeError` (a `ValueError`), and a mid-body
        # `ConnectionResetError` (an `OSError`, and not a `URLError`) escapes
        # too. Both are IdP trouble or malformed input, so both belong on the
        # 401 path rather than the generic 500 handler. Not widened to bare
        # `Exception`: a genuine bug in this module must still surface as a 500.
        except (pyjwt.PyJWTError, OSError, ValueError) as exc:
            logger.warning("OIDC bearer token rejected: %s", _log_safe(str(exc)))
            raise InvalidCredentialsError() from None

        for claim in self._email_claims:
            email = claims.get(claim)
            if isinstance(email, str) and email:
                return Principal(email=email, provider="oidc")
        logger.warning(
            "OIDC bearer token has none of the configured email claims: %s", self._email_claims
        )
        raise InvalidCredentialsError()
