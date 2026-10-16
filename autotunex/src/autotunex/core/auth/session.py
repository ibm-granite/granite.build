# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The BFF's own session token: minted and verified here, no server-side store.

Stateless by construction — no dict, no cache, nothing that would disagree
between two uvicorn workers. ``HS256`` is sufficient because both mint and
verify happen inside this one process with a shared secret; there is no
third party that ever needs to verify this token.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt

from autotunex.core.exceptions import ExpiredCredentialsError, InvalidCredentialsError
from autotunex.core.logging import get_logger
from autotunex.models.auth import Principal

logger = get_logger(__name__)

_ALGORITHM = "HS256"

_LEEWAY_SECONDS = 30
"""Clock-skew tolerance for ``iat``/``exp`` checks, matching ``oidc_leeway_seconds``'s default.

PyJWT validates ``iat`` by default (``verify_iat`` defaults to ``True``) and
raises ``ImmatureSignatureError`` when ``iat > now + leeway``. With no
leeway at all, a session cookie minted on one host and verified moments
later on another — the exact multi-worker scenario this module's docstring
frames statelessness around — would be rejected as "not yet valid" on
nothing more than ordinary clock drift between the two. 30 seconds is
generous for that while staying far short of anything that could matter for
``exp``: the gap between a real ``exp`` and "now" at rejection time is
measured in hours, set by ``session_ttl_hours``, never seconds.
"""


def mint_session_token(*, email: str, secret: str, ttl_hours: int) -> str:
    """Return a signed session token for ``email``, valid for ``ttl_hours``."""
    now = datetime.now(UTC)
    claims = {
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=ttl_hours)).timestamp()),
    }
    return jwt.encode(claims, secret, algorithm=_ALGORITHM)


class SessionCookieVerifier:
    """Satisfies :class:`autotunex.core.auth.protocols.CredentialVerifier`.

    ``algorithms`` is passed explicitly and singly to ``jwt.decode`` — never
    inferred from the token's own header — for the same reason
    ``OidcBearerVerifier`` does: it is what makes ``alg: none`` and
    HS256/RS256 key-confusion attacks structurally impossible.
    """

    name = "session"

    def __init__(self, secret: str) -> None:
        self._secret = secret

    async def verify(self, credential: str) -> Principal:
        """Verify the session JWT's signature and expiry, then read the email.

        Every rejection is logged, matching the discipline ``routing.py``
        established: it logs only the "no verifier registered" case, so each
        verifier — this one included — is responsible for logging its own.
        The split is deliberate, not merged into one level: expiry is
        routine and high-volume and also the one *non-opaque* rejection (the
        ``WWW-Authenticate`` challenge already tells the client to refresh),
        so it is logged at INFO; every other rejection — bad signature,
        malformed token, a missing/empty/whitespace-only/non-string ``email``
        claim — is logged at WARNING.

        Nothing derived from ``credential`` reaches the log: not the cookie
        value or any fragment of it, and not PyJWT's own exception text
        either, since ``PyJWKClient`` has shipped attacker-controlled text
        into a log line before (see ``oidc.py``'s ``_log_safe``). Only
        ``type(exc).__name__`` — library-controlled, not attacker-controlled
        — is logged, to give an operator a failure mode without that risk.
        """
        try:
            claims = jwt.decode(
                credential,
                key=self._secret,
                algorithms=[_ALGORITHM],
                leeway=_LEEWAY_SECONDS,
                options={"require": ["exp"]},
            )
        # A sibling of InvalidCredentialsError, not a subclass, and caught
        # before the general PyJWTError clause below (of which it is a
        # subclass) — order here is load-bearing, not incidental.
        except jwt.ExpiredSignatureError:
            logger.info("Session cookie rejected: expired.")
            raise ExpiredCredentialsError() from None
        except jwt.PyJWTError as exc:
            logger.warning("Session cookie rejected: %s.", type(exc).__name__)
            raise InvalidCredentialsError() from None

        email = claims.get("email")
        # Stripped, not merely checked non-empty: a whitespace-only claim
        # (`"   "`) is truthy and would otherwise authenticate as an email
        # that is not actually one. The stripped value is also what gets
        # returned — not the raw claim — so a padded claim never leaks
        # leading/trailing whitespace into `Principal.email`.
        email = email.strip() if isinstance(email, str) else None
        if not email:
            logger.warning("Session cookie rejected: missing or invalid email claim.")
            raise InvalidCredentialsError()
        return Principal(email=email, provider="session")
