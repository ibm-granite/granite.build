# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Deterministic credential routing.

Each credential kind arrives on its own transport, so — unlike shape-sniffing
one bearer slot for every provider — dispatch never depends on registration
order and two opaque tokens are never confused with each other.

Every rejection here is logged at WARNING. That is the counterweight the design
spec §5 requires for the fixed, opaque details this module's errors return: the
caller learns nothing, so the operator has to learn everything, or a
misconfigured provider is undiagnosable from the outside.

**No fragment of any credential is ever logged.** Only the credential *kind* is,
and every kind label is a literal chosen in this module — never a value read out
of the request. Adding a ``%s`` for a token here would turn every 401 into a
credential disclosure in the log file.
"""

from __future__ import annotations

from autotunex.core.auth.protocols import CredentialVerifier
from autotunex.core.exceptions import (
    ConflictingCredentialsError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from autotunex.core.logging import get_logger
from autotunex.models.auth import Principal

logger = get_logger(__name__)


class RoutingAuthenticator:
    """Satisfies :class:`autotunex.core.auth.protocols.Authenticator`.

    A verifier left as ``None`` means that credential kind has no registered
    provider; presenting it then fails exactly like an invalid credential of
    that kind, so an attacker cannot learn which schemes are configured.
    """

    def __init__(
        self,
        *,
        bearer_verifier: CredentialVerifier | None = None,
        api_key_verifier: CredentialVerifier | None = None,
        session_verifier: CredentialVerifier | None = None,
    ) -> None:
        self._bearer_verifier = bearer_verifier
        self._api_key_verifier = api_key_verifier
        self._session_verifier = session_verifier

    async def authenticate(
        self, *, bearer: str | None, api_key: str | None, session: str | None
    ) -> Principal:
        """Dispatch to the matching verifier.

        Order: two explicit credentials conflict; either explicit credential
        beats an ambient session cookie; a cookie alone is routed to the
        session verifier; nothing is a missing-credentials error.
        """
        if bearer is not None and api_key is not None:
            logger.warning(
                "Rejecting a request presenting both a bearer token and an API key. "
                "Exactly one credential is allowed; the client is sending two."
            )
            raise ConflictingCredentialsError()
        if bearer is not None:
            return await self._verify(self._bearer_verifier, bearer, kind="bearer token")
        if api_key is not None:
            return await self._verify(self._api_key_verifier, api_key, kind="API key")
        if session is not None:
            return await self._verify(self._session_verifier, session, kind="session cookie")
        logger.warning("Rejecting a request that presented no credential at all.")
        raise MissingCredentialsError()

    @staticmethod
    async def _verify(
        verifier: CredentialVerifier | None, credential: str, *, kind: str
    ) -> Principal:
        """Hand ``credential`` to ``verifier``, or reject it as invalid.

        ``kind`` exists only to be logged, and this is the one rejection an
        operator cannot diagnose from the outside: the caller gets the same 401 as
        a genuinely wrong credential — naming which schemes are configured is a
        leak — so without the log the symptom is "my valid API key returns 401"
        with no cause anywhere. ``credential`` itself is never logged.
        """
        if verifier is None:
            logger.warning(
                "Rejecting a %s: no verifier is registered for that credential kind. "
                "The caller sees the same 401 as an invalid credential. Check "
                "AUTOTUNEX_AUTH_PROVIDERS.",
                kind,
            )
            raise InvalidCredentialsError()
        return await verifier.verify(credential)
