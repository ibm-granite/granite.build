# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Authentication seam Protocols.

``Authenticator`` is the single entry point a request calls; it dispatches to
whichever ``CredentialVerifier`` matches the credential kind present.
Verifiers never touch the database — see the design spec's routing rules —
which is what keeps them unit-testable with no network and no session.
"""

from __future__ import annotations

from typing import Any, Protocol

from autotunex.models.auth import Principal


class Authenticator(Protocol):
    """Turns a request's raw credentials into a :class:`Principal`."""

    async def authenticate(
        self, *, bearer: str | None, api_key: str | None, session: str | None
    ) -> Principal:
        """Return the caller's identity, or raise an authentication error."""
        ...


class CredentialVerifier(Protocol):
    """Validates one credential kind (bearer, API key, or session cookie)."""

    name: str

    async def verify(self, credential: str) -> Principal:
        """Return the identity the credential proves, or raise."""
        ...


class SigningKeyResolver(Protocol):
    """Resolves the key material that verifies a bearer token's signature.

    Injected rather than constructed inside the verifier — the seam that lets
    tests sign real tokens with an in-process keypair and stub this instead of
    reaching a network.
    """

    async def resolve_signing_key(self, token: str) -> Any:  # noqa: ANN401
        """Return whatever ``jwt.decode``'s ``key=`` argument expects for ``token``.

        Typed ``Any`` because the concrete key type (a PEM string, or a
        ``cryptography`` public key object) is PyJWT's business, not a domain
        shape this codebase defines.
        """
        ...
