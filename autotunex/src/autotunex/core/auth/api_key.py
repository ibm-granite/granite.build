# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""API key verifier, for machine callers (monitors, CI, the tuning pipeline).

Deliberately opt-in and never a fallback. Keys map to a real user's email, not
a synthetic identity — attribution (``jobs.user_id``) depends on it.
"""

from __future__ import annotations

import hashlib
import hmac

from autotunex.core.exceptions import InvalidCredentialsError
from autotunex.core.logging import get_logger
from autotunex.models.auth import Principal

logger = get_logger(__name__)


class ApiKeyVerifier:
    """Satisfies :class:`autotunex.core.auth.protocols.CredentialVerifier`.

    ``keys`` maps a SHA-256 hex digest to the owner's email. Uses
    :func:`hmac.compare_digest` for each check to prevent timing attacks
    based on byte-by-byte comparison. A miss is always constant-time
    across all configured digests, but a valid key reveals its own index
    — harmless since only someone with a valid key learns their position,
    and an attacker without one always sees the full iteration count on a
    SHA-256 digest (no incremental prefix search possible).
    """

    name = "api_key"

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = keys

    async def verify(self, credential: str) -> Principal:
        """Return the mapped owner's email, or raise if no digest matches.

        Logs one WARNING on rejection, matching the discipline
        ``routing.py`` established for every other credential kind: this
        used to be the one rejection with no diagnostic anywhere (see the
        module docstring there), because it happens inside this verifier
        rather than in the router that already logs. No fragment of
        ``credential`` — and no digest of it either — is ever logged; only
        this fixed, literal message is.
        """
        presented = hashlib.sha256(credential.encode("utf-8")).hexdigest()
        for digest, email in self._keys.items():
            if hmac.compare_digest(presented, digest):
                return Principal(email=email, provider="api_key")
        logger.warning(
            "Rejecting an API key: it matches none of the configured digests. "
            "The caller sees the same 401 as a malformed credential. Check "
            "AUTOTUNEX_API_KEYS."
        )
        raise InvalidCredentialsError()
