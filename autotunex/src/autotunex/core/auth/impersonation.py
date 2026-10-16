# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The impersonation overlay token: a signed, expiring pointer to a target user.

Separate from the session cookie (``core/auth/session.py``) by design — the real
login is never re-minted or mutated when an admin assumes another user. This token
carries only the target ``users.id``; the security decision "may this caller
impersonate at all" is made in ``get_effective_principal`` from the real
principal's ``is_admin``, not from this token, so a forged or copied token can only
take effect for a caller who is already an admin.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from autotunex.core.logging import get_logger

logger = get_logger(__name__)

_ALGORITHM = "HS256"
_LEEWAY_SECONDS = 30
"""Clock-skew tolerance, matching ``core/auth/session.py``."""


def mint_assume_token(target_user_id: UUID, *, secret: str, ttl_hours: int) -> str:
    """Return a signed overlay token naming ``target_user_id``, valid for ``ttl_hours``."""
    now = datetime.now(UTC)
    claims = {
        "sub": str(target_user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=ttl_hours)).timestamp()),
    }
    return jwt.encode(claims, secret, algorithm=_ALGORITHM)


def read_assume_token(token: str, *, secret: str) -> UUID | None:
    """Return the target user id, or ``None`` for any invalid/expired/malformed token.

    Fail-closed: every failure path returns ``None`` so the caller falls back to
    "not impersonating". ``algorithms`` is passed explicitly and singly, never
    inferred from the token header. Only ``type(exc).__name__`` is ever logged —
    never the token or PyJWT's own text.
    """
    try:
        claims = jwt.decode(
            token,
            key=secret,
            algorithms=[_ALGORITHM],
            leeway=_LEEWAY_SECONDS,
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as exc:
        logger.info("Assume cookie rejected: %s.", type(exc).__name__)
        return None

    sub = claims.get("sub")
    if not isinstance(sub, str):
        logger.warning("Assume cookie rejected: subject claim is missing or not a string.")
        return None
    try:
        return UUID(sub)
    except ValueError:
        logger.warning("Assume cookie rejected: subject is not a UUID.")
        return None
