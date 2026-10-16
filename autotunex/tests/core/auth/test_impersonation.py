from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt

from autotunex.core.auth.impersonation import mint_assume_token, read_assume_token

_SECRET = "x" * 32


def test_mint_then_read_round_trips_the_target_id() -> None:
    target = uuid4()

    token = mint_assume_token(target, secret=_SECRET, ttl_hours=8)

    assert read_assume_token(token, secret=_SECRET) == target


def test_read_rejects_a_token_signed_with_a_different_secret() -> None:
    token = mint_assume_token(uuid4(), secret=_SECRET, ttl_hours=8)

    assert read_assume_token(token, secret="y" * 32) is None


def test_read_rejects_a_tampered_token() -> None:
    token = mint_assume_token(uuid4(), secret=_SECRET, ttl_hours=8)

    assert read_assume_token(token + "tamper", secret=_SECRET) is None


def test_read_rejects_an_expired_token() -> None:
    token = mint_assume_token(uuid4(), secret=_SECRET, ttl_hours=-1)

    assert read_assume_token(token, secret=_SECRET) is None


def test_read_rejects_a_non_uuid_subject() -> None:
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": "not-a-uuid",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        },
        _SECRET,
        algorithm="HS256",
    )

    assert read_assume_token(token, secret=_SECRET) is None
