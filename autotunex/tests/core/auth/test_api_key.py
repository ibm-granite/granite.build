"""ApiKeyVerifier: digest match only, never a raw-key comparison.

The verifier never touches the database — a key mapped to an email with no
``users`` row still authenticates here; it is stage two's job (already built
in the seam-and-scoping phase) to resolve that email to nothing.
"""

from __future__ import annotations

import hashlib

import pytest

from autotunex.core.auth.api_key import ApiKeyVerifier
from autotunex.core.auth.protocols import CredentialVerifier
from autotunex.core.exceptions import InvalidCredentialsError


def _digest(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def test_a_key_matching_a_configured_digest_authenticates() -> None:
    verifier: CredentialVerifier = ApiKeyVerifier({_digest("good-key"): "svc@example.com"})

    principal = await verifier.verify("good-key")

    assert principal.email == "svc@example.com"
    assert principal.provider == "api_key"


async def test_a_key_not_matching_any_configured_digest_is_invalid() -> None:
    verifier = ApiKeyVerifier({_digest("good-key"): "svc@example.com"})

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify("wrong-key")


async def test_the_configured_digest_itself_is_never_accepted_as_a_key() -> None:
    """Guards against ever comparing the presented key directly to a stored digest."""
    digest = _digest("good-key")
    verifier = ApiKeyVerifier({digest: "svc@example.com"})

    with pytest.raises(InvalidCredentialsError):
        await verifier.verify(digest)


async def test_a_key_mapped_to_any_email_authenticates_regardless_of_whether_a_user_exists() -> (
    None
):
    """The verifier has no database access, so it cannot know or care."""
    verifier = ApiKeyVerifier({_digest("ghost-key"): "ghost@example.com"})

    principal = await verifier.verify("ghost-key")

    assert principal.email == "ghost@example.com"


def test_verifier_name_is_api_key() -> None:
    assert ApiKeyVerifier({}).name == "api_key"
