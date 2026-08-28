from __future__ import annotations

from uuid import uuid4

from autotunex.models.auth import Principal


def test_principal_defaults_impersonator_to_none() -> None:
    principal = Principal(email="admin@example.com", provider="session")

    assert principal.impersonator is None


def test_principal_carries_the_impersonator_when_impersonating() -> None:
    principal = Principal(
        email="target@example.com",
        provider="session",
        user_id=uuid4(),
        is_admin=True,
        impersonator="admin@example.com",
    )

    assert principal.impersonator == "admin@example.com"
    assert principal.is_admin is True
