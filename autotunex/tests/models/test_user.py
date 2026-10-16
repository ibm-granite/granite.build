"""User schema validation: strict role input, lenient role output."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from autotunex.models.user import Role, UserMetadata, UserRead, UserRoleUpdate


def test_user_role_update_accepts_a_known_role() -> None:
    body = UserRoleUpdate(role="admin")

    assert body.role is Role.ADMIN


def test_user_role_update_rejects_an_unknown_role() -> None:
    with pytest.raises(ValidationError):
        UserRoleUpdate(role="root")


def test_user_role_update_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        UserRoleUpdate.model_validate({"role": "user", "email": "new@example.com"})


def test_user_read_tolerates_a_null_role() -> None:
    read = UserRead(
        id=uuid4(),
        email="a@example.com",
        role=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert read.role is None


def test_user_metadata_holds_three_counts() -> None:
    metadata = UserMetadata(number_of_jobs=3, number_of_configurations=2, number_of_datasets=1)

    assert (
        metadata.number_of_jobs,
        metadata.number_of_configurations,
        metadata.number_of_datasets,
    ) == (3, 2, 1)
