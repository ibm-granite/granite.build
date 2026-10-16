"""Case-insensitive email lookup.

MySQL's default collation is case-insensitive; SQLite and Postgres are not
(``docs/schema-review.md`` §E, "Cross-dialect portability"). Lowering both
sides at the query makes all three dialects agree, at the cost of the
``email`` index — not a real cost on a ``users`` table.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.core.exceptions import AmbiguousIdentityError
from autotunex.db.repositories.protocols import UserRepository
from autotunex.db.repositories.sqlalchemy import SqlAlchemyUserRepository
from autotunex.db.tables import UserTable


async def test_finds_a_user_by_exact_email(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="tester@example.com", role="admin")
    session.add(user)
    await session.commit()
    repository: UserRepository = SqlAlchemyUserRepository(session)

    found = await repository.get_by_email("tester@example.com")

    assert found is not None
    assert found.id == user.id


async def test_finds_a_user_regardless_of_email_casing(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="tester@example.com", role="admin")
    session.add(user)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    found = await repository.get_by_email("TESTER@EXAMPLE.COM")

    assert found is not None
    assert found.id == user.id


async def test_finds_a_user_whose_stored_email_is_not_lowercase(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="Tester@example.com", role="user")
    session.add(user)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    found = await repository.get_by_email("tester@example.com")

    assert found is not None
    assert found.id == user.id


async def test_returns_none_for_an_unknown_email(session: AsyncSession) -> None:
    repository = SqlAlchemyUserRepository(session)

    found = await repository.get_by_email("nobody@example.com")

    assert found is None


async def test_provision_creates_a_non_admin_user(session: AsyncSession) -> None:
    repository = SqlAlchemyUserRepository(session)

    provisioned = await repository.provision("fresh@example.com")

    assert provisioned.role == "user"
    assert (await repository.get_by_email("fresh@example.com")) is not None


async def test_provision_returns_the_existing_row_when_one_already_exists(
    session: AsyncSession,
) -> None:
    """The race path: a concurrent first request already inserted the row.

    Seeding the row first stands in for the winner of that race — ``provision``
    must return it rather than trip the ``UNIQUE(email)`` constraint, and must
    not create a duplicate.
    """
    existing = UserTable(id=uuid4(), email="racer@example.com", role="admin")
    session.add(existing)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    provisioned = await repository.provision("racer@example.com")

    assert provisioned.id == existing.id
    assert provisioned.role == "admin"


async def test_duplicate_case_variant_emails_raise_instead_of_picking_a_row(
    session: AsyncSession,
) -> None:
    """Ambiguous identity fails closed rather than guessing.

    ``users.email UNIQUE`` is case-sensitive on SQLite and Postgres, so these two
    rows coexist there and the case-insensitive lookup matches both. They carry
    different roles on purpose: a deterministic ``.first()`` tiebreak would settle
    admin-ness by row order and hide the underlying data bug.
    """
    session.add_all(
        [
            UserTable(id=uuid4(), email="Alice@example.com", role="admin"),
            UserTable(id=uuid4(), email="alice@example.com", role="user"),
        ]
    )
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    with pytest.raises(AmbiguousIdentityError):
        await repository.get_by_email("alice@example.com")


async def test_an_ambiguous_email_is_logged_at_warning_for_the_operator(
    session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """The email is the operator's only route to the offending rows.

    Safe to log because it arrives already verified by an ``Authenticator`` — the
    spec's no-logging rule covers unverified credentials, not a resolved
    identity. The client-facing detail says nothing about the duplication, so
    without this line the failure is undiagnosable.
    """
    session.add_all(
        [
            UserTable(id=uuid4(), email="Alice@example.com", role="admin"),
            UserTable(id=uuid4(), email="alice@example.com", role="user"),
        ]
    )
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    with caplog.at_level("WARNING"), pytest.raises(AmbiguousIdentityError):
        await repository.get_by_email("alice@example.com")

    assert "alice@example.com" in caplog.text
    assert [record.levelname for record in caplog.records] == ["WARNING"]


async def test_list_returns_users_newest_first_with_the_total(
    session: AsyncSession,
) -> None:
    older = UserTable(id=uuid4(), email="older@example.com", role="user")
    session.add(older)
    await session.commit()
    newer = UserTable(id=uuid4(), email="newer@example.com", role="admin")
    session.add(newer)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    users, total = await repository.list(limit=20, offset=0)

    assert total == 2
    assert [user.email for user in users] == ["newer@example.com", "older@example.com"]


async def test_list_respects_limit_and_offset(session: AsyncSession) -> None:
    for index in range(3):
        session.add(UserTable(id=uuid4(), email=f"u{index}@example.com", role="user"))
        await session.commit()
    repository = SqlAlchemyUserRepository(session)

    users, total = await repository.list(limit=1, offset=1)

    assert total == 3
    assert len(users) == 1


async def test_get_returns_the_user_by_id(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="a@example.com", role="user")
    session.add(user)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    found = await repository.get(user.id)

    assert found is not None
    assert found.id == user.id


async def test_get_returns_none_for_an_unknown_id(session: AsyncSession) -> None:
    repository = SqlAlchemyUserRepository(session)

    assert (await repository.get(uuid4())) is None


async def test_set_role_updates_and_returns_the_row(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="a@example.com", role="user")
    session.add(user)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    updated = await repository.set_role(user.id, "admin")

    assert updated is not None
    assert updated.role == "admin"
    refetched = await repository.get(user.id)
    assert refetched is not None
    assert refetched.role == "admin"


async def test_set_role_returns_none_for_an_unknown_id(session: AsyncSession) -> None:
    repository = SqlAlchemyUserRepository(session)

    assert (await repository.set_role(uuid4(), "admin")) is None


async def test_count_admins_counts_only_admins(session: AsyncSession) -> None:
    session.add_all(
        [
            UserTable(id=uuid4(), email="a1@example.com", role="admin"),
            UserTable(id=uuid4(), email="a2@example.com", role="admin"),
            UserTable(id=uuid4(), email="u1@example.com", role="user"),
        ]
    )
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    assert (await repository.count_admins()) == 2


async def test_metadata_counts_the_users_jobs_configs_and_datasets(
    session: AsyncSession,
    user: UserTable,
    configuration: object,
    dataset: object,
    job: object,
) -> None:
    # The conftest `job` fixture is owned by `user` and references `configuration`
    # and `dataset`, so the user owns exactly one of each.
    repository = SqlAlchemyUserRepository(session)

    jobs, configs, datasets = await repository.metadata(user.id)

    assert (jobs, configs, datasets) == (1, 1, 1)


async def test_touch_login_records_the_login_time(session: AsyncSession) -> None:
    user = UserTable(id=uuid4(), email="tester@example.com", role="user")
    session.add(user)
    await session.commit()
    repository: UserRepository = SqlAlchemyUserRepository(session)

    await repository.touch_login("tester@example.com")

    await session.refresh(user)
    assert user.last_login_at is not None


async def test_touch_login_advances_a_previously_recorded_login(session: AsyncSession) -> None:
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    user = UserTable(id=uuid4(), email="tester@example.com", role="user", last_login_at=earlier)
    session.add(user)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    await repository.touch_login("tester@example.com")

    await session.refresh(user)
    assert user.last_login_at is not None
    assert user.last_login_at > earlier


async def test_touch_login_matches_the_email_case_insensitively(session: AsyncSession) -> None:
    """Same dialect-independent folding ``get_by_email`` guarantees."""
    user = UserTable(id=uuid4(), email="tester@example.com", role="user")
    session.add(user)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    await repository.touch_login("TESTER@EXAMPLE.COM")

    await session.refresh(user)
    assert user.last_login_at is not None


async def test_touch_login_is_a_no_op_for_an_unknown_email(session: AsyncSession) -> None:
    """An authenticated caller with no ``users`` row has no login to record."""
    repository = SqlAlchemyUserRepository(session)

    await repository.touch_login("ghost@example.com")


async def test_a_role_change_does_not_disturb_the_recorded_login(session: AsyncSession) -> None:
    """The regression guard: this is what reusing ``updated_at`` got wrong."""
    logged_in_at = datetime(2026, 1, 1, tzinfo=UTC)
    user = UserTable(
        id=uuid4(), email="tester@example.com", role="user", last_login_at=logged_in_at
    )
    session.add(user)
    await session.commit()
    repository = SqlAlchemyUserRepository(session)

    await repository.set_role(user.id, "admin")

    await session.refresh(user)
    assert user.last_login_at == logged_in_at


async def test_provision_records_the_first_login(session: AsyncSession) -> None:
    """A caller provisioned on first sight is logging in at that moment."""
    repository = SqlAlchemyUserRepository(session)

    user = await repository.provision("newcomer@example.com")

    assert user.last_login_at is not None
