"""UserService domain rules: role-change guardrails and self metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from autotunex.core.exceptions import (
    CannotChangeOwnRoleError,
    LastAdminError,
    UserNotFoundError,
)
from autotunex.db.repositories.protocols import UserRepository
from autotunex.db.tables import UserTable
from autotunex.models.auth import Principal
from autotunex.models.user import Role
from autotunex.services.users import UserService


def _user(*, role: str, user_id: UUID | None = None) -> UserTable:
    return UserTable(
        id=user_id or uuid4(),
        email=f"{role}-{uuid4().hex[:6]}@example.com",
        role=role,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class FakeUserRepository:
    """In-memory ``UserRepository`` for isolating ``UserService``."""

    def __init__(
        self,
        users: list[UserTable] | None = None,
        *,
        admin_count: int = 1,
        counts: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self._users: dict[UUID, UserTable] = {u.id: u for u in (users or [])}
        self._admin_count = admin_count
        self._counts = counts

    async def touch_login(self, email: str) -> None:
        """Unused by ``UserService``; stubbed only to satisfy the Protocol."""
        raise NotImplementedError

    async def list(self, *, limit: int, offset: int) -> tuple[list[UserTable], int]:
        items = list(self._users.values())
        return items[offset : offset + limit], len(items)

    async def get(self, user_id: UUID) -> UserTable | None:
        return self._users.get(user_id)

    async def set_role(self, user_id: UUID, role: str) -> UserTable | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        user.role = role
        return user

    async def count_admins(self) -> int:
        return self._admin_count

    async def metadata(self, user_id: UUID) -> tuple[int, int, int]:
        return self._counts

    async def provision(self, email: str) -> UserTable:  # pragma: no cover - unused
        raise NotImplementedError

    async def get_by_email(self, email: str) -> UserTable | None:  # pragma: no cover - unused
        raise NotImplementedError


def _admin_principal(user_id: UUID | None) -> Principal:
    return Principal(email="admin@example.com", provider="session", user_id=user_id, is_admin=True)


async def test_get_raises_for_an_unknown_user() -> None:
    repository: UserRepository = FakeUserRepository()
    service = UserService(repository=repository, principal=_admin_principal(uuid4()))

    with pytest.raises(UserNotFoundError):
        await service.get(uuid4())


async def test_set_role_raises_not_found_before_the_own_role_check() -> None:
    missing_id = uuid4()
    repository: UserRepository = FakeUserRepository()
    # Principal's user_id equals the (missing) target: not-found must still win.
    service = UserService(repository=repository, principal=_admin_principal(missing_id))

    with pytest.raises(UserNotFoundError):
        await service.set_role(missing_id, Role.USER)


async def test_set_role_refuses_changing_your_own_role() -> None:
    me = _user(role="admin")
    repository: UserRepository = FakeUserRepository([me], admin_count=2)
    service = UserService(repository=repository, principal=_admin_principal(me.id))

    with pytest.raises(CannotChangeOwnRoleError):
        await service.set_role(me.id, Role.USER)


async def test_set_role_refuses_demoting_the_last_admin() -> None:
    sole_admin = _user(role="admin")
    repository: UserRepository = FakeUserRepository([sole_admin], admin_count=1)
    # A standalone-style admin caller with no counted row of its own.
    service = UserService(repository=repository, principal=_admin_principal(None))

    with pytest.raises(LastAdminError):
        await service.set_role(sole_admin.id, Role.USER)


async def test_set_role_allows_demoting_an_admin_when_another_exists() -> None:
    target = _user(role="admin")
    repository: UserRepository = FakeUserRepository([target], admin_count=2)
    service = UserService(repository=repository, principal=_admin_principal(uuid4()))

    result = await service.set_role(target.id, Role.USER)

    assert result.role == "user"


async def test_set_role_promotes_a_user() -> None:
    target = _user(role="user")
    repository: UserRepository = FakeUserRepository([target], admin_count=1)
    service = UserService(repository=repository, principal=_admin_principal(uuid4()))

    result = await service.set_role(target.id, Role.ADMIN)

    assert result.role == "admin"


async def test_my_metadata_returns_zeros_for_an_unresolvable_caller() -> None:
    repository: UserRepository = FakeUserRepository(counts=(9, 9, 9))
    principal = Principal(email=None, provider="disabled", user_id=None, is_admin=True)
    service = UserService(repository=repository, principal=principal)

    metadata = await service.my_metadata()

    assert (
        metadata.number_of_jobs,
        metadata.number_of_configurations,
        metadata.number_of_datasets,
    ) == (0, 0, 0)


async def test_my_metadata_returns_the_repository_counts() -> None:
    repository: UserRepository = FakeUserRepository(counts=(3, 2, 1))
    service = UserService(repository=repository, principal=_admin_principal(uuid4()))

    metadata = await service.my_metadata()

    assert (
        metadata.number_of_jobs,
        metadata.number_of_configurations,
        metadata.number_of_datasets,
    ) == (3, 2, 1)
