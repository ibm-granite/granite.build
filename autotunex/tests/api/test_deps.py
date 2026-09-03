"""The two-stage principal dependency.

Stage one never touches the database — these tests prove that by driving it
directly with a fake app.state, no session fixture involved. Stage two's
database resolution is exercised through the ``session`` fixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_authenticated_principal, get_principal, get_session
from autotunex.core.auth.disabled import DisabledAuthenticator
from autotunex.db.repositories.sqlalchemy import SqlAlchemyUserRepository
from autotunex.db.tables import UserTable
from autotunex.main import create_app
from autotunex.models.auth import Principal
from tests.conftest import API, make_settings


def _request_with_authenticator(authenticator: object) -> Request:
    scope = {
        "type": "http",
        "headers": [],
        "app": SimpleNamespace(state=SimpleNamespace(authenticator=authenticator)),
    }
    return Request(scope)


async def test_stage_one_delegates_to_the_app_s_authenticator() -> None:
    request = _request_with_authenticator(DisabledAuthenticator(make_settings()))

    principal = await get_authenticated_principal(
        request, bearer=None, x_api_key=None, session=None
    )

    assert principal.provider == "standalone"


class _RecordingAuthenticator:
    """An ``Authenticator`` that records the bearer slot it was handed.

    What stage one is responsible for now is the one-line *mapping* from a
    parsed credential to the bare token the ``Authenticator`` expects — parsing
    itself moved to ``HTTPBearer`` (see ``bearer_scheme`` in ``deps.py``) — so
    what these tests need to see is the value that reached the authenticator,
    not the principal that came back.
    """

    def __init__(self) -> None:
        self.bearers: list[str | None] = []

    async def authenticate(
        self, *, bearer: str | None, api_key: str | None, session: str | None
    ) -> Principal:
        """Record the bearer slot and return a fixed standalone principal."""
        self.bearers.append(bearer)
        return Principal(email=None, provider="standalone", is_admin=True)


async def test_stage_one_extracts_the_token_from_parsed_bearer_credentials() -> None:
    """The one piece of bearer handling still ours to unit-test.

    Parsing ``Authorization`` into a scheme and a token is now ``HTTPBearer``'s
    job (see ``bearer_scheme`` in ``deps.py``), exercised at the HTTP level in
    ``test_auth.py`` where the header actually gets parsed. What is still ours
    is the one-line mapping from FastAPI's ``HTTPAuthorizationCredentials`` to
    the bare token string the ``Authenticator`` expects — building the
    credentials object by hand here tests *that* mapping, not the library's
    parsing, so it does not fall foul of the "pre-parsed header" trap the three
    deleted tests fell into.
    """
    authenticator = _RecordingAuthenticator()
    request = _request_with_authenticator(authenticator)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="sometoken")

    await get_authenticated_principal(request, bearer=credentials, x_api_key=None, session=None)

    assert authenticator.bearers == ["sometoken"]


async def test_stage_two_leaves_an_unrestricted_standalone_principal_untouched(
    session: AsyncSession,
) -> None:
    authenticated = Principal(email=None, provider="standalone", is_admin=True)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.user_id is None
    assert resolved.is_admin is True


async def test_stage_two_resolves_user_id_for_a_narrowed_standalone_principal(
    session: AsyncSession,
) -> None:
    user = UserTable(id=uuid4(), email="dev@example.com", role="user")
    session.add(user)
    await session.commit()
    authenticated = Principal(email="dev@example.com", provider="standalone", is_admin=True)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.user_id == user.id


async def test_stage_two_does_not_let_a_standalone_user_s_db_role_override_the_setting(
    session: AsyncSession,
) -> None:
    """standalone_role wins over users.role — see the design spec's semantics."""
    user = UserTable(id=uuid4(), email="dev@example.com", role="admin")
    session.add(user)
    await session.commit()
    authenticated = Principal(email="dev@example.com", provider="standalone", is_admin=False)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.is_admin is False


async def test_stage_two_resolves_is_admin_from_the_database_for_a_real_provider(
    session: AsyncSession,
) -> None:
    user = UserTable(id=uuid4(), email="real@example.com", role="admin")
    session.add(user)
    await session.commit()
    authenticated = Principal(email="real@example.com", provider="session", is_admin=False)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.is_admin is True
    assert resolved.user_id == user.id


async def test_stage_two_resolves_identity_but_not_privilege_for_a_non_admin_role(
    session: AsyncSession,
) -> None:
    """Both halves matter: identity must resolve while privilege must not.

    A row that merely *exists* is not an admin. If ``is_admin`` were derived
    from a row's existence rather than its ``role``, every caller with a
    provisioned row could pass ``?scope=all`` and read every user's jobs — the
    unfiltered view is granted only to an admin, in
    ``autotunex.services.scoping.resolve_owner_filter``.
    """
    user = UserTable(id=uuid4(), email="plain@example.com", role="user")
    session.add(user)
    await session.commit()
    authenticated = Principal(email="plain@example.com", provider="session", is_admin=False)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.is_admin is False
    assert resolved.user_id == user.id


async def test_stage_two_treats_a_null_role_as_non_admin(session: AsyncSession) -> None:
    """``users.role`` is nullable and the ORM default never reaches these rows.

    ``role`` is ``Mapped[str | None]`` and the tuning pipeline writes ``users``
    directly, so SQLAlchemy's ``default="user"`` never applies — a ``NULL``
    role is a realistic production state, and it must not resolve to admin.
    """
    user = UserTable(id=uuid4(), email="roleless@example.com", role=None)
    session.add(user)
    await session.commit()
    authenticated = Principal(email="roleless@example.com", provider="session", is_admin=False)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.is_admin is False
    assert resolved.user_id == user.id


async def test_stage_two_leaves_user_id_none_for_a_real_provider_with_no_matching_row(
    session: AsyncSession,
) -> None:
    authenticated = Principal(email="ghost@example.com", provider="session", is_admin=False)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.user_id is None
    assert resolved.is_admin is False


async def test_stage_two_provisions_a_missing_user_when_enabled(session: AsyncSession) -> None:
    """JIT provisioning: a verified email with no row gets one, and owns nothing yet."""
    repository = SqlAlchemyUserRepository(session)
    authenticated = Principal(email="newcomer@example.com", provider="session", is_admin=False)

    resolved = await get_principal(
        authenticated, repository, make_settings(auto_provision_users=True)
    )

    assert resolved.user_id is not None
    assert await repository.get_by_email("newcomer@example.com") is not None


async def test_stage_two_provisioned_user_is_never_admin(session: AsyncSession) -> None:
    """A provisioned row is role='user', so a real provider resolves is_admin=False."""
    authenticated = Principal(email="newcomer@example.com", provider="session", is_admin=False)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings(auto_provision_users=True)
    )

    assert resolved.is_admin is False


async def test_stage_two_always_provisions_a_standalone_owner_without_the_flag(
    session: AsyncSession,
) -> None:
    """Standalone must own its writes even with auto_provision_users off."""
    authenticated = Principal(
        email="standalone@autotunex.local", provider="standalone", is_admin=True
    )
    repository = SqlAlchemyUserRepository(session)

    principal = await get_principal(
        authenticated, repository, make_settings(auto_provision_users=False)
    )

    assert principal.user_id is not None
    assert principal.is_admin is True


class _RecordingUserRepository:
    """A ``UserRepository`` that records every email it was asked about."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def provision(self, email: str) -> UserTable:
        """Fail loudly: the email-less path must never reach provisioning."""
        raise AssertionError("provision must not be called for an email-less principal")

    async def touch_login(self, email: str) -> None:
        """Fail loudly: with no row resolved there is no login to record."""
        raise AssertionError("touch_login must not be called when no user row matched")

    async def get_by_email(self, email: str) -> UserTable | None:
        """Record the lookup and report no match."""
        self.calls.append(email)
        return None

    async def list(self, *, limit: int, offset: int) -> tuple[Sequence[UserTable], int]:
        """Unused by ``get_principal``; stubbed only to satisfy the Protocol."""
        raise NotImplementedError

    async def get(self, user_id: UUID) -> UserTable | None:
        """Unused by ``get_principal``; stubbed only to satisfy the Protocol."""
        raise NotImplementedError

    async def set_role(self, user_id: UUID, role: str) -> UserTable | None:
        """Unused by ``get_principal``; stubbed only to satisfy the Protocol."""
        raise NotImplementedError

    async def count_admins(self) -> int:
        """Unused by ``get_principal``; stubbed only to satisfy the Protocol."""
        raise NotImplementedError

    async def metadata(self, user_id: UUID) -> tuple[int, int, int]:
        """Unused by ``get_principal``; stubbed only to satisfy the Protocol."""
        raise NotImplementedError


async def test_stage_two_never_queries_the_database_for_an_email_less_principal() -> None:
    """The ``email is None`` early return, asserted directly rather than inferred.

    Delete that branch and the observable failure today is an ``AttributeError``
    from ``None.lower()`` deep inside the repository — a different defect that
    merely happens to be loud, and one that would fall silent the moment a
    repository tolerated ``None``. Asserting the repository is never consulted
    pins the actual contract: an unrestricted standalone principal has nothing to
    resolve, so stage two does no I/O at all.
    """
    repository = _RecordingUserRepository()
    authenticated = Principal(email=None, provider="standalone", is_admin=True)

    # auto_provision_users=True on purpose: the email-less early return must win
    # even with provisioning enabled, so neither get_by_email nor provision runs.
    resolved = await get_principal(
        authenticated, repository, make_settings(auto_provision_users=True)
    )

    assert repository.calls == []
    assert resolved is authenticated


async def test_an_ambiguous_identity_is_a_problem_detail_that_names_neither_email(
    session: AsyncSession,
) -> None:
    """Duplicate case-variant emails fail closed, and opaquely.

    Drives the whole two-stage pipeline over HTTP because that is where the
    consequence lives: before this, ``MultipleResultsFound`` escaped as a bare
    ``500 {"detail": "An unexpected error occurred."}`` with only "Unhandled
    error" in the log, for every request from that caller, permanently. The
    response must not describe the duplication — the operator gets the email from
    the WARNING log instead.
    """
    session.add_all(
        [
            UserTable(id=uuid4(), email="Alice@example.com", role="admin"),
            UserTable(id=uuid4(), email="alice@example.com", role="user"),
        ]
    )
    await session.commit()
    app = create_app(make_settings(standalone_email="alice@example.com"))
    app.dependency_overrides[get_session] = lambda: session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(f"{API}/jobs")

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["detail"] == "The account could not be resolved."
    assert "alice" not in response.text.lower()


async def test_stage_two_records_a_login_for_a_caller_that_has_none(
    session: AsyncSession,
) -> None:
    """The regression: an authenticated request must register as a login."""
    user = UserTable(id=uuid4(), email="caller@example.com", role="user")
    session.add(user)
    await session.commit()
    authenticated = Principal(email="caller@example.com", provider="session", is_admin=False)

    await get_principal(authenticated, SqlAlchemyUserRepository(session), make_settings())

    await session.refresh(user)
    assert user.last_login_at is not None


async def test_stage_two_refreshes_a_login_older_than_the_throttle_window(
    session: AsyncSession,
) -> None:
    stale = datetime.now(UTC) - timedelta(hours=2)
    user = UserTable(id=uuid4(), email="caller@example.com", role="user", last_login_at=stale)
    session.add(user)
    await session.commit()
    authenticated = Principal(email="caller@example.com", provider="session", is_admin=False)

    await get_principal(
        authenticated,
        SqlAlchemyUserRepository(session),
        make_settings(login_activity_throttle_minutes=15),
    )

    await session.refresh(user)
    assert user.last_login_at is not None
    assert user.last_login_at > stale


async def test_stage_two_leaves_a_recent_login_alone(session: AsyncSession) -> None:
    """Throttling is what keeps this off the hot path: no write per request."""
    recent = datetime.now(UTC) - timedelta(minutes=1)
    user = UserTable(id=uuid4(), email="caller@example.com", role="user", last_login_at=recent)
    session.add(user)
    await session.commit()
    authenticated = Principal(email="caller@example.com", provider="session", is_admin=False)

    await get_principal(
        authenticated,
        SqlAlchemyUserRepository(session),
        make_settings(login_activity_throttle_minutes=15),
    )

    await session.refresh(user)
    assert user.last_login_at == recent


async def test_stage_two_records_no_login_for_an_unprovisioned_caller(
    session: AsyncSession,
) -> None:
    """No row means no login to record — and no crash."""
    authenticated = Principal(email="ghost@example.com", provider="session", is_admin=False)

    resolved = await get_principal(
        authenticated, SqlAlchemyUserRepository(session), make_settings()
    )

    assert resolved.user_id is None


async def test_a_failed_login_write_does_not_break_the_request(
    session: AsyncSession,
) -> None:
    """Bookkeeping must never turn a working GET into a 500."""

    class _FailingTouch(SqlAlchemyUserRepository):
        async def touch_login(self, email: str) -> None:
            raise OperationalError("UPDATE users", {}, Exception("lock wait timeout"))

    user = UserTable(id=uuid4(), email="caller@example.com", role="user")
    session.add(user)
    await session.commit()
    authenticated = Principal(email="caller@example.com", provider="session", is_admin=False)

    resolved = await get_principal(authenticated, _FailingTouch(session), make_settings())

    assert resolved.user_id == user.id
