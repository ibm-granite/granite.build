"""Configuration endpoints, end to end over HTTP.

Unlike jobs, configurations are the resource this API creates, updates and
deletes, so these tests pin the full CRUD surface and every domain-error path
(404, the two 409s, the create-refusal 403, and the invalid-search-space 422),
each mapped to the RFC 9457 ``application/problem+json`` shape.

The default test principal is an unrestricted standalone admin with no
``user_id`` (see ``make_settings``); its writes are attributed to and reused
from the lazily-provisioned default system owner
(``SYSTEM_STANDALONE_EMAIL``), so this caller *can* create — the create/update
tests still swap in a provisioned principal via ``as_principal`` to exercise
ownership as a real, non-admin user, and a real provider's unprovisioned
caller remains a create-refusal 403.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_autotune_core, get_session
from autotunex.core.auth.disabled import SYSTEM_STANDALONE_EMAIL
from autotunex.core.config import get_settings
from autotunex.core.exceptions import AutotuneCoreUnavailableError
from autotunex.db.tables import ConfigurationTable, JobTable, UserTable
from autotunex.main import create_app
from autotunex.models.auth import Principal
from tests.conftest import API, make_settings

SPACE: dict[str, Any] = {"learning_rate": {"kind": "float", "low": 1e-6, "high": 1e-3, "log": True}}

PROBLEM_JSON = "application/problem+json"


def _act_as(as_principal: Callable[[Principal], None], user: UserTable) -> None:
    """Resolve every request to ``user`` — a provisioned, non-admin owner."""
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))


class _FakeAutotuneCore:
    """Minimal ``AutotuneCore`` for overriding ``get_autotune_core`` in a test."""

    def __init__(
        self, *, config: dict[str, Any] | None = None, raises: Exception | None = None
    ) -> None:
        self._config = config if config is not None else {}
        self._raises = raises

    async def get_config_template(self) -> dict[str, Any]:
        if self._raises is not None:
            raise self._raises
        return self._config

    async def get_dataset_types(self) -> dict[str, Any]:
        return {}


# Template.


async def test_get_template_returns_the_autotune_template(
    app: FastAPI, client: AsyncClient
) -> None:
    template = {"tune_config": {"num_samples": {"default": 16, "min_val": 1, "max_val": 10000}}}
    app.dependency_overrides[get_autotune_core] = lambda: _FakeAutotuneCore(config=template)

    response = await client.get(f"{API}/configurations/template")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == template


async def test_get_template_is_503_when_autotune_is_absent(
    app: FastAPI, client: AsyncClient
) -> None:
    app.dependency_overrides[get_autotune_core] = lambda: _FakeAutotuneCore(
        raises=AutotuneCoreUnavailableError()
    )

    response = await client.get(f"{API}/configurations/template")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


# Create.


async def test_create_returns_201_and_the_created_configuration(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/configurations", json={"name": "lora-sweep", "config_data": SPACE}
    )

    assert response.status_code == HTTPStatus.CREATED
    body = response.json()
    assert body["name"] == "lora-sweep"
    assert body["user_id"] == str(user.id)


async def test_a_created_configuration_can_be_fetched_back(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)
    created = await client.post(f"{API}/configurations", json={"name": "c", "config_data": SPACE})

    response = await client.get(f"{API}/configurations/{created.json()['id']}")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["config_data"] == SPACE


async def test_create_with_empty_config_data_is_422(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.post(f"{API}/configurations", json={"name": "bad", "config_data": {}})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_create_accepts_a_rich_nested_config_data(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    """config_data is a schema-less JSON column: the real, deeply-nested blob is accepted."""
    _act_as(as_principal, user)
    blob = {
        "tune_config": {
            "num_samples": {"type": "int", "default": 16, "min_val": 1, "max_val": 10000},
            "search_alg": {"type": "str", "values": ["random", "blds"], "default": "blds"},
        },
        "tuners_config": {
            "lora": {"hyperparams": {"r": {"type": "int", "values": [8, 16, 32], "default": 32}}}
        },
        "training_rl_config": {"reward_model_path": {"type": "str", "default": None}},
    }

    response = await client.post(
        f"{API}/configurations", json={"name": "test-config", "config_data": blob}
    )

    assert response.status_code == HTTPStatus.CREATED
    assert response.json()["config_data"] == blob


async def test_create_with_a_duplicate_name_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/configurations", json={"name": configuration.name, "config_data": SPACE}
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_create_by_an_unprovisioned_caller_is_403(
    client: AsyncClient, as_principal: Callable[[Principal], None]
) -> None:
    as_principal(
        Principal(email="ghost@example.com", provider="session", user_id=None, is_admin=False)
    )

    response = await client.post(f"{API}/configurations", json={"name": "x", "config_data": SPACE})

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_standalone_without_an_email_can_create(client: AsyncClient) -> None:
    """No override: the default system owner is provisioned, so writes succeed."""
    response = await client.post(f"{API}/configurations", json={"name": "x", "config_data": SPACE})

    assert response.status_code == HTTPStatus.CREATED


# List.


async def test_list_returns_the_owner_s_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/configurations")

    assert response.status_code == HTTPStatus.OK
    assert [item["id"] for item in response.json()["items"]] == [str(configuration.id)]
    assert response.json()["total"] == 1


async def test_list_is_empty_for_an_unresolved_principal(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="ghost@example.com", provider="session", user_id=None, is_admin=False)
    )

    response = await client.get(f"{API}/configurations")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_an_admin_s_default_list_excludes_a_foreign_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/configurations")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_an_admin_lists_a_foreign_configuration_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/configurations", params={"scope": "all"})

    assert response.status_code == HTTPStatus.OK
    assert [item["id"] for item in response.json()["items"]] == [str(configuration.id)]
    assert response.json()["total"] == 1


async def test_a_non_admin_requesting_scope_all_on_list_is_403(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/configurations", params={"scope": "all"})

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_list_rejects_a_limit_above_100(client: AsyncClient) -> None:
    response = await client.get(f"{API}/configurations", params={"limit": 101})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_list_rejects_a_negative_offset(client: AsyncClient) -> None:
    response = await client.get(f"{API}/configurations", params={"offset": -1})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_list_configurations_passes_q_filter(client: AsyncClient) -> None:
    response = await client.get(f"{API}/configurations", params={"q": "zzz-no-such-config"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []


# Get one.


async def test_get_returns_the_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/configurations/{configuration.id}")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(configuration.id)


async def test_get_configuration_includes_caller_scoped_associated_jobs(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/configurations/{job.config_id}")

    assert response.status_code == HTTPStatus.OK
    assert [j["experiment_name"] for j in response.json()["associated_jobs"]] == [
        job.experiment_name
    ]


async def test_list_configurations_includes_associated_jobs(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))

    response = await client.get(f"{API}/configurations")

    item = next(c for c in response.json()["items"] if c["id"] == str(job.config_id))
    assert [j["experiment_name"] for j in item["associated_jobs"]] == [job.experiment_name]


async def test_get_of_an_unknown_configuration_is_404(client: AsyncClient) -> None:
    response = await client.get(f"{API}/configurations/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)
    assert response.json()["status"] == HTTPStatus.NOT_FOUND


async def test_get_of_another_users_configuration_is_404(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(
            email="someone-else@example.com", provider="session", user_id=uuid4(), is_admin=False
        )
    )

    response = await client.get(f"{API}/configurations/{configuration.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_an_admin_s_default_get_excludes_a_foreign_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/configurations/{configuration.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_an_admin_gets_a_foreign_configuration_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/configurations/{configuration.id}", params={"scope": "all"})

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(configuration.id)


async def test_a_non_admin_requesting_scope_all_on_get_is_403(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/configurations/{configuration.id}", params={"scope": "all"})

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


# Update (PUT — full replace).


async def test_update_replaces_the_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)
    new_space = {"lora_rank": {"kind": "int", "low": 4, "high": 64, "step": 4}}

    response = await client.put(
        f"{API}/configurations/{configuration.id}",
        json={"name": "renamed", "tuner_type": "optuna", "config_data": new_space},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["name"] == "renamed"
    assert response.json()["config_data"] == new_space


async def test_update_of_an_unknown_configuration_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.put(
        f"{API}/configurations/{uuid4()}", json={"name": "x", "config_data": SPACE}
    )

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_update_with_empty_config_data_is_422(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.put(
        f"{API}/configurations/{configuration.id}", json={"name": "x", "config_data": {}}
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_update_into_a_duplicate_name_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)
    other = await client.post(f"{API}/configurations", json={"name": "other", "config_data": SPACE})

    response = await client.put(
        f"{API}/configurations/{other.json()['id']}",
        json={"name": configuration.name, "config_data": SPACE},
    )

    assert response.status_code == HTTPStatus.CONFLICT


async def test_an_admin_s_default_update_is_a_404_for_a_foreign_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.put(
        f"{API}/configurations/{configuration.id}", json={"name": "x", "config_data": SPACE}
    )

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_an_admin_updates_a_foreign_configuration_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.put(
        f"{API}/configurations/{configuration.id}",
        params={"scope": "all"},
        json={"name": "renamed-by-admin", "config_data": SPACE},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["name"] == "renamed-by-admin"


async def test_a_non_admin_requesting_scope_all_on_update_is_403(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.put(
        f"{API}/configurations/{configuration.id}",
        params={"scope": "all"},
        json={"name": "x", "config_data": SPACE},
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


# Delete.


async def test_delete_returns_204(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/configurations/{configuration.id}")

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert (await client.get(f"{API}/configurations/{configuration.id}")).status_code == (
        HTTPStatus.NOT_FOUND
    )


async def test_delete_of_an_unknown_configuration_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/configurations/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_delete_of_a_referenced_configuration_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """The ``job`` fixture references the ``configuration`` fixture (ON DELETE RESTRICT)."""
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/configurations/{job.config_id}")

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_an_admin_s_default_delete_is_a_404_for_a_foreign_configuration(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.delete(f"{API}/configurations/{configuration.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_an_admin_deletes_a_foreign_configuration_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    configuration: ConfigurationTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.delete(
        f"{API}/configurations/{configuration.id}", params={"scope": "all"}
    )

    assert response.status_code == HTTPStatus.NO_CONTENT


async def test_a_non_admin_requesting_scope_all_on_delete_is_403(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    configuration: ConfigurationTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(
        f"{API}/configurations/{configuration.id}", params={"scope": "all"}
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


# JIT provisioning, end to end. `as_principal` overrides stage-two resolution
# outright, so these build a real app and let get_principal run: a standalone
# caller with a configured email but no users row.


async def _client_for(session: AsyncSession, **kwargs: object) -> AsyncClient:
    """An HTTP client onto a bespoke app whose settings drive real principal resolution.

    Overriding ``get_settings`` too is not optional: ``get_principal`` reads the
    ``auto_provision_users`` flag through ``SettingsDep``, so without this the
    flag would resolve from the real singleton rather than these settings.
    """
    settings = make_settings(**kwargs)  # type: ignore[arg-type]
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = lambda: session
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def test_jit_provisioning_lets_a_first_time_caller_create(session: AsyncSession) -> None:
    async with await _client_for(
        session, standalone_email="newcomer@example.com", auto_provision_users=True
    ) as client:
        response = await client.post(
            f"{API}/configurations", json={"name": "first", "config_data": SPACE}
        )

    assert response.status_code == HTTPStatus.CREATED
    assert response.json()["name"] == "first"


async def test_a_standalone_caller_is_provisioned_without_the_flag(
    session: AsyncSession,
) -> None:
    """Standalone always owns its writes, even with auto_provision_users off."""
    async with await _client_for(
        session, standalone_email="newcomer@example.com", auto_provision_users=False
    ) as client:
        response = await client.post(
            f"{API}/configurations", json={"name": "first", "config_data": SPACE}
        )

    assert response.status_code == HTTPStatus.CREATED
    assert response.json()["name"] == "first"


# Default system owner: attribution and row reuse.
#
# ``ConfigurationRead`` surfaces the owner as ``user_id`` (the owner's id, not
# their email — see ``models/configuration.py``), so these resolve the
# provisioned row through ``UserTable`` rather than asserting on an email
# field the response does not carry.


async def test_standalone_create_is_attributed_to_the_default_system_owner(
    session: AsyncSession,
) -> None:
    async with await _client_for(session) as client:  # unset standalone_email
        created = await client.post(
            f"{API}/configurations", json={"name": "sys", "config_data": SPACE}
        )

    assert created.status_code == HTTPStatus.CREATED
    owner_id = UUID(created.json()["user_id"])
    owner = (await session.execute(select(UserTable).where(UserTable.id == owner_id))).scalar_one()
    assert owner.email == SYSTEM_STANDALONE_EMAIL


async def test_the_default_system_owner_row_is_reused_across_requests(
    session: AsyncSession,
) -> None:
    async with await _client_for(session) as client:
        first = await client.post(f"{API}/configurations", json={"name": "a", "config_data": SPACE})
        second = await client.post(
            f"{API}/configurations", json={"name": "b", "config_data": SPACE}
        )

    assert first.status_code == HTTPStatus.CREATED
    assert second.status_code == HTTPStatus.CREATED
    users = (await session.execute(select(UserTable))).scalars().all()
    assert [u.email for u in users] == [SYSTEM_STANDALONE_EMAIL]


async def test_standalone_role_user_cannot_see_a_configuration_owned_by_someone_else(
    session: AsyncSession, configuration: ConfigurationTable
) -> None:
    """standalone_role="user" scopes the default owner to its own rows.

    ``configuration`` is owned by the ``user`` fixture (``tester@example.com``), a
    different row from the default system owner this narrowed standalone
    principal resolves to — so it must be invisible, exactly like any other
    non-admin caller's 404 on someone else's row.
    """
    async with await _client_for(session, standalone_role="user") as client:
        response = await client.get(f"{API}/configurations/{configuration.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_the_default_standalone_admin_cannot_see_a_foreign_configuration_by_default(
    session: AsyncSession, configuration: ConfigurationTable
) -> None:
    """Reads default to own-data even for the standalone admin.

    ``configuration`` is owned by the ``user`` fixture (``tester@example.com``), not
    the default system owner (``SYSTEM_STANDALONE_EMAIL``) this unrestricted,
    unset-``standalone_email`` caller resolves to — under the ``scope=own``
    default, an admin only sees rows it owns, unlike the pre-scoping behavior.
    See the ``?scope=all`` variant below for the unscoped admin view.
    """
    async with await _client_for(session) as client:  # unset standalone_email
        response = await client.get(f"{API}/configurations/{configuration.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_the_default_standalone_admin_sees_a_foreign_configuration_with_scope_all(
    session: AsyncSession, configuration: ConfigurationTable
) -> None:
    """``?scope=all`` is what unlocks the cross-user view for the standalone admin.

    ``configuration`` is owned by the ``user`` fixture (``tester@example.com``), not
    the default system owner (``SYSTEM_STANDALONE_EMAIL``) this unrestricted,
    unset-``standalone_email`` caller resolves to — admin sees every row when it
    explicitly asks for ``scope=all``, unlike the ``scope=own`` default above.
    """
    async with await _client_for(session) as client:  # unset standalone_email
        response = await client.get(
            f"{API}/configurations/{configuration.id}", params={"scope": "all"}
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(configuration.id)
