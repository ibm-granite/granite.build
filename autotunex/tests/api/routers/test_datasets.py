"""Dataset endpoints, end to end over HTTP.

Pins the full CRUD surface, the upload transition to ``uploading``, and every
domain-error path (404, the three 409s, 403-on-create, 413/415/422), each mapped
to RFC 9457 ``application/problem+json``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from autotunex.api.deps import get_dataset_runner, get_storage_backend
from autotunex.core.auth.disabled import SYSTEM_STANDALONE_EMAIL
from autotunex.core.config import Settings, get_settings
from autotunex.db.tables import DatasetTable, JobTable, UserTable
from autotunex.models.auth import Principal
from autotunex.models.status import DatasetStatus
from autotunex.services.dataset_runner import NoOpDatasetUploadRunner
from autotunex.services.storage.local import LocalStorageBackend
from tests.conftest import API

PROBLEM_JSON = "application/problem+json"


def _act_as(as_principal: Callable[[Principal], None], user: UserTable) -> None:
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))


@pytest.fixture
def local_storage(app: FastAPI, tmp_path: Path) -> NoOpDatasetUploadRunner:
    """Point storage at a tmp dir and stub the runner (no background work in tests)."""
    runner = NoOpDatasetUploadRunner()
    app.dependency_overrides[get_storage_backend] = lambda: LocalStorageBackend(root=tmp_path)
    app.dependency_overrides[get_dataset_runner] = lambda: runner
    return runner


@pytest.fixture
async def other_users_dataset(session: AsyncSession) -> DatasetTable:
    """A second dataset, owned by somebody other than the ``user`` fixture.

    Without a row that must be excluded, a positive scoping assertion proves
    nothing — see ``other_users_job`` in ``test_jobs_scoping.py`` for the same
    rationale.
    """
    other = UserTable(id=uuid4(), email="not-the-owner@example.com", role="user")
    session.add(other)
    await session.commit()
    other_dataset = DatasetTable(
        id=uuid4(), user_id=str(other.id), name="not-yours", description="Someone else's."
    )
    session.add(other_dataset)
    await session.commit()
    return other_dataset


# Create.


async def test_create_returns_201(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/datasets", json={"name": "alpaca", "description": "Instruction data."}
    )

    assert response.status_code == HTTPStatus.CREATED
    body = response.json()
    assert body["name"] == "alpaca"
    assert body["status"] == "empty"
    assert body["user_id"] == str(user.id)


async def test_create_with_a_bad_format_is_422(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/datasets", json={"name": "d", "description": "x", "data_format": "xml"}
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_create_without_a_description_succeeds_with_null(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.post(f"{API}/datasets", json={"name": "d"})

    assert response.status_code == HTTPStatus.CREATED
    assert response.json()["description"] is None


async def test_create_duplicate_name_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(f"{API}/datasets", json={"name": dataset.name, "description": "x"})

    assert response.status_code == HTTPStatus.CONFLICT


async def test_create_by_an_unprovisioned_caller_is_403(
    client: AsyncClient, as_principal: Callable[[Principal], None]
) -> None:
    as_principal(
        Principal(email="ghost@example.com", provider="session", user_id=None, is_admin=False)
    )

    response = await client.post(f"{API}/datasets", json={"name": "x", "description": "y"})

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_standalone_without_an_email_can_create(client: AsyncClient) -> None:
    """No-user mode: the default system owner is provisioned, so writes succeed."""
    response = await client.post(f"{API}/datasets", json={"name": "corpus"})

    assert response.status_code == HTTPStatus.CREATED


# Default system owner: attribution and row reuse. ``client``/``session`` are
# the plain fixtures (standalone, unset email) — no bespoke app needed, since
# that is already the default principal in this file.


async def test_standalone_create_is_attributed_to_the_default_system_owner(
    client: AsyncClient, session: AsyncSession
) -> None:
    created = await client.post(f"{API}/datasets", json={"name": "sys-corpus"})

    assert created.status_code == HTTPStatus.CREATED
    owner_id = UUID(created.json()["user_id"])
    owner = (await session.execute(select(UserTable).where(UserTable.id == owner_id))).scalar_one()
    assert owner.email == SYSTEM_STANDALONE_EMAIL


async def test_the_default_system_owner_row_is_reused_across_requests(
    client: AsyncClient, session: AsyncSession
) -> None:
    first = await client.post(f"{API}/datasets", json={"name": "a"})
    second = await client.post(f"{API}/datasets", json={"name": "b"})

    assert first.status_code == HTTPStatus.CREATED
    assert second.status_code == HTTPStatus.CREATED
    users = (await session.execute(select(UserTable))).scalars().all()
    assert [u.email for u in users] == [SYSTEM_STANDALONE_EMAIL]


# List / get.


async def test_list_returns_the_owners_dataset(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/datasets")

    assert response.status_code == HTTPStatus.OK
    assert [item["id"] for item in response.json()["items"]] == [str(dataset.id)]


async def test_list_is_empty_for_an_unresolved_principal(
    client: AsyncClient, as_principal: Callable[[Principal], None], dataset: DatasetTable
) -> None:
    as_principal(
        Principal(email="ghost@example.com", provider="session", user_id=None, is_admin=False)
    )

    response = await client.get(f"{API}/datasets")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_list_rejects_a_limit_above_100(client: AsyncClient) -> None:
    response = await client.get(f"{API}/datasets", params={"limit": 101})

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_list_datasets_passes_q_filter(client: AsyncClient) -> None:
    response = await client.get(f"{API}/datasets", params={"q": "zzz-no-such-dataset"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []


async def test_get_unknown_is_404(client: AsyncClient) -> None:
    response = await client.get(f"{API}/datasets/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_get_of_another_users_dataset_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], dataset: DatasetTable
) -> None:
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/datasets/{dataset.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_get_includes_caller_scoped_associated_jobs(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job_referencing_dataset: JobTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/datasets/{job_referencing_dataset.dataset_id}")

    assert response.status_code == HTTPStatus.OK
    assert [j["experiment_name"] for j in response.json()["associated_jobs"]] == [
        job_referencing_dataset.experiment_name
    ]


# Scope: admin default vs ?scope=all.


async def test_admin_sees_only_their_own_datasets_by_default(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    dataset: DatasetTable,
    other_users_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/datasets")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_admin_lists_every_dataset_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    dataset: DatasetTable,
    other_users_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/datasets?scope=all")

    assert response.json()["total"] == 2
    ids = {item["id"] for item in response.json()["items"]}
    assert {str(dataset.id), str(other_users_dataset.id)} == ids


async def test_admin_cannot_fetch_a_foreign_dataset_by_default(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    other_users_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/datasets/{other_users_dataset.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_admin_can_fetch_a_foreign_dataset_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    other_users_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/datasets/{other_users_dataset.id}?scope=all")

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == str(other_users_dataset.id)


async def test_admin_can_update_a_foreign_dataset_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    other_users_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.put(
        f"{API}/datasets/{other_users_dataset.id}?scope=all",
        json={"name": "renamed", "description": "new", "data_format": "csv"},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["name"] == "renamed"


async def test_admin_cannot_update_a_foreign_dataset_by_default(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    other_users_dataset: DatasetTable,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.put(
        f"{API}/datasets/{other_users_dataset.id}",
        json={"name": "renamed", "description": "new", "data_format": "csv"},
    )

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_admin_can_delete_a_foreign_dataset_with_scope_all(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    other_users_dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.delete(f"{API}/datasets/{other_users_dataset.id}?scope=all")

    assert response.status_code == HTTPStatus.NO_CONTENT


async def test_admin_cannot_delete_a_foreign_dataset_by_default(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    other_users_dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.delete(f"{API}/datasets/{other_users_dataset.id}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_list(
    client: AsyncClient, as_principal: Callable[[Principal], None], dataset: DatasetTable
) -> None:
    as_principal(
        Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/datasets?scope=all")

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_get(
    client: AsyncClient, as_principal: Callable[[Principal], None], dataset: DatasetTable
) -> None:
    as_principal(
        Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/datasets/{dataset.id}?scope=all")

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_put(
    client: AsyncClient, as_principal: Callable[[Principal], None], dataset: DatasetTable
) -> None:
    as_principal(
        Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.put(
        f"{API}/datasets/{dataset.id}?scope=all",
        json={"name": "x", "description": "y", "data_format": "csv"},
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_a_non_admin_requesting_scope_all_is_forbidden_on_delete(
    client: AsyncClient, as_principal: Callable[[Principal], None], dataset: DatasetTable
) -> None:
    as_principal(
        Principal(email="u@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.delete(f"{API}/datasets/{dataset.id}?scope=all")

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_standalone_default_scopes_to_its_own_owner(
    client: AsyncClient, dataset: DatasetTable
) -> None:
    """No override at all: the standalone admin now sees only its own rows.

    The ``dataset`` fixture is owned by ``tester@example.com``, a different owner
    than the standalone system account, so the default (own) scope excludes it.
    """
    response = await client.get(f"{API}/datasets")

    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_standalone_admin_sees_every_dataset_with_scope_all(
    client: AsyncClient, dataset: DatasetTable
) -> None:
    response = await client.get(f"{API}/datasets?scope=all")

    assert response.json()["total"] == 1


# Update / delete.


async def test_update_replaces_metadata(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
) -> None:
    _act_as(as_principal, user)

    response = await client.put(
        f"{API}/datasets/{dataset.id}",
        json={"name": "renamed", "description": "new", "data_format": "csv"},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["name"] == "renamed"
    assert response.json()["data_format"] == "csv"


async def test_update_unknown_is_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable
) -> None:
    _act_as(as_principal, user)

    response = await client.put(f"{API}/datasets/{uuid4()}", json={"name": "x", "description": "y"})

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_delete_returns_204(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/datasets/{dataset.id}")

    assert response.status_code == HTTPStatus.NO_CONTENT
    assert (await client.get(f"{API}/datasets/{dataset.id}")).status_code == HTTPStatus.NOT_FOUND


async def test_delete_unknown_is_404(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/datasets/{uuid4()}")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_delete_of_a_referenced_dataset_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job_referencing_dataset: JobTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)

    response = await client.delete(f"{API}/datasets/{job_referencing_dataset.dataset_id}")

    assert response.status_code == HTTPStatus.CONFLICT


# Upload.


def _multipart(content: bytes, filename: str) -> dict[str, tuple[str, io.BytesIO, str]]:
    return {"train_file": (filename, io.BytesIO(content), "application/octet-stream")}


async def test_upload_returns_202_and_uploading_status(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/datasets/{dataset.id}/upload",
        files=_multipart(json.dumps({"a": 1}).encode() + b"\n", "train.jsonl"),
    )

    assert response.status_code == HTTPStatus.ACCEPTED
    assert response.json()["status"] == "uploading"
    assert local_storage.submitted == [dataset.id]


async def test_upload_of_an_unsupported_extension_is_415(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/datasets/{dataset.id}/upload", files=_multipart(b"x", "train.txt")
    )

    assert response.status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


async def test_upload_of_an_empty_file_is_422(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/datasets/{dataset.id}/upload", files=_multipart(b"", "train.jsonl")
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_upload_while_uploading_is_409(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    ready_dataset: DatasetTable,
    session: AsyncSession,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)
    ready_dataset.status = DatasetStatus.UPLOADING
    await session.commit()

    response = await client.post(
        f"{API}/datasets/{ready_dataset.id}/upload",
        files=_multipart(b"{}\n", "train.jsonl"),
    )

    assert response.status_code == HTTPStatus.CONFLICT


async def test_upload_over_the_configured_cap_is_413(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
    app: FastAPI,
    settings: Settings,
) -> None:
    """A file over ``dataset_upload_max_bytes`` is rejected mid-stream as 413.

    Overrides ``get_settings`` with a copy pinning the cap to a handful of
    bytes so the endpoint's real streaming path (not just the unit-level
    ``stream_to_staging`` test) proves the 413 mapping end to end.
    """
    _act_as(as_principal, user)
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(
        update={"dataset_upload_max_bytes": 5}
    )

    response = await client.post(
        f"{API}/datasets/{dataset.id}/upload",
        files=_multipart(b"x" * 100, "train.jsonl"),
    )

    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert response.headers["content-type"].startswith(PROBLEM_JSON)


async def test_upload_with_both_validation_file_and_percentage_is_422(
    client: AsyncClient,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    dataset: DatasetTable,
    local_storage: NoOpDatasetUploadRunner,
) -> None:
    _act_as(as_principal, user)

    response = await client.post(
        f"{API}/datasets/{dataset.id}/upload",
        files={
            "train_file": ("train.jsonl", io.BytesIO(b"{}\n"), "application/octet-stream"),
            "validation_file": ("val.jsonl", io.BytesIO(b"{}\n"), "application/octet-stream"),
        },
        data={"validation_percentage": "20"},
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
