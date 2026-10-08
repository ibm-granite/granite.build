"""Scope rules and mapping for ``TunedModelService``, over a hand-written fake repository."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from autotunex.core.exceptions import ScopeNotPermittedError
from autotunex.db.tables import ConfigurationTable, JobTable, UserTable
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope
from autotunex.models.status import RunStatus
from autotunex.services.tuned_models import TunedModelService

URI = "hf://huggingface.co/models/example-org/autotunex_aaaa0001"


class FakeTunedModelRepository:
    """Satisfies ``TunedModelRepository``; records each call and returns canned rows."""

    def __init__(self, rows: Sequence[tuple[JobTable, str | None, str]] = ()) -> None:
        self.rows = list(rows)
        self.calls: list[dict[str, object]] = []

    async def tuned_models(
        self,
        *,
        namespaces: Sequence[str],
        limit: int,
        offset: int,
        owner_id: UUID | None = None,
        q: str | None = None,
    ) -> tuple[Sequence[tuple[JobTable, str | None, str]], int]:
        self.calls.append(
            {
                "namespaces": list(namespaces),
                "limit": limit,
                "offset": offset,
                "owner_id": owner_id,
                "q": q,
            }
        )
        return self.rows, len(self.rows)


def _principal(*, user_id: UUID | None, is_admin: bool = False) -> Principal:
    return Principal(
        email="tester@example.com", provider="session", user_id=user_id, is_admin=is_admin
    )


def _job() -> JobTable:
    """A transient job with the relationships the mapper reads."""
    return JobTable(
        id=uuid4(),
        user_id=str(uuid4()),
        status=RunStatus.COMPLETED,
        config_id=uuid4(),
        dataset_id=uuid4(),
        model="ibm-granite/granite-4.0-h-micro",
        model_source="huggingface",
        experiment_name="grpo-run",
        tuning_type=None,
        config_snapshot={"rl_tuner_type": "grpo"},
        user=UserTable(id=uuid4(), email="owner@example.com", role="user"),
        configuration=ConfigurationTable(id=uuid4(), user_id="x", name="c", config_data={"a": 1}),
    )


async def test_search_refuses_the_cross_user_view_to_a_non_admin() -> None:
    service = TunedModelService(
        FakeTunedModelRepository(), _principal(user_id=uuid4()), ["example-org"]
    )

    with pytest.raises(ScopeNotPermittedError):
        await service.search(scope=DataScope.ALL)


async def test_search_returns_an_empty_page_for_an_unresolvable_caller_without_querying() -> None:
    repository = FakeTunedModelRepository()
    service = TunedModelService(repository, _principal(user_id=None), ["example-org"])

    page = await service.search(limit=5, offset=10)

    assert (page.items, page.total, page.limit, page.offset) == ([], 0, 5, 10)
    assert repository.calls == []


async def test_search_returns_an_empty_page_when_no_namespace_is_allowlisted() -> None:
    repository = FakeTunedModelRepository()
    service = TunedModelService(repository, _principal(user_id=uuid4()), [])

    page = await service.search()

    assert page.total == 0
    assert repository.calls == []


async def test_search_scopes_to_the_caller_and_passes_the_allowlist() -> None:
    repository = FakeTunedModelRepository()
    caller = uuid4()
    service = TunedModelService(repository, _principal(user_id=caller), ["example-org"])

    await service.search(limit=7, offset=3, q="sft")

    assert repository.calls == [
        {"namespaces": ["example-org"], "limit": 7, "offset": 3, "owner_id": caller, "q": "sft"}
    ]


async def test_search_lets_an_admin_see_every_owner() -> None:
    repository = FakeTunedModelRepository()
    service = TunedModelService(
        repository, _principal(user_id=uuid4(), is_admin=True), ["example-org"]
    )

    await service.search(scope=DataScope.ALL)

    assert repository.calls[0]["owner_id"] is None


async def test_search_maps_a_row_to_a_submittable_repo_id() -> None:
    job = _job()
    service = TunedModelService(
        FakeTunedModelRepository([(job, "2026-09-20T10:00:00Z", URI)]),
        _principal(user_id=uuid4()),
        ["example-org"],
    )

    page = await service.search()

    assert page.model_dump(mode="json")["items"] == [
        {
            "job_id": str(job.id),
            "repo_id": "example-org/autotunex_aaaa0001",
            "model_source": "huggingface",
            "experiment_name": "grpo-run",
            "base_model": "ibm-granite/granite-4.0-h-micro",
            "tuning_type": None,
            "rl_tuner_type": "grpo",
            "finished_at": "2026-09-20T10:00:00Z",
            "user": "owner@example.com",
        }
    ]


async def test_search_drops_a_row_whose_repo_the_launch_path_would_not_bind() -> None:
    # SQL LIKE is case-insensitive on SQLite and MySQL; is_allowlisted is not.
    mis_cased = "hf://huggingface.co/models/Example-Org/autotunex_aaaa0002"
    service = TunedModelService(
        FakeTunedModelRepository([(_job(), None, mis_cased)]),
        _principal(user_id=uuid4()),
        ["example-org"],
    )

    page = await service.search()

    assert page.items == []
