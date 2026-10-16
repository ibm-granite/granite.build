"""``GET /jobs`` returns what the ``autotunex_jobs`` view describes.

The view itself is not read — Postgres rejects its ``GROUP BY``, and joining
``gb_tasks`` multiplied job rows. These tests pin the shape and the fix.
"""

from __future__ import annotations

from uuid import uuid4

from httpx import AsyncClient
from sqlalchemy import event
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from autotunex.db.repositories.sqlalchemy import SqlAlchemyJobRepository
from autotunex.db.tables import JobTable
from autotunex.models.status import RunStatus
from tests.conftest import API


async def test_list_jobs_reports_the_owner_email_as_user(
    client: AsyncClient, job: JobTable
) -> None:
    response = await client.get(f"{API}/jobs?scope=all")

    assert response.json()["items"][0]["user"] == "tester@example.com"


async def test_list_jobs_reports_the_configuration_name(client: AsyncClient, job: JobTable) -> None:
    response = await client.get(f"{API}/jobs?scope=all")

    assert response.json()["items"][0]["config_name"] == "lora-sweep"


async def test_list_jobs_reports_the_dataset_name(client: AsyncClient, job: JobTable) -> None:
    response = await client.get(f"{API}/jobs?scope=all")

    assert response.json()["items"][0]["dataset"] == "alpaca"


async def test_status_is_lowercase_in_the_response(client: AsyncClient, job: JobTable) -> None:
    response = await client.get(f"{API}/jobs?scope=all")

    assert response.json()["items"][0]["status"] == "pending"


async def test_snapshot_config_name_wins_in_the_response(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    job.config_snapshot = {"name": "as-it-ran"}
    session.add(job)
    await session.commit()

    response = await client.get(f"{API}/jobs?scope=all")

    assert response.json()["items"][0]["config_name"] == "as-it-ran"


async def test_the_page_query_inner_joins_its_three_parents(session: AsyncSession) -> None:
    """The view uses INNER JOIN for user, configuration and dataset.

    Enforced foreign keys mean no job is currently missing a parent, so this
    cannot be observed through a persisted row — asserting on the emitted SQL is
    what keeps the loader strategy from silently degrading to LEFT OUTER JOIN,
    which would scan more than it needs to.
    """
    repository = SqlAlchemyJobRepository(session=session)

    compiled = str(repository._view_shaped().compile(dialect=sqlite.dialect()))

    assert compiled.count("JOIN users") == 1
    assert "LEFT OUTER JOIN users" not in compiled
    assert "LEFT OUTER JOIN configurations" not in compiled
    assert "LEFT OUTER JOIN datasets" not in compiled


async def test_summary_omits_heavy_and_detail_only_fields(
    client: AsyncClient, job: JobTable
) -> None:
    """The lean list carries identity + labels only.

    Tasks, the JSON blobs, and the runtime/type fields are fetched from
    ``GET /jobs/{id}``.
    """
    response = await client.get(f"{API}/jobs?scope=all")

    item = response.json()["items"][0]
    for absent in (
        "tasks",
        "config_snapshot",
        "output_artifacts",
        "num_trials",
        "model_source",
        "tuning_type",
        "rl_tuner_type",
        "ray_address",
        "cleanup",
        "autotune",
        "is_stale",
    ):
        assert absent not in item
    for present in (
        "id",
        "user_id",
        "status",
        "seed",
        "config_id",
        "config_name",
        "dataset_id",
        "dataset",
        "model",
        "experiment_name",
        "user",
        "created_at",
        "updated_at",
    ):
        assert present in item


async def test_pagination_is_stable_for_jobs_sharing_a_created_at(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    """Without the id tiebreaker, rows repeat or vanish between pages.

    This row-level check is insufficient on its own: SQLite happens to return a
    small table in insertion order, so ``set(first_ids).isdisjoint(second_ids)``
    can hold even without the ``id`` tiebreaker. Kept because it is cheap and not
    harmful, but ``test_the_list_query_orders_by_created_at_then_id`` below is
    what actually attests to the tiebreaker existing.
    """
    for index in range(4):
        session.add(
            JobTable(
                id=uuid4(),
                user_id=job.user_id,
                status=RunStatus.PENDING,
                config_id=job.config_id,
                dataset_id=job.dataset_id,
                model="m",
                model_source="huggingface",
                experiment_name=f"exp-{index}",
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
        )
    await session.commit()

    first = await client.get(f"{API}/jobs", params={"limit": 2, "offset": 0, "scope": "all"})
    second = await client.get(f"{API}/jobs", params={"limit": 2, "offset": 2, "scope": "all"})

    first_ids = [item["id"] for item in first.json()["items"]]
    second_ids = [item["id"] for item in second.json()["items"]]
    assert set(first_ids).isdisjoint(second_ids)


async def test_the_list_query_orders_by_created_at_then_id(
    client: AsyncClient, engine: AsyncEngine, job: JobTable
) -> None:
    """The page query's ``ORDER BY`` names both ``created_at`` and ``id``.

    ``created_at`` alone is not unique, so two jobs sharing one have no defined
    relative order without ``id`` — pages could repeat or vanish a row between
    requests. SQLite happens to return a small table in insertion order, so a
    row-level pagination test can pass even with the tiebreaker missing; only
    the ordering clause itself proves the guarantee. Captured from a real
    ``GET /jobs`` call — not a re-declared statement — so this fails if
    :meth:`SqlAlchemyJobRepository.list` ever stops using ``_PAGE_ORDER``.
    """
    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _capture(conn: object, cursor: object, statement: str, *_args: object) -> None:
        statements.append(statement)

    await client.get(f"{API}/jobs")

    page_query = next(s for s in statements if "ORDER BY" in s)
    assert "ORDER BY jobs.created_at DESC, jobs.id DESC" in page_query


async def test_total_equals_the_item_count_on_a_full_page(
    client: AsyncClient, session: AsyncSession, job: JobTable
) -> None:
    """``total`` is counted through the same joins as the page, so the two agree.

    A bare ``COUNT(*)`` over ``jobs`` would drift from the item count the moment
    the page's joins excluded anything. The schema's foreign keys mean no job can
    currently be excluded, so this pins the invariant rather than a discrepancy.
    ``test_total_is_counted_through_the_same_joins_as_the_page`` below is what
    actually attests to the joins existing in the count statement.
    """
    for index in range(2):
        session.add(
            JobTable(
                id=uuid4(),
                user_id=job.user_id,
                status=RunStatus.PENDING,
                config_id=job.config_id,
                dataset_id=job.dataset_id,
                model="m",
                model_source="huggingface",
                experiment_name=f"exp-{index}",
            )
        )
    await session.commit()

    response = await client.get(f"{API}/jobs?scope=all")

    body = response.json()
    assert body["total"] == len(body["items"]) == 3


async def test_total_is_counted_through_the_same_joins_as_the_page(
    client: AsyncClient, engine: AsyncEngine, job: JobTable
) -> None:
    """``total`` must not be able to drift from the item count.

    A bare ``COUNT(*)`` over ``jobs`` would count rows the page's inner joins
    exclude. Enforced foreign keys mean nothing is currently excluded, so the two
    counts agree on any constructible data — the joins in the count statement are
    what keep that true if a parent ever becomes optional. Captured from a real
    ``GET /jobs`` call so this fails if the count statement is ever simplified
    back to a bare ``COUNT(*)`` over ``jobs``.
    """
    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _capture(conn: object, cursor: object, statement: str, *_args: object) -> None:
        statements.append(statement)

    await client.get(f"{API}/jobs")

    count_query = next(s for s in statements if "count(*)" in s.lower())
    assert "JOIN users" in count_query
    assert "JOIN configurations" in count_query
    assert "JOIN datasets" in count_query
