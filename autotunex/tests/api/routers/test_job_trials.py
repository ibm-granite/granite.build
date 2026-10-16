"""``GET /jobs/{id}/trials`` — the job's trials, paged.

Trials used to be nested on the job detail response, which made every read of a
job pay for every trial's ``config`` and ``metrics`` blob. The assertions on the
trial shape (the ``results`` merge, ``config`` replacing the old ``params``)
moved here from ``test_jobs_detail.py`` when they moved endpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from uuid import uuid4

from httpx import AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from autotunex.db.tables import JobTable, ResultTable, TrialTable, UserTable
from autotunex.models.auth import Principal
from autotunex.models.status import RunStatus
from tests.conftest import API


def _act_as(as_principal: Callable[[Principal], None], user: UserTable) -> None:
    """Resolve every request to ``user`` — a provisioned, non-admin owner."""
    as_principal(Principal(email=user.email, provider="session", user_id=user.id, is_admin=False))


async def test_a_job_with_no_trials_yet_is_an_empty_page_not_a_404(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable, job: JobTable
) -> None:
    """The pre-rewrite implementation 404'd here, conflating "no trials" with "no job"."""
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/{job.id}/trials")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


async def test_it_lists_the_job_s_trials(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    session.add_all(
        [
            TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED),
            TrialTable(id="t2", job_id=job.id, status=RunStatus.RUNNING),
        ]
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/trials")

    body = response.json()
    assert {trial["id"] for trial in body["items"]} == {"t1", "t2"}
    assert body["total"] == 2


async def test_trials_come_back_oldest_first(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """Chronological, which is the order the search evaluated them in."""
    _act_as(as_principal, user)
    session.add_all(
        [
            TrialTable(
                id="later",
                job_id=job.id,
                status=RunStatus.COMPLETED,
                created_at=datetime(2026, 3, 2, tzinfo=UTC),
            ),
            TrialTable(
                id="earlier",
                job_id=job.id,
                status=RunStatus.COMPLETED,
                created_at=datetime(2026, 3, 1, tzinfo=UTC),
            ),
        ]
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/trials")

    assert [trial["id"] for trial in response.json()["items"]] == ["earlier", "later"]


async def test_the_page_is_bounded_by_limit_while_total_counts_them_all(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """The point of the split: a caller asks for a page, not the whole search."""
    _act_as(as_principal, user)
    session.add_all(
        [
            TrialTable(
                id=f"t{index}",
                job_id=job.id,
                status=RunStatus.COMPLETED,
                created_at=datetime(2026, 3, 1, 0, index, tzinfo=UTC),
            )
            for index in range(5)
        ]
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/trials", params={"limit": 2, "offset": 2})

    body = response.json()
    assert [trial["id"] for trial in body["items"]] == ["t2", "t3"]
    assert body["total"] == 5
    assert (body["limit"], body["offset"]) == (2, 2)


async def test_trial_metrics_come_from_the_results_row(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """``metrics`` lives on ``results``, not on ``trials``."""
    _act_as(as_principal, user)
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    session.add(
        ResultTable(
            id=uuid4(),
            job_id=job.id,
            trial_id="t1",
            metric="eval_loss",
            metrics={"eval_loss": 0.42},
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/trials")

    trial = response.json()["items"][0]
    assert trial["metrics"] == {"eval_loss": 0.42}
    assert trial["metric"] == "eval_loss"


async def test_a_trial_with_no_result_has_empty_metrics(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.RUNNING))
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/trials")

    trial = response.json()["items"][0]
    assert trial["metrics"] == {}
    assert trial["metric"] is None


async def test_trial_config_replaces_the_scaffold_s_params(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    _act_as(as_principal, user)
    session.add(
        TrialTable(
            id="t1", job_id=job.id, status=RunStatus.COMPLETED, config={"learning_rate": 3e-5}
        )
    )
    await session.commit()

    response = await client.get(f"{API}/jobs/{job.id}/trials")

    trial = response.json()["items"][0]
    assert trial["config"] == {"learning_rate": 3e-5}
    assert "params" not in trial


async def test_trials_of_an_unknown_job_are_a_404_problem_detail(client: AsyncClient) -> None:
    response = await client.get(f"{API}/jobs/{uuid4()}/trials")

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_trials_of_another_users_job_are_a_404(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    job: JobTable,
) -> None:
    """Indistinguishable from an unknown job — the trials must not leak either."""
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    as_principal(
        Principal(email="other@example.com", provider="session", user_id=uuid4(), is_admin=False)
    )

    response = await client.get(f"{API}/jobs/{job.id}/trials")

    assert response.status_code == HTTPStatus.NOT_FOUND


async def test_a_non_admin_asking_for_scope_all_is_forbidden(
    client: AsyncClient, as_principal: Callable[[Principal], None], user: UserTable, job: JobTable
) -> None:
    _act_as(as_principal, user)

    response = await client.get(f"{API}/jobs/{job.id}/trials", params={"scope": "all"})

    assert response.status_code == HTTPStatus.FORBIDDEN


async def test_an_admin_with_scope_all_reads_another_owners_trials(
    client: AsyncClient,
    session: AsyncSession,
    as_principal: Callable[[Principal], None],
    job: JobTable,
) -> None:
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    as_principal(
        Principal(email="admin@example.com", provider="session", user_id=uuid4(), is_admin=True)
    )

    response = await client.get(f"{API}/jobs/{job.id}/trials", params={"scope": "all"})

    assert response.status_code == HTTPStatus.OK
    assert [trial["id"] for trial in response.json()["items"]] == ["t1"]


def _capture_selects(engine: AsyncEngine, statements: list[str]) -> None:
    """Record every ``SELECT`` the engine issues, for the round-trip assertions."""

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _capture(conn: object, cursor: object, statement: str, *_args: object) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)


async def test_the_detail_endpoint_no_longer_queries_trials_or_results(
    client: AsyncClient,
    session: AsyncSession,
    engine: AsyncEngine,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """``GET /jobs/{id}`` costs two SELECTs: the job with its joins, then tasks.

    This is the whole point of the split, and the only assertion that actually
    pins it — dropping the field from ``JobRead`` without dropping the eager load
    would still serialize leanly while paying for every trial and result row.
    """
    _act_as(as_principal, user)
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    statements: list[str] = []
    _capture_selects(engine, statements)

    await client.get(f"{API}/jobs/{job.id}")

    assert len(statements) == 2
    assert not any("FROM trials" in s for s in statements)
    assert not any("FROM results" in s for s in statements)


async def test_the_trials_page_joins_the_results_row_rather_than_selecting_it_separately(
    client: AsyncClient,
    session: AsyncSession,
    engine: AsyncEngine,
    as_principal: Callable[[Principal], None],
    user: UserTable,
    job: JobTable,
) -> None:
    """``results`` is a scalar one-to-one, so a join cannot multiply rows.

    Pinning the join keeps the page at three round trips (visibility probe,
    count, page) instead of four — a ``selectinload`` here would be a silent
    regression that no response-shape assertion would catch.
    """
    _act_as(as_principal, user)
    session.add(TrialTable(id="t1", job_id=job.id, status=RunStatus.COMPLETED))
    await session.commit()
    statements: list[str] = []
    _capture_selects(engine, statements)

    await client.get(f"{API}/jobs/{job.id}/trials")

    assert len(statements) == 3
    page_query = next(s for s in statements if "ORDER BY" in s)
    assert "JOIN results" in page_query
    assert "ORDER BY trials.created_at ASC, trials.id ASC" in page_query
