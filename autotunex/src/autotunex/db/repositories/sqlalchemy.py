# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""SQLAlchemy implementation of the repository protocols."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ColumnElement, Select, String, cast, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError, MultipleResultsFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute, defer, joinedload, selectinload

from autotunex.core.config import ADMIN_ROLE
from autotunex.core.constants import SYSTEM_USER_ID
from autotunex.core.exceptions import (
    AmbiguousIdentityError,
    ConfigurationInUseError,
    ConfigurationNameConflictError,
    DatasetInUseError,
    DatasetNameConflictError,
    JobReferenceConflictError,
)
from autotunex.core.logging import get_logger
from autotunex.db.repositories.protocols import ReconcilableJob
from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    LogEntryTable,
    ResultTable,
    TrainingMetricTable,
    TrialTable,
    UserTable,
)
from autotunex.db.tables._helpers import utcnow
from autotunex.models.status import TERMINAL_RUN_STATUSES, DatasetStatus, GbTaskType, RunStatus

logger = get_logger(__name__)

_PAGE_ORDER = (JobTable.created_at.desc(), JobTable.id.desc())
"""The list page's ordering, newest first with an ``id`` tiebreaker.

``created_at`` alone is not unique, so two jobs sharing one have no defined
relative order without ``id`` — pages could repeat or vanish a row between
requests. Pulled out to a module-level constant so a test can assert on the
exact clause :meth:`SqlAlchemyJobRepository.list` uses, rather than restating
it and asserting on the restatement.
"""

_TRIAL_PAGE_ORDER = (TrialTable.created_at.asc(), TrialTable.id.asc())
"""The trials page's ordering, oldest first with an ``id`` tiebreaker.

Ascending, unlike :data:`_PAGE_ORDER`: trials read as the chronological record of
what the search evaluated, and the UX renders them that way. ``created_at`` alone
is not unique, so the ``id`` tiebreaker is what keeps offset pagination stable —
two trials sharing a timestamp could otherwise repeat or vanish between requests.
``trials.created_at`` is ``NOT NULL``, so there is no dialect divergence over
where NULLs sort (a zero date written by a lax MySQL ``sql_mode`` reads back as
``None`` through ``UtcDateTime``, but it is a real value in the ``ORDER BY``).
Pulled out so a test can assert on the exact clause rather than a restatement of
it.
"""


def _finished_at_column() -> ColumnElement[str | None]:
    """The job's run end as a correlated scalar subquery: ``MAX(gb_tasks.updated_at)``.

    Used by every read that does not load ``tasks`` — the lean list and the
    build-id lookup — so those shapes can still report ``finished_at`` in a single
    round trip. ``gb_tasks.updated_at`` is a ``VARCHAR``; ``MAX`` over the
    zero-padded ISO-8601 gbserver emits is the chronological latest. ``NULL`` when
    the job has no task with an update time.

    A function rather than a module constant because a correlated subquery is
    bound to the statement it is compiled into, and reusing one object across two
    statements risks carrying correlation state between them.
    """
    return (
        select(func.max(GbTaskTable.updated_at))
        .where(GbTaskTable.job_id == JobTable.id)
        .correlate(JobTable)
        .scalar_subquery()
        .label("finished_at")
    )


_DELETE_BATCH_SIZE = 5_000
"""Rows removed per statement when clearing a job's unbounded child tables.

Bounds the lock window of
:meth:`SqlAlchemyJobRepository._delete_children_in_batches`. At stage's observed
~264 bytes/row average this is well under a couple of MB per statement — small
enough to finish far inside MySQL's 50s ``innodb_lock_wait_timeout``, large
enough that clearing millions of rows does not cost millions of round trips.
Module-level so a test can shrink it rather than inserting 5,000 rows to prove
the loop iterates.
"""


def _search_pattern(q: str) -> str:
    r"""Return a LIKE pattern matching ``q`` as a literal substring.

    ``%``, ``_`` and the escape character itself are escaped so a user typing a
    wildcard searches for that character, not a wildcard. Use with
    ``.ilike(_search_pattern(q), escape="\\")``.
    """
    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _owner_or_shared(
    user_id_column: InstrumentedAttribute[str], owner_id: UUID, *, include_shared: bool
) -> ColumnElement[bool]:
    """Ownership predicate: the caller's rows, plus the shared system tier when asked.

    ``include_shared`` widens the filter to rows owned by the reserved system
    user (:data:`~autotunex.core.constants.SYSTEM_USER_ID`) — the curated
    starter configs/datasets every caller may read and use, but that only the
    system user itself (or an admin via ``scope=all``) may modify. Read and
    read-for-use paths pass ``include_shared=True``; every mutation leaves it
    ``False``, so system rows stay read-only by construction.

    The comparison is a raw, unfolded string match, matching
    :class:`SqlAlchemyJobRepository`'s equivalent predicate: ``str(owner_id)``
    is already canonical lowercase, and folding case here would cost the index.
    """
    own = user_id_column == str(owner_id)
    if include_shared:
        return or_(own, user_id_column == str(SYSTEM_USER_ID))
    return own


class SqlAlchemyJobRepository:
    """Job persistence backed by an :class:`AsyncSession`.

    Satisfies :class:`autotunex.db.repositories.protocols.JobRepository`.

    Recomposes the ``autotunex_jobs`` view's query rather than reading the view
    itself — Postgres rejects that view's ``GROUP BY``, and the view multiplies
    job rows once ``gb_tasks`` is joined. Pagination here applies to jobs, and
    tasks arrive in a separate ``SELECT``, so three tasks can never become three
    job rows.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _view_shaped(self) -> Select[tuple[JobTable]]:
        """Return a job select with everything the lean list needs loaded.

        ``innerjoin=True`` reproduces the view's ``INNER JOIN`` semantics: a job
        whose owner, configuration or dataset is missing is invisible, exactly as
        it is today. The parents are many-to-one, so joining them cannot multiply
        rows. ``tasks`` is deliberately not loaded here — the lean list
        (``JobSummary``) does not need it, and dropping its ``selectinload``
        removes a DB round trip from every ``GET /jobs`` page. :meth:`get` adds
        it back for the detail response.
        """
        return select(JobTable).options(
            joinedload(JobTable.user, innerjoin=True),
            joinedload(JobTable.configuration, innerjoin=True),
            joinedload(JobTable.dataset, innerjoin=True),
        )

    def _total_statement(self) -> Select[tuple[int]]:
        """Return the statement :meth:`list` uses to count ``total``.

        Counts through the same three inner joins as the page rather than a bare
        ``COUNT(*)`` over ``jobs``: otherwise a job whose owner, configuration or
        dataset is missing would be counted even though ``innerjoin=True`` hides
        it from the page, and a client would see ``total`` disagree with the
        number of items returned. Pulled out so a test can assert on the real
        statement instead of a copy of it.
        """
        return (
            select(func.count())
            .select_from(JobTable)
            .join(JobTable.user)
            .join(JobTable.configuration)
            .join(JobTable.dataset)
        )

    async def get(self, job_id: UUID, *, owner_id: UUID | None = None) -> JobTable | None:
        """Return the job with ``job_id`` owned by ``owner_id``, or ``None``.

        Loads ``tasks`` too — the detail response nests them, and a lazy load
        would raise ``MissingGreenlet`` under async SQLAlchemy. Loaded here rather
        than in ``_view_shaped`` because the lean list (``JobSummary``) never
        nests tasks, only the detail response (``JobRead``) does.

        Trials are deliberately **not** loaded. They are an unbounded child
        collection, and eager-loading them with their one-to-one ``results`` rows
        cost two further round trips and put every trial's ``config`` and
        ``metrics`` blob into a response that is also polled while a job runs.
        ``JobRead`` no longer carries them; they are paged by
        :meth:`SqlAlchemyTrialRepository.page` behind ``GET /jobs/{id}/trials``.
        ``num_trials`` needs no query at all: it is the trial budget declared by the
        job's own ``config_snapshot``, resolved by
        :func:`~autotunex.services.mappers.resolve_planned_trials`.
        """
        statement = (
            self._view_shaped().options(selectinload(JobTable.tasks)).where(JobTable.id == job_id)
        )
        if owner_id is not None:
            # Deliberate asymmetry with get_by_email below, which folds case on
            # both sides. This comparison does not, and cannot cheaply: the email
            # is free text, whereas ``users.id`` normalizes through ``Uuid36`` so
            # ``str(owner_id)`` is always canonical lowercase. The exposure is the
            # other side — ``jobs.user_id`` is a raw ``VARCHAR(255)`` the tuning
            # pipeline writes, so a job stored with an uppercase UUID there
            # matches on MySQL (case-insensitive collation) and matches nothing on
            # SQLite or Postgres. That fails closed — a user loses sight of their
            # own jobs rather than seeing someone else's — and folding it here
            # would cost the index on every scoped read. Recorded, not fixed.
            statement = statement.where(JobTable.user_id == str(owner_id))
        result = await self._session.execute(statement)
        return result.unique().scalar_one_or_none()

    async def get_by_build_id(
        self, build_id: UUID, *, owner_id: UUID | None = None
    ) -> tuple[JobTable, str | None] | None:
        """Return ``(job, finished_at)`` for the job whose task carries ``build_id``.

        One round trip, not three. This used to resolve the build to a job id and
        then delegate to :meth:`get`, which cost a second query plus a third for
        the ``tasks`` this shape no longer returns. Instead the build filter is a
        join, and ``finished_at`` — the only thing the mapper needed ``tasks`` for —
        arrives as a correlated subquery, exactly as it does for the lean list.

        The join cannot multiply job rows: it is filtered to the single task
        carrying this ``build_id``. ``LIMIT 1`` still guards the pathological
        duplicate-``build_id`` case without raising.
        """
        statement = (
            self._view_shaped()
            .add_columns(_finished_at_column())
            .join(GbTaskTable, GbTaskTable.job_id == JobTable.id)
            .where(GbTaskTable.build_id == build_id)
            .limit(1)
        )
        if owner_id is not None:
            # Raw, unfolded string comparison, for the reasons in :meth:`get`.
            statement = statement.where(JobTable.user_id == str(owner_id))
        result = await self._session.execute(statement)
        row = result.unique().first()
        if row is None:
            return None
        return row[0], row[1]

    async def is_visible(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Cheap ``SELECT`` existence-and-scope probe; see the Protocol docstring."""
        statement = select(JobTable.id).where(JobTable.id == job_id)
        if owner_id is not None:
            # Raw, unfolded str match, for the index/normalization reasons in `get`.
            statement = statement.where(JobTable.user_id == str(owner_id))
        result = await self._session.execute(statement.limit(1))
        return result.scalar_one_or_none() is not None

    async def logs_page(
        self, job_id: UUID, *, trial_id: str | None, before_id: int, limit: int
    ) -> tuple[Sequence[LogEntryTable], bool]:
        """Keyset page of log lines, newest first; see the Protocol docstring."""
        statement = select(LogEntryTable).where(LogEntryTable.job_id == job_id)
        if trial_id is None:
            statement = statement.where(LogEntryTable.trial_id.is_(None))
        else:
            statement = statement.where(LogEntryTable.trial_id == trial_id)
        if before_id > 0:
            statement = statement.where(LogEntryTable.id < before_id)
        statement = statement.order_by(LogEntryTable.id.desc()).limit(limit + 1)
        rows = (await self._session.execute(statement)).scalars().all()
        has_more = len(rows) > limit
        return rows[:limit], has_more

    async def append_log(
        self,
        job_id: UUID,
        *,
        trial_id: str | None,
        level: str | None,
        filename: str | None,
        message: str | None,
        iteration: int | None,
        epoch: float | None,
    ) -> None:
        """Append one log line to ``job_id``, committing; see the Protocol docstring.

        ``timestamp`` is stamped as a *naive* UTC ``datetime`` to match the
        column, which is a bare ``DATETIME`` with no timezone — a mirrored schema
        quirk (see :class:`~autotunex.db.tables.log_entries.LogEntryTable`).
        Stamping it here rather than relying on a column default keeps the value
        deterministic and dialect-independent.
        """
        entry = LogEntryTable(
            job_id=job_id,
            trial_id=trial_id,
            level=level,
            filename=filename,
            message=message,
            iteration=iteration,
            epoch=epoch,
            timestamp=datetime.now(UTC).replace(tzinfo=None),
        )
        self._session.add(entry)
        await self._session.commit()

    async def list(
        self, *, limit: int, offset: int, owner_id: UUID | None = None, q: str | None = None
    ) -> tuple[Sequence[tuple[JobTable, str | None]], int]:
        """Return one page of ``(job, finished_at)`` rows, newest first, plus the total.

        ``q``, when given, is a case-insensitive substring filter matched
        against ``experiment_name``, ``model`` or ``status`` (cast to text,
        since ``status`` is a native ``Enum`` column and PostgreSQL rejects
        ``ILIKE`` against it directly). Applied to both statements so ``total``
        can never disagree with the number of items returned.
        """
        # finished_at keeps this a single round trip — the lean list never loads
        # tasks (see _view_shaped). See _finished_at_column for the semantics.
        finished_at = _finished_at_column()
        total_statement = self._total_statement()
        page_statement = (
            self._view_shaped()
            .add_columns(finished_at)
            .order_by(*_PAGE_ORDER)
            .limit(limit)
            .offset(offset)
        )
        if owner_id is not None:
            # Raw, unfolded string comparison, for the reasons in :meth:`get`.
            # Applied to both statements so ``total`` can never disagree with the
            # number of items returned.
            total_statement = total_statement.where(JobTable.user_id == str(owner_id))
            page_statement = page_statement.where(JobTable.user_id == str(owner_id))
        if q:
            pattern = _search_pattern(q)
            predicate = or_(
                JobTable.experiment_name.ilike(pattern, escape="\\"),
                JobTable.model.ilike(pattern, escape="\\"),
                cast(JobTable.status, String).ilike(pattern, escape="\\"),
            )
            total_statement = total_statement.where(predicate)
            page_statement = page_statement.where(predicate)
        total = await self._session.scalar(total_statement)
        result = await self._session.execute(page_statement)
        rows = result.unique().all()
        return [(row[0], row[1]) for row in rows], total or 0

    async def create(
        self,
        *,
        user_id: str,
        config_id: UUID,
        dataset_id: UUID,
        model: str,
        model_source: str,
        experiment_name: str,
        tuning_type: str | None,
        seed: int,
        autotune: bool,
        config_snapshot: dict[str, Any],
        reward_function_code: str | None,
        reward_function_name: str | None,
    ) -> JobTable:
        """Persist a new job (``status='pending'``) owned by ``user_id``."""
        job = JobTable(
            user_id=user_id,
            config_id=config_id,
            dataset_id=dataset_id,
            model=model,
            model_source=model_source,
            experiment_name=experiment_name,
            tuning_type=tuning_type,
            seed=seed,
            autotune=autotune,
            status=RunStatus.PENDING,
            config_snapshot=config_snapshot,
            reward_function_code=reward_function_code,
            reward_function_name=reward_function_name,
        )
        self._session.add(job)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise JobReferenceConflictError() from exc
        await self._session.commit()
        return job

    async def set_status(self, job_id: UUID, status: RunStatus) -> None:
        """Set the job's status, committing; no-op if the job is gone."""
        result = await self._session.execute(select(JobTable).where(JobTable.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return
        job.status = status
        await self._session.commit()

    async def _delete_children_in_batches(
        self,
        table: type[LogEntryTable] | type[TrainingMetricTable],
        job_id: UUID,
    ) -> None:
        """Delete one high-volume child table's rows for ``job_id`` in bounded batches.

        ``log_entries`` and ``training_metrics`` are the two tables a long tuning
        run grows without bound — stage carried 16.4M ``log_entries`` rows over
        4.1GB on 2026-08-31 — so a single ``DELETE ... WHERE job_id = :id`` takes
        millions of row locks and unlinks gigabytes of off-page ``MEDIUMTEXT`` in
        one statement. That ran past MySQL's 50s ``innodb_lock_wait_timeout``
        against ``src/api-bridge``'s concurrent ``insert_logs`` batches and
        surfaced as ``(1205, 'Lock wait timeout exceeded')``. Batching keeps each
        statement's lock set proportional to :data:`_DELETE_BATCH_SIZE` rather than
        to the job's lifetime row count.

        Selects a page of primary keys and deletes by ``id IN (...)`` rather than
        using ``DELETE ... LIMIT``: that clause is MySQL-only (PostgreSQL rejects
        it outright, and SQLite needs a non-default compile flag), and this
        repository must run on all three. Both tables have an autoincrement
        integer ``id``, so ordering by it makes the paging deterministic.

        Commits per batch, which is what bounds the lock window — see
        :meth:`delete` for why losing single-transaction atomicity is safe here.
        """
        while True:
            ids = (
                (
                    await self._session.execute(
                        select(table.id)
                        .where(table.job_id == job_id)
                        .order_by(table.id)
                        .limit(_DELETE_BATCH_SIZE)
                    )
                )
                .scalars()
                .all()
            )
            if not ids:
                return
            await self._session.execute(
                delete(table).where(table.id.in_(ids)).execution_options(synchronize_session=False)
            )
            await self._session.commit()
            if len(ids) < _DELETE_BATCH_SIZE:
                return

    async def delete(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Delete a job scoped to ``owner_id``, cascading its trials, results, tasks and logs.

        Removes the children with set-based ``DELETE ... WHERE job_id = :id``
        statements rather than loading each child collection and deleting it row
        by row. That keeps the delete's cost — and the lock window it holds —
        independent of how many trials, results or log entries the job
        accumulated: a job with millions of ``log_entries`` no longer hydrates
        them all into memory inside the transaction, which was the cause of the
        minute-long deletes and the ``gb_tasks`` ``Lock wait timeout exceeded``
        seen in production. The explicit per-table deletes work on every dialect
        regardless of FK enforcement (dev/test SQLite runs without
        ``PRAGMA foreign_keys=ON``), so they do not rely on the database's own
        ``ON DELETE CASCADE``; ``synchronize_session=False`` keeps each one a pure
        emit with no pre-``SELECT`` to reconcile the (unused) identity map.

        Set-based was not sufficient on its own: ``log_entries`` and
        ``training_metrics`` grow without bound per job, so those two go through
        :meth:`_delete_children_in_batches` and commit incrementally. The other
        children are bounded by trial count and stay single statements.

        **This is deliberately no longer one atomic transaction.** A single
        transaction is what produced the ``log_entries`` lock-wait timeout, and
        atomicity buys nothing for a delete-everything operation. The order makes
        an interrupted delete safe rather than corrupting: children are removed
        before ``jobs``, so a crash mid-way leaves the job row present and still
        visible, and the caller (or the user) can simply issue the delete again —
        it is idempotent, and each retry starts from a smaller remainder. The
        inverse order would leave unreachable orphans.

        ``results`` is deleted before ``trials`` because ``results.trial_id``
        references it, ``training_metrics`` is included because nothing else
        removes it on a dialect that does not enforce ``ON DELETE CASCADE``, and
        every child precedes ``jobs`` — so the order stays legal where FKs are
        enforced and is harmless where they are not.
        """
        if not await self.is_visible(job_id, owner_id=owner_id):
            return False
        await self._session.execute(
            delete(ResultTable)
            .where(ResultTable.job_id == job_id)
            .execution_options(synchronize_session=False)
        )
        await self._delete_children_in_batches(LogEntryTable, job_id)
        await self._delete_children_in_batches(TrainingMetricTable, job_id)
        await self._session.execute(
            delete(TrialTable)
            .where(TrialTable.job_id == job_id)
            .execution_options(synchronize_session=False)
        )
        await self._session.execute(
            delete(GbTaskTable)
            .where(GbTaskTable.job_id == job_id)
            .execution_options(synchronize_session=False)
        )
        await self._session.execute(
            delete(JobTable)
            .where(JobTable.id == job_id)
            .execution_options(synchronize_session=False)
        )
        await self._session.commit()
        return True

    async def get_task(self, job_id: UUID, task_type: GbTaskType) -> GbTaskTable | None:
        """Return the job's build task of ``task_type``, or ``None``."""
        result = await self._session.execute(
            select(GbTaskTable).where(GbTaskTable.job_id == job_id, GbTaskTable.type == task_type)
        )
        return result.scalars().first()

    async def upsert_task(
        self,
        job_id: UUID,
        task_type: GbTaskType,
        *,
        status: RunStatus,
        build_id: UUID | None = None,
        pr_url: str | None = None,
        build_status: dict[str, Any] | None = None,
        artifact_id: UUID | None = None,
        artifact_uri: str | None = None,
        started_at: str | None = None,
        updated_at: str | None = None,
    ) -> GbTaskTable:
        """Insert or update the job's ``task_type`` build task, committing."""
        now = datetime.now(UTC).isoformat()
        task = await self.get_task(job_id, task_type)
        if task is None:
            task = GbTaskTable(
                job_id=job_id,
                type=task_type,
                status=status,
                build_id=build_id,
                pr_url=pr_url,
                build_status=build_status,
                artifact_id=artifact_id,
                artifact_uri=artifact_uri,
                started_at=started_at or now,
                updated_at=updated_at or now,
            )
            self._session.add(task)
        else:
            task.status = status
            if build_id is not None:
                task.build_id = build_id
            if pr_url is not None:
                task.pr_url = pr_url
            if build_status is not None:
                task.build_status = build_status
            if artifact_id is not None:
                task.artifact_id = artifact_id
            if artifact_uri is not None:
                task.artifact_uri = artifact_uri
            if started_at is not None:
                task.started_at = started_at
            task.updated_at = updated_at if updated_at is not None else now
        await self._session.commit()
        return task

    async def list_reconcilable(self) -> Sequence[ReconcilableJob]:
        """Return non-terminal jobs that have a ``TUNING`` build id to poll."""
        statement = (
            select(JobTable.id, JobTable.status, GbTaskTable.build_id)
            .join(GbTaskTable, GbTaskTable.job_id == JobTable.id)
            .where(
                JobTable.status.not_in(list(TERMINAL_RUN_STATUSES)),
                GbTaskTable.type == GbTaskType.TUNING,
                GbTaskTable.build_id.is_not(None),
            )
        )
        result = await self._session.execute(statement)
        return [
            ReconcilableJob(job_id=row.id, status=row.status, build_id=row.build_id)
            for row in result.all()
        ]


class SqlAlchemyTrialRepository:
    """Trial persistence backed by an :class:`AsyncSession`.

    Satisfies :class:`autotunex.db.repositories.protocols.TrialRepository`.
    Owns its transactions (``commit`` lives here, per the layering rule).

    Reads live here too, in :meth:`page`. They used to go through
    :meth:`SqlAlchemyJobRepository.get`, which eager-loaded a job's whole trial
    list for the detail response; that response no longer carries trials, so
    trial reads belong with trial persistence.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        job_id: UUID,
        trial_id: str,
        *,
        status: RunStatus,
        config: dict[str, Any] | None,
    ) -> None:
        """Insert the trial or update its status and config in place, committing.

        Get-or-create by the primary key ``trial_id`` — no dialect ``ON
        CONFLICT`` — so a repeated report for the same trial updates in place and
        the behaviour is identical on SQLite, PostgreSQL and MySQL.
        """
        trial = await self._session.get(TrialTable, trial_id)
        if trial is None:
            self._session.add(TrialTable(id=trial_id, job_id=job_id, status=status, config=config))
        else:
            trial.status = status
            trial.config = config
        await self._session.commit()

    async def set_status(self, trial_id: str, status: RunStatus) -> None:
        """Set the trial's status, committing; no-op if the trial is gone."""
        trial = await self._session.get(TrialTable, trial_id)
        if trial is None:
            return
        trial.status = status
        await self._session.commit()

    async def fail_running(self, job_id: UUID) -> None:
        """Set every ``running`` trial of ``job_id`` to ``error``, committing.

        A single bulk ``UPDATE`` scoped to the job and the ``running`` status,
        so trials already in a terminal state are untouched. The local runner's
        failure path calls this so a run that dies mid-flight leaves no trial
        stuck in ``running`` while the job itself is ``error``.
        """
        await self._session.execute(
            update(TrialTable)
            .where(TrialTable.job_id == job_id, TrialTable.status == RunStatus.RUNNING)
            .values(status=RunStatus.ERROR)
        )
        await self._session.commit()

    async def terminate_running(self, job_id: UUID) -> None:
        """Set every ``running`` trial of ``job_id`` to ``terminated``, committing.

        The cancellation analogue of :meth:`fail_running`; a single bulk ``UPDATE``
        scoped to the job and the ``running`` status, so terminal trials are
        untouched. The local runner's cancellation path calls this so a cancelled
        run leaves no trial stuck in ``running`` while the job itself is ``terminated``.
        """
        await self._session.execute(
            update(TrialTable)
            .where(TrialTable.job_id == job_id, TrialTable.status == RunStatus.RUNNING)
            .values(status=RunStatus.TERMINATED)
        )
        await self._session.commit()

    async def page(
        self, job_id: UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[TrialTable], int]:
        """Return one page of the job's trials plus the unpaginated total.

        ``joinedload`` for the trial's ``results`` row, not ``selectinload``:
        the relationship is a scalar one-to-one, so the join cannot multiply
        rows, and folding it into this statement costs one round trip instead of
        two. The caller must have verified the job is visible to the principal
        first (``JobRepository.is_visible``) — this method applies no ownership
        filter of its own, exactly like ``logs_page`` and ``metrics_page``.
        """
        total = await self._session.scalar(
            select(func.count()).select_from(TrialTable).where(TrialTable.job_id == job_id)
        )
        statement = (
            select(TrialTable)
            .options(joinedload(TrialTable.result))
            .where(TrialTable.job_id == job_id)
            .order_by(*_TRIAL_PAGE_ORDER)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(statement)
        return result.unique().scalars().all(), total or 0


class SqlAlchemyResultRepository:
    """Result persistence backed by an :class:`AsyncSession`.

    Satisfies :class:`autotunex.db.repositories.protocols.ResultRepository`.
    ``results.trial_id`` is ``UNIQUE`` (one result per trial), so :meth:`upsert`
    updates the existing row in place rather than inserting a duplicate. Owns its
    transactions.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        job_id: UUID,
        trial_id: str,
        *,
        metric: str,
        metrics: dict[str, Any] | None,
    ) -> None:
        """Insert the trial's result or update it in place, committing.

        Looks up the existing row by ``trial_id`` (the ``UNIQUE`` one-to-one key)
        and updates it if present, otherwise inserts. Get-or-create rather than a
        dialect ``ON CONFLICT``, so it works identically across dialects.
        """
        result = (
            await self._session.execute(select(ResultTable).where(ResultTable.trial_id == trial_id))
        ).scalar_one_or_none()
        if result is None:
            self._session.add(
                ResultTable(job_id=job_id, trial_id=trial_id, metric=metric, metrics=metrics)
            )
        else:
            result.metric = metric
            result.metrics = metrics
        await self._session.commit()


class SqlAlchemyTrainingMetricsRepository:
    """Per-step metrics persistence backed by an :class:`AsyncSession`.

    Satisfies :class:`~autotunex.db.repositories.protocols.TrainingMetricsRepository`.
    Append-only; owns its transactions.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(
        self,
        job_id: UUID,
        *,
        trial_id: str | None,
        global_step: int,
        epoch: float | None,
        loss: float | None,
        grad_norm: float | None,
        learning_rate: float | None,
        split: str,
        extra: dict[str, Any] | None,
    ) -> None:
        """Append one metrics row to ``job_id``, committing."""
        self._session.add(
            TrainingMetricTable(
                job_id=job_id,
                trial_id=trial_id,
                global_step=global_step,
                epoch=epoch,
                loss=loss,
                grad_norm=grad_norm,
                learning_rate=learning_rate,
                split=split,
                extra=extra,
            )
        )
        await self._session.commit()

    async def metrics_page(
        self, job_id: UUID, *, trial_id: str | None, after_id: int, limit: int
    ) -> tuple[Sequence[TrainingMetricTable], bool]:
        """Ascending keyset page (oldest first) for charting; see the Protocol."""
        statement = select(TrainingMetricTable).where(TrainingMetricTable.job_id == job_id)
        if trial_id is not None:
            statement = statement.where(TrainingMetricTable.trial_id == trial_id)
        if after_id > 0:
            statement = statement.where(TrainingMetricTable.id > after_id)
        statement = statement.order_by(TrainingMetricTable.id.asc()).limit(limit + 1)
        rows = (await self._session.execute(statement)).scalars().all()
        has_more = len(rows) > limit
        return rows[:limit], has_more


_CONFIG_PAGE_ORDER = (ConfigurationTable.created_at.desc(), ConfigurationTable.id.desc())
"""Newest-first ordering for the configuration list, with an ``id`` tiebreaker.

Same rationale as :data:`_PAGE_ORDER`: ``created_at`` alone is not unique, so two
rows sharing one need ``id`` to have a stable relative order across pages.
"""


class SqlAlchemyConfigurationRepository:
    """Configuration persistence backed by an :class:`AsyncSession`.

    Satisfies :class:`autotunex.db.repositories.protocols.ConfigurationRepository`.

    Owns transactions for the write path (``commit`` lives here, per the layering
    rule), and translates the two database-level constraint violations into the
    domain exceptions the service expects. That translation is per method rather
    than by inspecting the error text: on ``create``/``update`` the only
    integrity constraint a well-formed call can hit is ``UNIQUE (user_id, name)``
    — ``user_id`` is a resolved principal that already exists in ``users`` — and
    on ``delete`` it is the ``ON DELETE RESTRICT`` from ``jobs.config_id``.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        configuration_id: UUID,
        *,
        owner_id: UUID | None = None,
        include_shared: bool = False,
    ) -> ConfigurationTable | None:
        """Return the configuration with ``configuration_id``, scoped to ``owner_id``.

        With ``include_shared`` and an ``owner_id`` set, rows owned by the
        reserved system user (the shared starter-content tier) are visible too.
        ``include_shared`` is ignored when ``owner_id`` is ``None`` — the
        admin/standalone unscoped view already returns every row.
        """
        statement = select(ConfigurationTable).where(ConfigurationTable.id == configuration_id)
        if owner_id is not None:
            statement = statement.where(
                _owner_or_shared(
                    ConfigurationTable.user_id, owner_id, include_shared=include_shared
                )
            )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        limit: int,
        offset: int,
        owner_id: UUID | None = None,
        q: str | None = None,
        include_shared: bool = False,
    ) -> tuple[Sequence[ConfigurationTable], int]:
        """Return one page of configurations, newest first, plus the total.

        ``q``, when given, is a case-insensitive substring filter on ``name``,
        applied to both statements so ``total`` can never disagree with the
        number of items returned. ``include_shared`` widens the ``owner_id``
        filter to also admit the reserved system user's rows (the shared
        starter-content tier); ignored when ``owner_id`` is ``None``.

        Per the Protocol, ``config_data`` is not loaded on these rows and touching
        it raises rather than fetching it — see the ``defer`` call below.
        """
        total_statement = select(func.count()).select_from(ConfigurationTable)
        page_statement = (
            select(ConfigurationTable)
            # `config_data` is the whole search space, and no caller of this list
            # wants it — the API's list shape drops it and the frontend refetches
            # the detail when it needs one. Deferring stops it being selected at
            # all, so a page of 20 no longer reads, transfers and parses 20 JSON
            # blobs; trimming only the response shape would still pay for every
            # one of those. `raiseload` makes an accidental access raise instead
            # of quietly issuing a fresh SELECT per row — an N+1 that tests would
            # not notice and production would feel. `get`/`create`/`update` are
            # untouched; they legitimately need the column.
            .options(defer(ConfigurationTable.config_data, raiseload=True))
            .order_by(*_CONFIG_PAGE_ORDER)
            .limit(limit)
            .offset(offset)
        )
        if owner_id is not None:
            predicate = _owner_or_shared(
                ConfigurationTable.user_id, owner_id, include_shared=include_shared
            )
            total_statement = total_statement.where(predicate)
            page_statement = page_statement.where(predicate)
        if q:
            predicate = ConfigurationTable.name.ilike(_search_pattern(q), escape="\\")
            total_statement = total_statement.where(predicate)
            page_statement = page_statement.where(predicate)
        total = await self._session.scalar(total_statement)
        result = await self._session.execute(page_statement)
        return result.scalars().all(), total or 0

    async def create(
        self,
        *,
        user_id: str,
        name: str,
        tuner_type: str | None,
        rl_tuner_type: str | None,
        config_data: dict[str, Any],
    ) -> ConfigurationTable:
        """Persist a new configuration owned by ``user_id``."""
        configuration = ConfigurationTable(
            user_id=user_id,
            name=name,
            tuner_type=tuner_type,
            rl_tuner_type=rl_tuner_type,
            config_data=config_data,
        )
        self._session.add(configuration)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ConfigurationNameConflictError(name) from exc
        await self._session.commit()
        return configuration

    async def update(
        self,
        configuration_id: UUID,
        *,
        owner_id: UUID | None = None,
        name: str,
        tuner_type: str | None,
        rl_tuner_type: str | None,
        config_data: dict[str, Any],
    ) -> ConfigurationTable | None:
        """Fully replace a configuration's mutable fields, scoped to ``owner_id``."""
        configuration = await self.get(configuration_id, owner_id=owner_id)
        if configuration is None:
            return None
        configuration.name = name
        configuration.tuner_type = tuner_type
        configuration.rl_tuner_type = rl_tuner_type
        configuration.config_data = config_data
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ConfigurationNameConflictError(name) from exc
        await self._session.commit()
        return configuration

    async def delete(self, configuration_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Delete a configuration scoped to ``owner_id``, returning whether a row went."""
        configuration = await self.get(configuration_id, owner_id=owner_id)
        if configuration is None:
            return False
        await self._session.delete(configuration)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ConfigurationInUseError(configuration_id) from exc
        await self._session.commit()
        return True

    async def jobs_for_config(
        self, config_ids: Sequence[UUID], *, owner_id: UUID | None = None
    ) -> dict[UUID, builtins.list[JobTable]]:
        """Return jobs per configuration id, scoped to ``owner_id``."""
        if not config_ids:
            return {}
        statement = select(JobTable).where(JobTable.config_id.in_(list(config_ids)))
        if owner_id is not None:
            statement = statement.where(JobTable.user_id == str(owner_id))
        result = await self._session.execute(statement)
        grouped: dict[UUID, builtins.list[JobTable]] = {}
        for job in result.scalars().all():
            grouped.setdefault(job.config_id, []).append(job)
        return grouped


_DATASET_PAGE_ORDER = (DatasetTable.created_at.desc(), DatasetTable.id.desc())
"""Newest-first dataset ordering with an ``id`` tiebreaker; see :data:`_PAGE_ORDER`."""


class SqlAlchemyDatasetRepository:
    """Dataset persistence backed by an :class:`AsyncSession`.

    Satisfies :class:`autotunex.db.repositories.protocols.DatasetRepository`.
    Owns transactions for the write path. ``create``/``update`` refresh the
    generated ``train_file``/``validation_file`` columns after flush so the
    returned object carries them.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        dataset_id: UUID,
        *,
        owner_id: UUID | None = None,
        include_shared: bool = False,
    ) -> DatasetTable | None:
        """Return the dataset with ``dataset_id``, scoped to ``owner_id``.

        With ``include_shared`` and an ``owner_id`` set, system-owned datasets
        (the shared starter tier) are visible too; ignored when ``owner_id`` is
        ``None``.
        """
        statement = select(DatasetTable).where(DatasetTable.id == dataset_id)
        if owner_id is not None:
            statement = statement.where(
                _owner_or_shared(DatasetTable.user_id, owner_id, include_shared=include_shared)
            )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        limit: int,
        offset: int,
        owner_id: UUID | None = None,
        q: str | None = None,
        include_shared: bool = False,
    ) -> tuple[Sequence[DatasetTable], int]:
        """Return one page of datasets, newest first, plus the total.

        ``q``, when given, is a case-insensitive substring filter on ``name``,
        applied to both statements so ``total`` can never disagree with the
        number of items returned. ``include_shared`` widens the ``owner_id``
        filter to also admit the reserved system user's rows (the shared
        starter-content tier); ignored when ``owner_id`` is ``None``.
        """
        total_statement = select(func.count()).select_from(DatasetTable)
        page_statement = (
            select(DatasetTable).order_by(*_DATASET_PAGE_ORDER).limit(limit).offset(offset)
        )
        if owner_id is not None:
            predicate = _owner_or_shared(
                DatasetTable.user_id, owner_id, include_shared=include_shared
            )
            total_statement = total_statement.where(predicate)
            page_statement = page_statement.where(predicate)
        if q:
            predicate = DatasetTable.name.ilike(_search_pattern(q), escape="\\")
            total_statement = total_statement.where(predicate)
            page_statement = page_statement.where(predicate)
        total = await self._session.scalar(total_statement)
        result = await self._session.execute(page_statement)
        return result.scalars().all(), total or 0

    async def create(
        self, *, user_id: str, name: str, description: str | None, data_format: str
    ) -> DatasetTable:
        """Persist a new dataset owned by ``user_id`` (``status='empty'``)."""
        dataset = DatasetTable(
            user_id=user_id, name=name, description=description, data_format=data_format
        )
        self._session.add(dataset)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise DatasetNameConflictError(name) from exc
        await self._session.commit()
        await self._session.refresh(dataset, ["status", "train_file", "validation_file"])
        return dataset

    async def update(
        self,
        dataset_id: UUID,
        *,
        owner_id: UUID | None = None,
        name: str,
        description: str | None,
        data_format: str,
    ) -> DatasetTable | None:
        """Fully replace a dataset's mutable metadata, scoped to ``owner_id``."""
        dataset = await self.get(dataset_id, owner_id=owner_id)
        if dataset is None:
            return None
        dataset.name = name
        dataset.description = description
        dataset.data_format = data_format
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise DatasetNameConflictError(name) from exc
        await self._session.commit()
        await self._session.refresh(dataset, ["train_file", "validation_file"])
        return dataset

    async def delete(self, dataset_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Delete a dataset scoped to ``owner_id``, returning whether a row went."""
        dataset = await self.get(dataset_id, owner_id=owner_id)
        if dataset is None:
            return False
        await self._session.delete(dataset)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise DatasetInUseError(dataset_id) from exc
        await self._session.commit()
        return True

    async def set_status(
        self, dataset_id: UUID, status: DatasetStatus, *, status_detail: str | None = None
    ) -> None:
        """Set the dataset's status and optional detail, committing.

        Refreshes ``train_file``/``validation_file`` after commit, like
        ``create``/``update``: any ``UPDATE`` on this table re-expires those
        ``Computed`` columns regardless of the columns actually written, and
        the caller may read them from this same object right after (the
        upload endpoint does, via ``dataset_to_read``) — an unrefreshed
        expired attribute would otherwise raise ``MissingGreenlet`` on the
        next synchronous access under the async driver.
        """
        dataset = await self.get(dataset_id)
        if dataset is None:
            return
        dataset.status = status
        dataset.status_detail = status_detail
        await self._session.commit()
        await self._session.refresh(dataset, ["train_file", "validation_file"])

    async def set_upload_result(
        self,
        dataset_id: UUID,
        *,
        train_records: int,
        train_file_size: int,
        validation_records: int | None,
        validation_file_size: int | None,
        data_format: str,
        artifact_id: UUID | None,
        artifact_url: str | None,
        status: DatasetStatus = DatasetStatus.READY,
    ) -> None:
        """Record a completed upload's counts, sizes and artifact refs, committing."""
        dataset = await self.get(dataset_id)
        if dataset is None:
            return
        dataset.train_records = train_records
        dataset.train_file_size = train_file_size
        dataset.validation_records = validation_records
        dataset.validation_file_size = validation_file_size
        dataset.data_format = data_format
        dataset.artifact_id = artifact_id
        dataset.artifact_url = artifact_url
        dataset.status = status
        dataset.status_detail = None
        await self._session.commit()
        await self._session.refresh(dataset, ["train_file", "validation_file"])

    async def jobs_for_dataset(
        self, dataset_ids: Sequence[UUID], *, owner_id: UUID | None = None
    ) -> dict[UUID, builtins.list[JobTable]]:
        """Return referencing jobs per dataset id, scoped to ``owner_id``."""
        if not dataset_ids:
            return {}
        statement = select(JobTable).where(JobTable.dataset_id.in_(list(dataset_ids)))
        if owner_id is not None:
            statement = statement.where(JobTable.user_id == str(owner_id))
        result = await self._session.execute(statement)
        grouped: dict[UUID, builtins.list[JobTable]] = {}
        for job in result.scalars().all():
            grouped.setdefault(job.dataset_id, []).append(job)
        return grouped


_USER_PAGE_ORDER = (UserTable.created_at.desc(), UserTable.id.desc())
"""Users list ordering, newest first with an ``id`` tiebreaker (see _PAGE_ORDER)."""


class SqlAlchemyUserRepository:
    """Satisfies :class:`autotunex.db.repositories.protocols.UserRepository`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def provision(self, email: str) -> UserTable:
        """Get-or-create the ``users`` row for an already-verified ``email``.

        ``last_login_at`` is stamped here because reaching provisioning *is* a
        successful authentication — a first-time caller has just logged in, and
        leaving it ``None`` until the throttle window lapsed would show a brand-new
        user a blank "last login" while they were demonstrably using the app.

        ``role`` is set explicitly to ``"user"`` rather than relying on the
        column default: a provisioned account must never be an admin, and pinning
        it here keeps that true even if the schema default ever changes. On the
        race where a concurrent first request already inserted the row, the
        ``UNIQUE(email)`` insert fails; this rolls back and re-reads the winner
        (case-insensitively, so a different-cased duplicate is never created)
        rather than surfacing the error.
        """
        user = UserTable(email=email, role="user", last_login_at=utcnow())
        self._session.add(user)
        try:
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            existing = await self.get_by_email(email)
            if existing is not None:
                return existing
            raise
        await self._session.commit()
        return user

    async def touch_login(self, email: str) -> None:
        """Stamp ``users.last_login_at`` for ``email`` with the current time.

        A single ``UPDATE`` keyed on ``func.lower(email)`` — the same predicate
        :meth:`get_by_email` uses, so the two agree across dialects — rather than
        loading the row and mutating it. That keeps this to one round trip on a
        path that runs per request, and confines the write to the one column: an
        ORM flush would emit whatever else happened to be dirty in the shared
        request session.

        Silently affects no rows when the email is unknown, which is the intended
        no-op for an authenticated-but-unprovisioned caller.

        ``updated_at`` is expected to move as a side effect — the ORM's
        ``onupdate`` fires here, and the live MySQL schema's ``ON UPDATE
        CURRENT_TIMESTAMP`` would regardless. That is correct and harmless: the
        row really did change. The invariant that matters runs the other way, and
        holds by construction — nothing but this method and :meth:`provision`
        writes ``last_login_at``, so a role change can never masquerade as a login.
        """
        await self._session.execute(
            update(UserTable)
            .where(func.lower(UserTable.email) == email.lower())
            .values(last_login_at=utcnow())
        )
        await self._session.commit()

    async def get_by_email(self, email: str) -> UserTable | None:
        """Return the user with ``email``, comparing case-insensitively.

        ``func.lower`` on both sides makes SQLite, Postgres and MySQL agree —
        MySQL's default collation already folds case, the other two do not.

        Exactly one row or none, per the Protocol's multiplicity contract.
        Several matching rows is a data bug in the deployment: ``users.email
        UNIQUE`` is case-*sensitive* on SQLite and Postgres, so ``Alice@example.com``
        and ``alice@example.com`` coexist there and this lookup matches both. It
        raises rather than choosing, because the rows can carry different
        ``role`` values and a tiebreak would decide admin-ness by row order while
        hiding the duplication. The root fix is a ``UNIQUE INDEX ON users
        (lower(email))``, tracked under CLAUDE.md open decision 6's schema work;
        this method only has to fail safely until then.

        Raises:
            AmbiguousIdentityError: several rows matched ``email``.
        """
        result = await self._session.execute(
            select(UserTable).where(func.lower(UserTable.email) == email.lower())
        )
        try:
            return result.scalar_one_or_none()
        except MultipleResultsFound as exc:
            # Logging the email is deliberate and safe: an Authenticator has
            # already verified it by the time stage two calls this, so it is a
            # resolved identity, not an unverified credential. It is also the
            # operator's only route to the offending rows — the client-facing
            # detail says nothing about the duplication.
            logger.warning(
                "Ambiguous identity: several users rows match %s case-insensitively. "
                "De-duplicate them; until then every request from this caller fails.",
                email,
            )
            raise AmbiguousIdentityError() from exc

    async def list(self, *, limit: int, offset: int) -> tuple[Sequence[UserTable], int]:
        """Return one page of users, newest first, plus the total."""
        total = await self._session.scalar(select(func.count()).select_from(UserTable))
        result = await self._session.execute(
            select(UserTable).order_by(*_USER_PAGE_ORDER).limit(limit).offset(offset)
        )
        return result.scalars().all(), total or 0

    async def get(self, user_id: UUID) -> UserTable | None:
        """Return the user with ``user_id``, or ``None``."""
        return await self._session.get(UserTable, user_id)

    async def set_role(self, user_id: UUID, role: str) -> UserTable | None:
        """Set a user's ``role``, returning the refreshed row, or ``None`` if absent."""
        user = await self._session.get(UserTable, user_id)
        if user is None:
            return None
        user.role = role
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def count_admins(self) -> int:
        """Return how many users are admins."""
        total = await self._session.scalar(
            select(func.count()).select_from(UserTable).where(UserTable.role == ADMIN_ROLE)
        )
        return total or 0

    async def metadata(self, user_id: UUID) -> tuple[int, int, int]:
        """Return ``(jobs, configurations, datasets)`` counts for ``user_id``.

        ``user_id`` is compared as a string because the child tables declare
        ``user_id`` as ``VARCHAR`` while ``users.id`` is a UUID (see the
        ``UserTable`` relationship notes).
        """
        owner = str(user_id)
        jobs = await self._session.scalar(
            select(func.count()).select_from(JobTable).where(JobTable.user_id == owner)
        )
        configurations = await self._session.scalar(
            select(func.count())
            .select_from(ConfigurationTable)
            .where(ConfigurationTable.user_id == owner)
        )
        datasets = await self._session.scalar(
            select(func.count()).select_from(DatasetTable).where(DatasetTable.user_id == owner)
        )
        return jobs or 0, configurations or 0, datasets or 0
