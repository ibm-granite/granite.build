# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""SQLAlchemy implementation of the repository protocols."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, String, cast, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError, MultipleResultsFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from autotunex.core.config import ADMIN_ROLE
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
    TrialTable,
    UserTable,
)
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


def _search_pattern(q: str) -> str:
    r"""Return a LIKE pattern matching ``q`` as a literal substring.

    ``%``, ``_`` and the escape character itself are escaped so a user typing a
    wildcard searches for that character, not a wildcard. Use with
    ``.ilike(_search_pattern(q), escape="\\")``.
    """
    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


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

        Loads tasks, trials and their results too — the detail response needs
        them, and a lazy load would raise ``MissingGreenlet`` under async
        SQLAlchemy. ``tasks`` is loaded here rather than in ``_view_shaped``: the
        lean list (``JobSummary``) never nests tasks, only the detail response
        (``JobRead``) does.
        """
        statement = (
            self._view_shaped()
            .options(
                selectinload(JobTable.tasks),
                selectinload(JobTable.trials).selectinload(TrialTable.result),
            )
            .where(JobTable.id == job_id)
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
    ) -> JobTable | None:
        """Return the job whose ``gb_task`` carries ``build_id``; see the Protocol docstring.

        Resolves the build to its job id from ``gb_tasks`` and delegates to
        :meth:`get`, so the detail response's eager-loading and the ``owner_id``
        filter are reused verbatim rather than restated. ``LIMIT 1`` guards the
        pathological duplicate-``build_id`` case without raising, and the scope
        check stays entirely in :meth:`get`.
        """
        job_id = await self._session.scalar(
            select(GbTaskTable.job_id).where(GbTaskTable.build_id == build_id).limit(1)
        )
        if job_id is None:
            return None
        return await self.get(job_id, owner_id=owner_id)

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
        # finished_at = the job's run end for the Total-time column: the latest
        # gb_tasks.updated_at. A correlated scalar subquery keeps this a single
        # round trip — the lean list never loads tasks (see _view_shaped). It is a
        # VARCHAR string; MAX over the zero-padded ISO-8601 gbserver emits is the
        # chronological latest. NULL when the job has no task with an update time.
        finished_at = (
            select(func.max(GbTaskTable.updated_at))
            .where(GbTaskTable.job_id == JobTable.id)
            .correlate(JobTable)
            .scalar_subquery()
            .label("finished_at")
        )
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

        ``results`` is deleted before ``trials`` because ``results.trial_id``
        references it, and every child before ``jobs`` — so the order stays legal
        where FKs are enforced and is harmless where they are not.
        """
        if not await self.is_visible(job_id, owner_id=owner_id):
            return False
        await self._session.execute(
            delete(ResultTable)
            .where(ResultTable.job_id == job_id)
            .execution_options(synchronize_session=False)
        )
        await self._session.execute(
            delete(LogEntryTable)
            .where(LogEntryTable.job_id == job_id)
            .execution_options(synchronize_session=False)
        )
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
    Write-only: a job's trials are read back through
    :meth:`SqlAlchemyJobRepository.get`, which eager-loads them for the detail
    response. Owns its transactions (``commit`` lives here, per the layering
    rule).
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
        self, configuration_id: UUID, *, owner_id: UUID | None = None
    ) -> ConfigurationTable | None:
        """Return the configuration with ``configuration_id`` owned by ``owner_id``."""
        statement = select(ConfigurationTable).where(ConfigurationTable.id == configuration_id)
        if owner_id is not None:
            # Raw, unfolded string comparison, matching SqlAlchemyJobRepository:
            # str(owner_id) is canonical lowercase, and folding would cost the index.
            statement = statement.where(ConfigurationTable.user_id == str(owner_id))
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list(
        self, *, limit: int, offset: int, owner_id: UUID | None = None, q: str | None = None
    ) -> tuple[Sequence[ConfigurationTable], int]:
        """Return one page of configurations, newest first, plus the total.

        ``q``, when given, is a case-insensitive substring filter on ``name``,
        applied to both statements so ``total`` can never disagree with the
        number of items returned.
        """
        total_statement = select(func.count()).select_from(ConfigurationTable)
        page_statement = (
            select(ConfigurationTable).order_by(*_CONFIG_PAGE_ORDER).limit(limit).offset(offset)
        )
        if owner_id is not None:
            total_statement = total_statement.where(ConfigurationTable.user_id == str(owner_id))
            page_statement = page_statement.where(ConfigurationTable.user_id == str(owner_id))
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

    async def get(self, dataset_id: UUID, *, owner_id: UUID | None = None) -> DatasetTable | None:
        """Return the dataset with ``dataset_id`` owned by ``owner_id``, or ``None``."""
        statement = select(DatasetTable).where(DatasetTable.id == dataset_id)
        if owner_id is not None:
            statement = statement.where(DatasetTable.user_id == str(owner_id))
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list(
        self, *, limit: int, offset: int, owner_id: UUID | None = None, q: str | None = None
    ) -> tuple[Sequence[DatasetTable], int]:
        """Return one page of datasets, newest first, plus the total.

        ``q``, when given, is a case-insensitive substring filter on ``name``,
        applied to both statements so ``total`` can never disagree with the
        number of items returned.
        """
        total_statement = select(func.count()).select_from(DatasetTable)
        page_statement = (
            select(DatasetTable).order_by(*_DATASET_PAGE_ORDER).limit(limit).offset(offset)
        )
        if owner_id is not None:
            total_statement = total_statement.where(DatasetTable.user_id == str(owner_id))
            page_statement = page_statement.where(DatasetTable.user_id == str(owner_id))
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

        ``role`` is set explicitly to ``"user"`` rather than relying on the
        column default: a provisioned account must never be an admin, and pinning
        it here keeps that true even if the schema default ever changes. On the
        race where a concurrent first request already inserted the row, the
        ``UNIQUE(email)`` insert fails; this rolls back and re-reads the winner
        (case-insensitively, so a different-cased duplicate is never created)
        rather than surfacing the error.
        """
        user = UserTable(email=email, role="user")
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
        (lower(email))``, tracked under CLAUDE.md open decision 7's schema work;
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
