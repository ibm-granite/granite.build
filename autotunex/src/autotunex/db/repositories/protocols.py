# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Repository interfaces.

Services depend on these Protocols, never on a concrete implementation. Any
object with matching methods satisfies them — no inheritance required — which
keeps test doubles trivial (see ``tests/services/test_jobs.py``).
"""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    LogEntryTable,
    TrainingMetricTable,
    TrialTable,
    UserTable,
)
from autotunex.models.status import DatasetStatus, GbTaskType, RunStatus


@dataclass(frozen=True)
class ReconcilableJob:
    """A job the reconcile loop should poll: its id, current status, and build id.

    A projection over ``jobs`` joined to its ``TUNING`` ``gb_task``, not a full
    ORM row — the loop needs only these three fields, and ``status`` is what
    ``check_transition`` gates the write on.
    """

    job_id: UUID
    status: RunStatus
    build_id: UUID


class JobRepository(Protocol):
    """Persistence operations for jobs and their trials."""

    async def get(self, job_id: UUID, *, owner_id: UUID | None = None) -> JobTable | None:
        """Return the job with ``job_id`` owned by ``owner_id``, or ``None``.

        ``owner_id=None`` applies no ownership filter — the admin/standalone
        case. A non-``None`` value that does not match the job's owner
        behaves exactly like an unknown id.
        """
        ...

    async def get_by_build_id(
        self, build_id: UUID, *, owner_id: UUID | None = None
    ) -> tuple[JobTable, str | None] | None:
        """Return ``(job, finished_at)`` for the job carrying ``build_id``, or ``None``.

        Locates a job by its granite.build ``build_id`` (stored on
        ``gb_tasks.build_id``) rather than by its own id. Deliberately loads
        **less** than :meth:`get`: no ``tasks`` collection, because the response
        shape for this lookup (``JobDetail``) does not nest them — a caller that
        arrived by build id already holds the field that array exists to expose.
        ``finished_at`` therefore comes back as a computed value rather than being
        derived from loaded tasks, in the same ``(job, finished_at)`` pair shape
        :meth:`list` returns.

        ``owner_id`` scopes the result exactly as :meth:`get` does: ``None``
        applies no filter (admin/standalone), and a non-``None`` value that does
        not match the job's owner yields ``None`` — identical to an unknown build
        id, so a scoped caller cannot tell them apart.
        """
        ...

    async def list(
        self, *, limit: int, offset: int, owner_id: UUID | None = None, q: str | None = None
    ) -> tuple[Sequence[tuple[JobTable, str | None]], int]:
        """Return one page of ``(job, finished_at)`` rows, newest first, plus the total.

        ``finished_at`` is the latest ``gb_tasks.updated_at`` for the job (its run
        end, for the Total-time column), computed by the page query so the lean list
        stays a single round trip without loading ``tasks``. ``owner_id=None``
        applies no ownership filter. The filter applies to both the page and the
        total count, so the two numbers never disagree.
        ``q``, when given, is a case-insensitive substring filter over
        ``experiment_name``, ``model`` and ``status``, applied to both the page
        and the total count for the same reason.
        """
        ...

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
        """Persist a new job (``status='pending'``) owned by ``user_id``.

        Raises:
            JobReferenceConflictError: the referenced configuration or dataset
                was deleted between validation and insert.
        """
        ...

    async def set_status(self, job_id: UUID, status: RunStatus) -> None:
        """Set the job's status, committing. No-op if the job is gone.

        Transition legality is the service's concern (``check_transition``); this
        method only writes. Used by the runner's failure path and, later, the
        status-reconcile follow-on.
        """
        ...

    async def delete(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Delete a job scoped to ``owner_id``, cascading its trials, results and tasks.

        Returns ``True`` if a row was deleted, ``False`` if none matched under the
        ``owner_id`` filter — the caller maps ``False`` to "not found".
        ``owner_id=None`` applies no ownership filter (admin/standalone).

        Unlike the configuration and dataset deletes, no ``IntegrityError``
        translation is needed: nothing references ``jobs`` with ``ON DELETE
        RESTRICT`` (those FKs point the other way, from ``jobs`` to its config and
        dataset), so a job delete cannot violate a constraint.
        """
        ...

    async def get_task(self, job_id: UUID, task_type: GbTaskType) -> GbTaskTable | None:
        """Return the job's build task of ``task_type``, or ``None``."""
        ...

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
        """Insert or update the job's ``task_type`` build task, committing.

        A ``None`` ``build_id``/``pr_url``/``build_status``/``artifact_id``/
        ``artifact_uri``/``started_at`` on update leaves the stored value untouched
        (so a status-only update never clobbers a recorded handle). ``updated_at``
        defaults to now() when not supplied; the reconcile loop passes the cluster's
        own timestamps. ``build_status`` carries the transformed cluster response
        the reconcile loop persists on a terminal state; ``artifact_id`` /
        ``artifact_uri`` identify the produced model, extracted from that response
        when the build ends. Upsert, not blind insert, so a retry does not
        duplicate the row.
        """
        ...

    async def list_reconcilable(self) -> Sequence[ReconcilableJob]:
        """Return non-terminal jobs that have a ``TUNING`` build id to poll.

        Jobs whose ``status NOT IN (completed, error, terminated)``, joined to
        their ``TUNING`` ``gb_task`` where ``build_id IS NOT NULL``. This is the
        reconcile loop's entire per-sweep working set: nothing is cached between
        sweeps, so a restart resumes exactly where it left off.
        """
        ...

    async def is_visible(self, job_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Return whether ``job_id`` exists and is visible to ``owner_id``.

        A cheap existence-and-scope probe for the log endpoints, which verify the
        parent job before querying ``log_entries`` and may be polled frequently
        during a live run — reusing the view-shaped ``get`` here would issue its
        joins and child selectinloads on every poll for no benefit. ``owner_id=None``
        applies no ownership filter (admin/standalone), exactly as ``get``.
        """
        ...

    async def logs_page(
        self,
        job_id: UUID,
        *,
        trial_id: str | None,
        before_id: int,
        limit: int,
    ) -> tuple[Sequence[LogEntryTable], bool]:
        """Return one keyset page of a job's log lines, newest first, plus ``has_more``.

        ``trial_id=None`` selects job-level lines (``log_entries.trial_id IS NULL``);
        a non-``None`` value selects that trial's lines. Both are scoped to
        ``job_id``. ``before_id <= 0`` returns the latest page; a positive value
        returns lines with ``id < before_id``. Fetches ``limit + 1`` rows to derive
        ``has_more`` without a second ``COUNT``; the caller receives rows trimmed to
        ``limit``. Ownership is not checked here — the service verifies the parent
        job with ``is_visible`` first.
        """
        ...

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
        """Append one log line to ``job_id``, committing.

        The write counterpart to :meth:`logs_page`: the local runner's log
        bridge calls this from a worker thread for every line the tuning
        pipeline emits. ``trial_id=None`` records a job-level line; a non-``None``
        value attributes the line to that trial, matched by plain string
        equality (``log_entries`` carries no foreign key to ``trials`` — a
        mirrored schema quirk). ``timestamp`` is stamped server-side, so callers
        never pass one.
        """
        ...


class TrialRepository(Protocol):
    """Persistence for a job's trials — the writes, and the paged read.

    The local runner's :class:`~autotunex.services.local.protocols.TrialSink`
    persists each trial's lifecycle through the write methods as Ray reports it.
    :meth:`page` is the only read: trials used to be eager-loaded by
    :meth:`JobRepository.get` for the job detail response, which is why this
    Protocol was once write-only, but that response no longer carries them.
    """

    async def upsert(
        self,
        job_id: UUID,
        trial_id: str,
        *,
        status: RunStatus,
        config: dict[str, Any] | None,
    ) -> None:
        """Insert the trial or update its status and config in place, committing.

        Keyed on the pipeline-assigned ``trial_id`` (the primary key), so a
        repeated report for the same trial updates rather than duplicating.
        Get-or-create rather than a dialect ``ON CONFLICT``, so the behaviour is
        identical on SQLite, PostgreSQL and MySQL.
        """
        ...

    async def set_status(self, trial_id: str, status: RunStatus) -> None:
        """Set the trial's status, committing. No-op if the trial is gone.

        Transition legality is the caller's concern; this method only writes.
        Used to move a trial to a terminal state and by the runner's failure
        path to sweep still-``running`` trials to ``error``.
        """
        ...

    async def fail_running(self, job_id: UUID) -> None:
        """Set every ``running`` trial of ``job_id`` to ``error``, committing.

        A bulk write scoped to the job and the ``running`` status: trials in a
        terminal state are left untouched. The local runner's failure path calls
        this so a crashed run leaves no trial stuck in ``running`` while the job
        itself has already moved to ``error``.
        """
        ...

    async def terminate_running(self, job_id: UUID) -> None:
        """Set every ``running`` trial of ``job_id`` to ``terminated``, committing.

        The cancellation analogue of :meth:`fail_running`: a bulk write scoped to
        the job and the ``running`` status, leaving already-terminal trials
        untouched, so a cancelled run leaves no trial stuck ``running``.
        """
        ...

    async def page(
        self, job_id: UUID, *, limit: int, offset: int
    ) -> tuple[Sequence[TrialTable], int]:
        """Return one page of the job's trials, oldest first, plus the total.

        Each trial arrives with its one-to-one ``results`` row loaded, so the
        caller can report the trial's metrics without a second lookup. Ordered
        ``created_at`` ascending with an ``id`` tiebreaker, which is what makes
        the offsets stable across requests.

        Applies **no** ownership filter — like :meth:`JobRepository.logs_page`,
        the caller is responsible for having established that the job is visible
        to the principal (``is_visible``) before paging its children. An unknown
        ``job_id`` yields an empty page and a total of 0, not an error: a job with
        no trials yet is not a missing job.
        """
        ...


class ResultRepository(Protocol):
    """Write operations for a trial's one-to-one result row.

    ``results.trial_id`` is ``UNIQUE`` — one result per trial — so :meth:`upsert`
    updates the existing row rather than inserting a second when a trial reports
    fresh metrics.
    """

    async def upsert(
        self,
        job_id: UUID,
        trial_id: str,
        *,
        metric: str,
        metrics: dict[str, Any] | None,
    ) -> None:
        """Insert the trial's result or update it in place, committing.

        Respects the ``results.trial_id`` ``UNIQUE`` constraint: an existing row
        for ``trial_id`` is updated, never duplicated. Get-or-create rather than
        a dialect ``ON CONFLICT``, so it works identically across dialects.
        """
        ...


class TrainingMetricsRepository(Protocol):
    """Write + keyset-read for the per-step ``training_metrics`` time series."""

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
        ...

    async def metrics_page(
        self, job_id: UUID, *, trial_id: str | None, after_id: int, limit: int
    ) -> tuple[Sequence[TrainingMetricTable], bool]:
        """Ascending keyset page of metrics rows (oldest first) for charting."""
        ...


class ConfigurationRepository(Protocol):
    """Persistence operations for configurations.

    The read methods mirror :class:`JobRepository`'s ``owner_id``-scoped
    signatures exactly: ``owner_id=None`` applies no ownership filter (the
    admin/standalone case), and a non-``None`` value that does not match a
    configuration's owner behaves like an unknown id.

    The write methods translate the two database constraints into domain
    exceptions so a service never sees a raw ``IntegrityError``:
    ``UNIQUE (user_id, name)`` becomes
    :class:`~autotunex.core.exceptions.ConfigurationNameConflictError`, and the
    ``ON DELETE RESTRICT`` from ``jobs.config_id`` becomes
    :class:`~autotunex.core.exceptions.ConfigurationInUseError`.
    """

    async def get(
        self,
        configuration_id: UUID,
        *,
        owner_id: UUID | None = None,
        include_shared: bool = False,
    ) -> ConfigurationTable | None:
        """Return the configuration with ``configuration_id``, scoped to ``owner_id``.

        ``include_shared`` additionally admits rows owned by the reserved system
        user; ignored when ``owner_id`` is ``None``.
        """
        ...

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

        The ``owner_id`` filter applies to both the page and the total count;
        ``include_shared`` widens it to the system tier. ``owner_id=None``
        applies no filter.

        **The returned rows do not carry ``config_data``**, and this is part of the
        contract rather than an implementation detail of one backend. Callers get
        the scalar columns; anything needing the search space must read the row
        through :meth:`get`. An implementation is expected to leave the column
        unfetched — that omission is the whole point of the list path, since the
        blob is by far the heaviest thing on the table — and to make an access of
        it *fail* rather than silently fetch it per row. So a caller that touches
        ``config_data`` on one of these rows should expect an exception, not a
        value and not a hidden query.
        """
        ...

    async def create(
        self,
        *,
        user_id: str,
        name: str,
        tuner_type: str | None,
        rl_tuner_type: str | None,
        config_data: dict[str, Any],
    ) -> ConfigurationTable:
        """Persist a new configuration owned by ``user_id``.

        Raises:
            ConfigurationNameConflictError: ``user_id`` already owns a
                configuration named ``name``.
        """
        ...

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
        """Fully replace a configuration's mutable fields, scoped to ``owner_id``.

        Ownership (``user_id``) and identity are never changed. Returns ``None``
        if no configuration matches ``configuration_id`` under the ``owner_id``
        filter — the caller maps that to "not found".

        Raises:
            ConfigurationNameConflictError: the new ``name`` collides with
                another configuration owned by the same user.
        """
        ...

    async def delete(self, configuration_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Delete a configuration scoped to ``owner_id``.

        Returns ``True`` if a row was deleted, ``False`` if none matched under
        the ``owner_id`` filter — the caller maps ``False`` to "not found".

        Raises:
            ConfigurationInUseError: a job still references the configuration.
        """
        ...

    async def jobs_for_config(
        self, config_ids: Sequence[UUID], *, owner_id: UUID | None = None
    ) -> dict[UUID, builtins.list[JobTable]]:
        """Return jobs ("tunings") per configuration id, scoped to ``owner_id``.

        Mirrors :meth:`DatasetRepository.jobs_for_dataset`. ``owner_id=None``
        applies no ownership filter (the admin ``scope=all`` case); an empty
        ``config_ids`` returns ``{}``.

        ``builtins.list`` (rather than the bare builtin) is required here too:
        this class also declares a method named ``list``, which shadows the
        builtin name for annotations resolved in the class's namespace — the
        same reason :meth:`DatasetRepository.jobs_for_dataset` needs it.
        """
        ...


class DatasetRepository(Protocol):
    """Persistence operations for datasets.

    Read methods mirror :class:`ConfigurationRepository`'s ``owner_id``-scoped
    signatures exactly. Write methods translate the two DB constraints into
    domain exceptions: ``UNIQUE (user_id, name)`` →
    :class:`~autotunex.core.exceptions.DatasetNameConflictError`, and
    ``jobs.dataset_id``'s ``ON DELETE RESTRICT`` →
    :class:`~autotunex.core.exceptions.DatasetInUseError`. ``set_status`` and
    ``set_upload_result`` are the runner's two write-backs; they are separate
    from ``update`` because they touch server-owned columns a client never sends.
    """

    async def get(
        self,
        dataset_id: UUID,
        *,
        owner_id: UUID | None = None,
        include_shared: bool = False,
    ) -> DatasetTable | None:
        """Return the dataset with ``dataset_id``, scoped to ``owner_id``.

        ``include_shared`` additionally admits rows owned by the reserved
        system user; ignored when ``owner_id`` is ``None``.
        """
        ...

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

        The ``owner_id`` filter applies to both the page and the count. ``q``,
        when given, is a case-insensitive substring filter on ``name``, applied
        to both for the same reason. ``include_shared`` widens the ``owner_id``
        filter to the system tier.
        """
        ...

    async def create(
        self, *, user_id: str, name: str, description: str | None, data_format: str
    ) -> DatasetTable:
        """Persist a new dataset (``status='empty'``) owned by ``user_id``.

        Raises:
            DatasetNameConflictError: ``user_id`` already owns a dataset named ``name``.
        """
        ...

    async def update(
        self,
        dataset_id: UUID,
        *,
        owner_id: UUID | None = None,
        name: str,
        description: str | None,
        data_format: str,
    ) -> DatasetTable | None:
        """Fully replace a dataset's mutable metadata, scoped to ``owner_id``.

        Ownership, status and upload results are untouched. Returns ``None`` if
        no dataset matches under the filter.

        Raises:
            DatasetNameConflictError: the new ``name`` collides for the same owner.
        """
        ...

    async def delete(self, dataset_id: UUID, *, owner_id: UUID | None = None) -> bool:
        """Delete a dataset scoped to ``owner_id``; ``False`` if none matched.

        Raises:
            DatasetInUseError: a job still references the dataset.
        """
        ...

    async def set_status(
        self, dataset_id: UUID, status: DatasetStatus, *, status_detail: str | None = None
    ) -> None:
        """Set the dataset's status (and optional safe detail), committing.

        Also refreshes the ``Computed`` ``train_file``/``validation_file``
        columns on the in-memory row after the write: any ``UPDATE`` on this
        table re-expires them regardless of which columns were actually
        written, and a caller (``DatasetService.upload``, via
        ``dataset_to_read``) may read them from this same object right after.
        An implementation must refresh them too, or a synchronous access under
        an async driver raises ``MissingGreenlet``.
        """
        ...

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
        """Record a completed upload's record counts, sizes and artifact refs.

        Also refreshes the ``Computed`` ``train_file``/``validation_file``
        columns on the in-memory row after the write, for the same reason
        :meth:`set_status` does — preserve that in any implementation.
        """
        ...

    async def jobs_for_dataset(
        self, dataset_ids: Sequence[UUID], *, owner_id: UUID | None = None
    ) -> dict[UUID, builtins.list[JobTable]]:
        """Return referencing jobs per dataset id, scoped to ``owner_id``.

        Keyed by ``dataset_id``; a dataset with no referencing (visible) jobs is
        absent from the mapping. ``owner_id=None`` applies no filter (admin).

        ``builtins.list`` (rather than the bare builtin) is required here: this
        class also declares a method named ``list``, which shadows the builtin
        name for annotations resolved in the class's namespace.
        """
        ...


class UserRepository(Protocol):
    """Persistence operations for ``users``.

    Beyond principal resolution (``provision``, ``get_by_email``), this also
    backs the admin-facing user-management endpoints: listing, single-row
    lookup, role changes, and the counts a delete/demote confirmation needs.
    """

    async def provision(self, email: str) -> UserTable:
        """Get-or-create the ``users`` row for an already-verified ``email``.

        Called only when just-in-time provisioning is enabled and
        :meth:`get_by_email` found no row. The created row is always
        ``role='user'`` — never admin — so provisioning can never escalate
        privilege. Race-safe: if a concurrent first request wins the
        ``UNIQUE(email)`` insert, this re-reads and returns that row rather than
        raising. ``email`` must be non-empty and already verified by an
        ``Authenticator``, exactly as :meth:`get_by_email` requires.
        """
        ...

    async def touch_login(self, email: str) -> None:
        """Record that ``email`` just authenticated, in ``users.last_login_at``.

        Called on a completed browser login and, throttled, on any authenticated
        request (``api.deps.get_principal``) so callers who never pass through
        the login flow — API keys, OIDC bearer tokens — still register activity.

        Matches ``email`` case-insensitively, exactly as :meth:`get_by_email`
        must. A no-op when no row matches: an authenticated caller without a
        ``users`` row (provisioning off) has no login to record, and that is not
        an error. Implementations must write ``last_login_at`` and nothing else —
        ``role`` in particular — since this runs on the read path.
        """
        ...

    async def get_by_email(self, email: str) -> UserTable | None:
        """Return the user with ``email``, or ``None``.

        The lookup is case-insensitive regardless of database dialect — a
        contract implementations must uphold, not merely a MySQL default.

        Multiplicity is equally part of the contract, and the case-insensitivity
        above is what makes it reachable: exactly one row, or none. An
        implementation that matches several must raise rather than guess. The
        matched rows can carry different ``role`` values, so choosing between
        them would settle whether the caller is an admin by row order — and it
        would bury the real problem. Duplicate case-variant emails are a data bug
        in the deployment (``users.email UNIQUE`` is case-sensitive on SQLite and
        Postgres), and failing closed is what makes that bug visible.

        Raises:
            AmbiguousIdentityError: several rows matched ``email``.
        """
        ...

    async def list(self, *, limit: int, offset: int) -> tuple[Sequence[UserTable], int]:
        """Return one page of users, newest first, plus the total count.

        Unlike the owned-resource repositories this takes no ``owner_id`` filter:
        users are not owner-scoped, and the ``require_admin`` route gate is what
        authorizes reaching it. Ordered newest-first with an ``id`` tiebreaker,
        as the other list methods are.
        """
        ...

    async def get(self, user_id: UUID) -> UserTable | None:
        """Return the user with ``user_id``, or ``None``."""
        ...

    async def set_role(self, user_id: UUID, role: str) -> UserTable | None:
        """Set a user's ``role`` and return the refreshed row, or ``None`` if absent.

        The domain guardrails (own-role, last-admin) live in ``UserService``;
        this method only performs the write.
        """
        ...

    async def count_admins(self) -> int:
        """Return how many users have ``role == 'admin'`` — backs the last-admin guard."""
        ...

    async def metadata(self, user_id: UUID) -> tuple[int, int, int]:
        """Return ``(jobs, configurations, datasets)`` counts owned by ``user_id``."""
        ...
