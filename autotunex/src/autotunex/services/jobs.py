# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Job business logic.

This layer owns every domain rule about jobs. It knows nothing about HTTP: it
raises the exceptions in :mod:`autotunex.core.exceptions` and lets the API layer
translate them.
"""

from __future__ import annotations

from typing import Any, Literal, overload
from uuid import UUID

from autotunex.core.exceptions import (
    BuildNotFoundError,
    CallerNotProvisionedError,
    ConfigurationNotFoundError,
    DatasetNotFoundError,
    DatasetNotReadyForJobError,
    InvalidStateTransitionError,
    JobNotCancellableError,
    JobNotFoundError,
    MissingRewardFunctionError,
)
from autotunex.db.repositories.protocols import (
    ConfigurationRepository,
    DatasetRepository,
    JobRepository,
)
from autotunex.models.auth import Principal
from autotunex.models.common import DataScope, Page
from autotunex.models.job import (
    ALLOWED_JOB_TRANSITIONS,
    ONLINE_RL_TUNER_TYPES,
    TERMINAL_JOB_STATUSES,
    JobCreate,
    JobDetail,
    JobRead,
    JobShape,
    JobSummary,
)
from autotunex.models.status import DatasetStatus, GbTaskType, RunStatus
from autotunex.services.mappers import (
    job_to_detail,
    job_to_read,
    job_to_summary,
    latest_task_update,
)
from autotunex.services.protocols import JobRunner
from autotunex.services.scoping import resolve_owner_filter, sees_nothing


class JobService:
    """Creates, reads, and lists tuning jobs, scoped to the calling principal.

    Submission (:meth:`create`) validates caller-scoped ownership of the
    referenced configuration and dataset, requires the dataset to be ready, and
    hands the accepted job to the ``JobRunner`` seam. Jobs may also be written
    directly by the tuning pipeline.
    """

    def __init__(
        self,
        repository: JobRepository,
        configuration_repository: ConfigurationRepository,
        dataset_repository: DatasetRepository,
        principal: Principal,
        runner: JobRunner,
    ) -> None:
        self._repository = repository
        self._configuration_repository = configuration_repository
        self._dataset_repository = dataset_repository
        self._principal = principal
        self._runner = runner

    # Overloaded so that ``shape`` selects the *static* return type, not just the
    # runtime one. Without this every caller of the default shape — the chat
    # assistant's get_job tool reads ``job.tasks`` — would see the union and need a
    # narrowing assert to reach a JobRead-only field. The third signature is what
    # the router matches: it passes a runtime ``JobShape``, not a literal.
    @overload
    async def get(
        self,
        job_id: UUID,
        *,
        scope: DataScope = ...,
        shape: Literal[JobShape.FULL] = ...,
    ) -> JobRead: ...

    @overload
    async def get(
        self, job_id: UUID, *, scope: DataScope = ..., shape: Literal[JobShape.LEAN]
    ) -> JobDetail: ...

    @overload
    async def get(
        self, job_id: UUID, *, scope: DataScope = ..., shape: JobShape
    ) -> JobRead | JobDetail: ...

    async def get(
        self,
        job_id: UUID,
        *,
        scope: DataScope = DataScope.OWN,
        shape: JobShape = JobShape.FULL,
    ) -> JobRead | JobDetail:
        """Return the job with ``job_id``, scoped to the caller.

        With the default ``scope=own`` the caller sees only its own job; an
        admin passing ``scope=all`` may fetch any owner's. A non-admin passing
        ``scope=all`` is refused (403) before any row is read.

        ``shape`` chooses how much of the job to report and is orthogonal to
        ``scope``: :attr:`~autotunex.models.job.JobShape.FULL` (the default) returns
        :class:`JobRead`, and ``LEAN`` returns :class:`JobDetail` — the job's own
        record without the nested ``tasks`` array or the ``config_snapshot`` blob.

        ``LEAN`` trims the *response*, not the query. The repository still eager-loads
        ``tasks`` either way, so both shapes cost the same two round trips; do not read
        this parameter as a performance switch. Skipping the load would mean returning
        ``(job, finished_at)`` from ``JobRepository.get`` — ``finished_at`` cannot be
        derived from an unloaded collection without a ``MissingGreenlet`` — which is a
        Protocol change rippling through every caller of that method. Because tasks
        *are* loaded, ``LEAN`` derives ``finished_at`` from them exactly as ``FULL``
        does, so the field holds the same value in both shapes. That is the one way
        this differs from :meth:`get_by_build_id`, which returns the same
        :class:`JobDetail` shape but computes ``finished_at`` by subquery.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it belongs to someone else.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise JobNotFoundError(job_id)
        job = await self._repository.get(job_id, owner_id=owner_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if shape is JobShape.LEAN:
            return job_to_detail(job, finished_at=latest_task_update(job.tasks))
        return job_to_read(job)

    async def get_by_build_id(
        self, build_id: UUID, *, scope: DataScope = DataScope.OWN
    ) -> JobDetail:
        """Return the job whose build task carries ``build_id``, scoped to the caller.

        Locates the job by its granite.build ``build_id`` (stored on ``gb_tasks``)
        instead of its own id, and applies the same scoping as :meth:`get`: own
        data by default, an admin may pass ``scope=all``, and a non-admin passing
        it is refused (403) before any row is read. A build whose job belongs to
        someone else (under ``scope=own``) is a 404, indistinguishable from an
        unknown build.

        Returns the leaner :class:`JobDetail`, **not** :class:`JobRead`: this
        lookup omits the nested build ``tasks`` array and the ``config_snapshot``
        blob. A caller arriving by build id already holds the identifier the tasks
        array exists to expose, and the snapshot embeds the whole configuration as
        it ran — by far the heaviest field on the response. Both remain on
        ``GET /jobs/{id}``. Dropping ``tasks`` also takes this lookup from three
        round trips to one.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            BuildNotFoundError: no job has this build id, or it is another owner's.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise BuildNotFoundError(build_id)
        found = await self._repository.get_by_build_id(build_id, owner_id=owner_id)
        if found is None:
            raise BuildNotFoundError(build_id)
        job, finished_at = found
        return job_to_detail(job, finished_at)

    async def list(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        scope: DataScope = DataScope.OWN,
        q: str | None = None,
    ) -> Page[JobSummary]:
        """Return one page of jobs, newest first — own rows by default.

        ``q`` is an optional case-insensitive substring filter on the job's
        experiment name, model, or status.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            return Page[JobSummary](items=[], total=0, limit=limit, offset=offset)
        rows, total = await self._repository.list(
            limit=limit, offset=offset, owner_id=owner_id, q=q
        )
        return Page[JobSummary](
            items=[job_to_summary(job, finished_at) for job, finished_at in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def create(self, data: JobCreate) -> JobRead:
        """Submit a tuning job owned by the calling principal.

        Ownership comes from ``principal.user_id``, never the request. Both the
        configuration and the dataset are re-verified against the caller's own
        ownership or the shared system tier (``include_shared=True`` widens the
        lookup only that far — even an admin may not reference another real
        owner's configuration or dataset), the dataset must be ``ready``, and
        an online-RL configuration must carry a reward function. The created
        job is always owned by the caller regardless of which tier the
        configuration or dataset came from, and the configuration is
        snapshotted so later edits do not rewrite what this job ran.

        Raises:
            CallerNotProvisionedError: the caller has no ``user_id`` to own the job.
            ConfigurationNotFoundError: the config is unknown or another user's.
            DatasetNotFoundError: the dataset is unknown or another user's.
            DatasetNotReadyForJobError: the dataset's upload has not finished.
            MissingRewardFunctionError: online-RL config without a reward function.
            JobReferenceConflictError: config/dataset deleted mid-submission.
        """
        owner_id = self._principal.user_id
        if owner_id is None:
            raise CallerNotProvisionedError()
        # Create is always own-scoped: an admin submits jobs against its own
        # configuration and dataset, never another owner's.
        owner_filter = owner_id

        configuration = await self._configuration_repository.get(
            data.config_id, owner_id=owner_filter, include_shared=True
        )
        if configuration is None:
            raise ConfigurationNotFoundError(data.config_id)

        dataset = await self._dataset_repository.get(
            data.dataset_id, owner_id=owner_filter, include_shared=True
        )
        if dataset is None:
            raise DatasetNotFoundError(data.dataset_id)
        if dataset.status != DatasetStatus.READY:
            raise DatasetNotReadyForJobError(data.dataset_id, dataset.status)

        rl_type = (configuration.rl_tuner_type or "").lower()
        reward_code = (data.reward_function_code or "").strip()
        if rl_type in ONLINE_RL_TUNER_TYPES and not reward_code:
            raise MissingRewardFunctionError(configuration.rl_tuner_type or rl_type)
        reward_name = data.reward_function_name or ("compute_score" if reward_code else None)

        config_snapshot: dict[str, Any] = {
            "name": configuration.name,
            "tuner_type": configuration.tuner_type,
            "rl_tuner_type": configuration.rl_tuner_type,
            "config_data": configuration.config_data,
        }

        job = await self._repository.create(
            user_id=str(owner_id),
            config_id=data.config_id,
            dataset_id=data.dataset_id,
            model=data.model,
            model_source=data.model_source,
            experiment_name=data.experiment_name,
            tuning_type=configuration.tuner_type,
            seed=data.seed,
            autotune=data.autotune,
            config_snapshot=config_snapshot,
            reward_function_code=data.reward_function_code,
            reward_function_name=reward_name,
        )

        await self._runner.submit(job.id)

        created = await self._repository.get(job.id, owner_id=owner_filter)
        if created is None:  # pragma: no cover - just-created row is always visible
            raise JobNotFoundError(job.id)
        return job_to_read(created)

    async def cancel(self, job_id: UUID, *, scope: DataScope = DataScope.OWN) -> JobRead:
        """Cancel a live job and drive it to ``terminated``, scoped to the caller.

        Idempotent for an already-``terminated`` job (returns it unchanged). A
        ``completed`` or ``error`` job has no work to stop (409). Otherwise the
        backend is told to stop (``runner.cancel``) and the job + its ``TUNING``
        task are written to ``terminated``. Scoped exactly like :meth:`get`.

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it belongs to someone else.
            JobNotCancellableError: the job is already ``completed`` or ``error``.
            BuildCancelUpstreamError: the cluster refused the cancel (llmb).
            JobCancellationInProgressError: a local run did not stop in time.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise JobNotFoundError(job_id)
        job = await self._repository.get(job_id, owner_id=owner_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.status == RunStatus.TERMINATED:
            return job_to_read(job)  # idempotent — already cancelled
        if job.status in TERMINAL_JOB_STATUSES:
            raise JobNotCancellableError(job_id, job.status)

        await self._runner.cancel(job_id)
        check_transition(job.status, RunStatus.TERMINATED)
        await self._repository.set_status(job_id, RunStatus.TERMINATED)
        await self._repository.upsert_task(job_id, GbTaskType.TUNING, status=RunStatus.TERMINATED)

        refreshed = await self._repository.get(job_id, owner_id=owner_id)
        if refreshed is None:  # pragma: no cover - just written
            raise JobNotFoundError(job_id)
        return job_to_read(refreshed)

    async def delete(self, job_id: UUID, *, scope: DataScope = DataScope.OWN) -> None:
        """Delete a job owned by the calling principal, cancelling live work first.

        Scoped exactly like :meth:`get`. A non-terminal job's backend work is
        stopped via ``runner.cancel`` before removal — for llmb the build is
        cancelled (the cluster winds down independently of our row); for local the
        in-process run is stopped and awaited so the cascade cannot race the
        trainer's sink. ``runner.cancel`` is a no-op when there is nothing to stop,
        so a terminal or pending-no-build job deletes directly. The delete cascades
        to the job's trials, results and build tasks (see ``JobRepository.delete``).

        Raises:
            ScopeNotPermittedError: a non-admin requested ``scope=all``.
            JobNotFoundError: no such job, or it belongs to someone else.
            BuildCancelUpstreamError: the cluster refused the cancel (llmb).
            JobCancellationInProgressError: a local run did not stop in time.
        """
        owner_id = resolve_owner_filter(self._principal, scope)
        if sees_nothing(self._principal, scope):
            raise JobNotFoundError(job_id)
        job = await self._repository.get(job_id, owner_id=owner_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.status not in TERMINAL_JOB_STATUSES:
            await self._runner.cancel(job_id)
        deleted = await self._repository.delete(job_id, owner_id=owner_id)
        if not deleted:  # deleted in a race between the get and the delete
            raise JobNotFoundError(job_id)


def check_transition(current: RunStatus, requested: RunStatus) -> None:
    """Assert that a job may move from ``current`` to ``requested``.

    The state machine is :data:`autotunex.models.job.ALLOWED_JOB_TRANSITIONS`.
    Terminal states accept no transitions.

    Raises:
        InvalidStateTransitionError: the transition is not allowed.
    """
    if requested not in ALLOWED_JOB_TRANSITIONS[current]:
        raise InvalidStateTransitionError(current, requested)
