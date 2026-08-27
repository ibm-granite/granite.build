# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Conversion between ORM tables and API schemas.

Kept in one place so that a change to the database layout has exactly one blast
radius on the API contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from autotunex.db.tables import (
    ConfigurationTable,
    DatasetTable,
    GbTaskTable,
    JobTable,
    LogEntryTable,
    TrialTable,
    UserTable,
)
from autotunex.models.configuration import ConfigurationJobRef, ConfigurationRead
from autotunex.models.dataset import DatasetJobRef, DatasetPreview, DatasetRead
from autotunex.models.job import JobRead, JobSummary
from autotunex.models.log import LogEntryRead
from autotunex.models.task import GbTaskRead
from autotunex.models.trial import TrialRead
from autotunex.models.user import UserRead


def configuration_to_read(
    configuration: ConfigurationTable, associated_jobs: list[ConfigurationJobRef]
) -> ConfigurationRead:
    """Convert a configuration row to its API representation.

    ``associated_jobs`` is passed in (already scoped to the caller by the
    service) rather than traversed here, so this mapper stays free of ownership
    policy — identical to :func:`dataset_to_read`.
    """
    return ConfigurationRead(
        id=configuration.id,
        user_id=configuration.user_id,
        name=configuration.name,
        tuner_type=configuration.tuner_type,
        rl_tuner_type=configuration.rl_tuner_type,
        config_data=configuration.config_data,
        associated_jobs=associated_jobs,
        created_at=configuration.created_at,
        updated_at=configuration.updated_at,
    )


def dataset_to_read(
    dataset: DatasetTable,
    associated_jobs: list[DatasetJobRef],
    preview: DatasetPreview | None = None,
) -> DatasetRead:
    """Convert a dataset row to its API representation.

    ``associated_jobs`` is passed in (already scoped to the caller by the
    service) rather than traversed here, so this mapper stays free of ownership
    policy. ``artifact_id`` is stringified — the ORM stores it as a ``Uuid36``
    but the contract does not assume the identifier is a UUID. Built
    field-by-field, like :func:`configuration_to_read`, so a schema/column
    divergence is a visible edit here.
    """
    return DatasetRead(
        id=dataset.id,
        user_id=dataset.user_id,
        name=dataset.name,
        description=dataset.description,
        data_format=dataset.data_format,
        status=dataset.status,
        status_detail=dataset.status_detail,
        train_file=dataset.train_file,
        train_records=dataset.train_records,
        train_file_size=dataset.train_file_size,
        validation_file=dataset.validation_file,
        validation_records=dataset.validation_records,
        validation_file_size=dataset.validation_file_size,
        artifact_id=str(dataset.artifact_id) if dataset.artifact_id is not None else None,
        artifact_url=dataset.artifact_url,
        associated_jobs=associated_jobs,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        preview=preview,
    )


def resolve_config_name(job: JobTable) -> str:
    """Return the configuration name this job actually ran with.

    ``config_snapshot`` captures the configuration as it was at submission, so it
    takes precedence over the live ``configurations`` row — a job keeps reporting
    what it ran with even after that configuration is renamed. This mirrors the
    ``COALESCE(JSON_UNQUOTE(JSON_EXTRACT(...)), c.name)`` in ``autotunex_jobs``.
    """
    snapshot = job.config_snapshot or {}
    name = snapshot.get("name")
    if isinstance(name, str):
        return name
    return job.configuration.name


def resolve_rl_tuner_type(job: JobTable) -> str | None:
    """Return the RL tuner type, preferring the snapshot over the live row.

    Same precedence rule as :func:`resolve_config_name`.
    """
    snapshot = job.config_snapshot or {}
    tuner = snapshot.get("rl_tuner_type")
    if isinstance(tuner, str):
        return tuner
    return job.configuration.rl_tuner_type


def gb_task_to_read(task: GbTaskTable) -> GbTaskRead:
    """Convert a build task row to its API representation."""
    return GbTaskRead(
        task_id=task.id,
        build_id=task.build_id,
        task_status=task.status,
        task_type=task.type,
        github_pr_url=task.pr_url,
        artifact_id=task.artifact_id,
        artifact_uri=task.artifact_uri,
        build_status=task.build_status,
        task_started_at=task.started_at,
        task_updated_at=task.updated_at,
        rits_url=task.rits_url,
    )


def latest_task_update(tasks: Sequence[GbTaskTable]) -> str | None:
    """Return the latest ``gb_tasks.updated_at`` across a job's tasks, or ``None``.

    This is the job's end for the "Total time" column: the last build task to
    report an update. gb_task timestamps are free-text ``VARCHAR(255)`` gbserver
    strings (schema-review A5) kept unparsed, so the lexicographic ``max`` stands in
    for the chronological one — exact for the zero-padded ISO-8601 form gbserver
    emits. Tasks without an ``updated_at`` are ignored; ``None`` when none has one.
    """
    updates = [task.updated_at for task in tasks if task.updated_at]
    return max(updates) if updates else None


def job_to_summary(job: JobTable, finished_at: str | None = None) -> JobSummary:
    """Convert a job row to the lean list representation (``GET /jobs``).

    ``finished_at`` (the latest ``gb_tasks.updated_at``) is supplied by the list
    query as a computed column, because the lean list does not load ``tasks``.
    :func:`job_to_read`, which does load them, derives it via
    :func:`latest_task_update` instead.
    """
    return JobSummary(
        id=job.id,
        user_id=job.user_id,
        status=job.status,
        seed=job.seed,
        config_id=job.config_id,
        config_name=resolve_config_name(job),
        dataset_id=job.dataset_id,
        dataset=job.dataset.name,
        model=job.model,
        experiment_name=job.experiment_name,
        user=job.user.email,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=finished_at,
    )


def trial_to_read(trial: TrialTable) -> TrialRead:
    """Convert a trial row to its API representation.

    ``metric`` and ``metrics`` come from the one-to-one ``results`` row; a trial
    that has not reported yet gets an empty mapping rather than nulls.
    """
    result = trial.result
    return TrialRead(
        id=trial.id,
        job_id=trial.job_id,
        status=trial.status,
        config=trial.config,
        metric=result.metric if result is not None else None,
        metrics=(result.metrics or {}) if result is not None else {},
        created_at=trial.created_at,
        updated_at=trial.updated_at,
    )


def _config_is_stale(job: JobTable) -> bool:
    """Return True when the live configuration has drifted from the job's snapshot.

    Content comparison, not a timestamp: fires only when a *behavioural* field the
    snapshot froze — ``config_data``, ``tuner_type`` or ``rl_tuner_type`` — differs
    from the live configuration. A cosmetic rename or a no-op re-save therefore does
    not flag historical jobs, and an edit-then-revert correctly reads not-stale. This
    deliberately supersedes the pre-rewrite ``updated_at > created_at`` proxy, which
    over-reported because ``configurations.updated_at`` bumps on any column write.

    The ``config_data`` comparison is a recursive, key-order-independent ``dict``
    compare, so MySQL's JSON key reordering cannot cause a false diff (compare parsed
    JSON, never raw strings). A job with no ``config_snapshot`` — some pipeline-written
    rows — has no baseline, so it is reported not-stale rather than guessed.
    ``job.configuration`` is always eager-loaded on the detail path (``_view_shaped``),
    so this triggers no async lazy load.
    """
    snapshot = job.config_snapshot
    if not snapshot:
        return False
    live = job.configuration
    return (
        snapshot.get("config_data") != live.config_data
        or snapshot.get("tuner_type") != live.tuner_type
        or snapshot.get("rl_tuner_type") != live.rl_tuner_type
    )


def job_to_read(job: JobTable) -> JobRead:
    """Convert a job row to the full detail representation.

    Reuses :func:`job_to_summary`'s already-validated lean fields via attribute
    access, then adds the detail-only fields directly from ``job``.
    """
    summary = job_to_summary(job, finished_at=latest_task_update(job.tasks))
    return JobRead(
        id=summary.id,
        user_id=summary.user_id,
        status=summary.status,
        seed=summary.seed,
        config_id=summary.config_id,
        config_name=summary.config_name,
        dataset_id=summary.dataset_id,
        dataset=summary.dataset,
        model=summary.model,
        experiment_name=summary.experiment_name,
        user=summary.user,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        finished_at=summary.finished_at,
        model_source=job.model_source,
        tuning_type=job.tuning_type,
        rl_tuner_type=resolve_rl_tuner_type(job),
        ray_address=job.ray_address,
        cleanup=job.cleanup,
        autotune=job.autotune,
        num_trials=job.num_trials,
        tasks=[gb_task_to_read(task) for task in job.tasks],
        config_snapshot=job.config_snapshot,
        output_artifacts=job.output_artifacts,
        trials=[trial_to_read(trial) for trial in job.trials],
        is_stale=_config_is_stale(job),
    )


def log_entry_to_read(entry: LogEntryTable) -> LogEntryRead:
    """Convert a ``log_entries`` row to its API shape (field-by-field, per house style)."""
    return LogEntryRead(
        id=entry.id,
        level=entry.level,
        filename=entry.filename,
        message=entry.message,
        iteration=entry.iteration,
        epoch=entry.epoch,
        timestamp=entry.timestamp,
    )


def user_to_read(user: UserTable) -> UserRead:
    """Convert a user row to its API representation.

    Built field-by-field to match this module's style and keep the ORM-to-schema
    contract an explicit edit here rather than a silent ``model_validate`` match.
    """
    return UserRead(
        id=user.id,
        email=user.email,
        role=user.role,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
