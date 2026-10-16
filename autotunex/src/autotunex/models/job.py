# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Job schemas.

A *job* is one automated tuning run, searching a hyperparameter space by
running trials. Jobs may be submitted through this API via ``POST /jobs`` (see
:class:`JobCreate`) or written directly by the tuning pipeline; ``GET /jobs``
and ``GET /jobs/{id}`` report what exists.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from autotunex.models.status import RunStatus
from autotunex.models.task import GbTaskRead

ALLOWED_JOB_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset(
        {RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.TERMINATED, RunStatus.ERROR}
    ),
    RunStatus.RUNNING: frozenset(
        {RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.TERMINATED}
    ),
    RunStatus.PAUSED: frozenset({RunStatus.RUNNING, RunStatus.TERMINATED, RunStatus.ERROR}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.ERROR: frozenset(),
    RunStatus.TERMINATED: frozenset(),
}
"""The job state machine, keyed by current status.

Total over :class:`RunStatus`, so a lookup can never ``KeyError``. Terminal
states map to the empty set rather than being absent, which keeps the
"unreachable" property explicit rather than implied by omission.

``PENDING`` -> ``COMPLETED`` is legal, not just a `RUNNING`-only path: the
job-status reconcile loop polls gbserver, not the cluster's event stream, so it
can observe a build that already reached ``success`` while our copy of the job
never caught the intervening ``running`` transit (e.g. the API process was down
for the whole running phase and restarted after the build finished). That build
genuinely ran to completion; refusing the transition would leave the job a
permanent zombie that re-polls forever instead of recording what happened.

Enforced today by the cancel path (``services/jobs.py``), the reconcile sweep
(``services/reconcile/loop.py``), and both in-process runners
(``services/runner.py``, ``services/local/runner.py``); the task queue will add
more callers.
"""

TERMINAL_JOB_STATUSES: frozenset[RunStatus] = frozenset(
    {RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.TERMINATED}
)
"""Job states with no outgoing transitions — a job here has no work left to stop.

The same three states that map to the empty set in :data:`ALLOWED_JOB_TRANSITIONS`,
named as a set so cancel/delete can test membership without restating them.
"""

ONLINE_RL_TUNER_TYPES: frozenset[str] = frozenset({"ppo", "grpo", "dapo"})
"""``rl_tuner_type`` values that require a reward function.

Online-RL tuners score generated rollouts with a reward function; offline-RL
(``dpo``/``kto``) and SFT do not. Compared case-insensitively at submit time.
"""

_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class JobCreate(BaseModel):
    """Request body for ``POST /jobs``.

    Ownership is resolved from the calling principal, so there is no ``user_id``.
    ``tuning_type`` is derived server-side from the referenced configuration and
    is likewise absent. ``model_source`` admits only ``huggingface`` and
    ``custom_path``; other model catalogues are unsupported. ``protected_namespaces=()`` is required
    because ``model``/``model_source`` collide with Pydantic's ``model_`` prefix.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    config_id: UUID
    dataset_id: UUID
    model: _NonEmptyStr
    model_source: Literal["huggingface", "custom_path"] = "huggingface"
    experiment_name: _NonEmptyStr
    autotune: bool = True
    seed: int = 42
    reward_function_code: str | None = None
    reward_function_name: str | None = None


class JobShape(StrEnum):
    """Which response shape a caller wants from ``GET /jobs/{id}``.

    ``FULL`` (the default) is :class:`JobRead`: the job's own record plus the
    nested build ``tasks`` array and the ``config_snapshot`` blob. ``LEAN`` is
    :class:`JobDetail` — the record alone, exactly what
    ``GET /jobs/by-build-id/{build_id}`` returns, so the two lean job reads agree
    rather than diverging by a key.

    An enum rather than a boolean for two reasons. It names a *shape*, so
    ``LEAN`` dropping ``config_snapshot`` as well as ``tasks`` is part of the
    contract instead of a surprise hiding behind a field-named flag. And it leaves
    room for a third value (``summary``, for :class:`JobSummary`) without a second
    parameter that could contradict the first. ``lean`` means :class:`JobDetail`
    exactly; the prose elsewhere calls both child-free shapes "lean", this does not.

    Orthogonal to :class:`~autotunex.models.common.DataScope`: ``shape`` selects
    what a caller sees *of a job it may already read*, never which jobs it may read.
    """

    FULL = "full"
    LEAN = "lean"


class JobSummary(BaseModel):
    """A job in the compact shape ``GET /jobs`` returns — one page row.

    Deliberately lean: the tuning-tab list needs only identity, status, the
    owner/config/dataset labels, the model, and timestamps. Everything heavier —
    the nested build tasks, the RL/tuning-type and runtime flags, the trial
    count, and the two JSON blobs — lives on :class:`JobRead` (``GET /jobs/{id}``),
    so a page of jobs stays small and costs one fewer DB round trip (no ``tasks``
    ``selectinload``). Task and trial detail is fetched per-job from the detail
    endpoint after the list is shown.
    """

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    user_id: str
    status: RunStatus
    seed: int | None = None
    config_id: UUID
    config_name: str
    dataset_id: UUID
    dataset: str = Field(description="Dataset name, from datasets.name.")
    model: str
    experiment_name: str
    user: str = Field(description="Owner's email, from users.email.")
    created_at: datetime
    updated_at: datetime
    finished_at: str | None = Field(
        default=None,
        description=(
            "The run's end — the latest gb_tasks.updated_at across the job's build "
            "tasks — used by the UI to render Total time as (finished_at - "
            "created_at). A free-text string, not a datetime, because gb_tasks "
            "timestamps are VARCHAR(255) (schema-review A5). None when the job has "
            "no build task with an update time (e.g. local-backend jobs), which the "
            "UI renders as '—'. Deliberately NOT jobs.updated_at: that column is "
            "ON UPDATE CURRENT_TIMESTAMP and is bumped by any row write, so it is a "
            "last-modified time, not a completion time."
        ),
    )


class JobDetail(JobSummary):
    """A job's own record — everything on the row, none of its child collections.

    Returned by ``GET /jobs/by-build-id/{build_id}``, and by ``GET /jobs/{id}`` when
    the caller asks for :attr:`JobShape.LEAN`. Adds what
    :class:`JobSummary` drops for leanness: the model source, the tuning/RL type
    and runtime flags, the planned trial budget, the artifact descriptor, and the
    configuration-drift flag.

    Carries no child collection and no snapshot. Trials live behind
    ``GET /jobs/{id}/trials`` (paged), and the build ``tasks`` array plus the
    ``config_snapshot`` blob are added by :class:`JobRead` for
    ``GET /jobs/{id}``. A caller that arrived *by build id* already holds the one
    field the tasks array exists to expose, and the snapshot is the heaviest blob
    on the response — it embeds the whole configuration as it ran.

    :attr:`is_stale` stays here even though it is derived from the snapshot: a
    caller learns its configuration has drifted without being handed the snapshot
    to diff.
    """

    model_source: str
    tuning_type: str | None = None
    rl_tuner_type: str | None = None
    ray_address: str | None = None
    cleanup: bool | None = None
    autotune: bool | None = None
    num_trials: int = Field(
        ge=0,
        description=(
            "How many trials this job's configuration asked the search to evaluate — "
            "the planned budget, read from the job's snapshotted configuration "
            "(config_data.tune_config.num_samples), not a count of trial rows. A "
            "pending job therefore reports its full budget rather than 0. Reports 0 "
            "when the job has no snapshot or the snapshot declares no budget. The "
            "budget is reported even when autotune is false, in which case the "
            "pipeline runs a single default-configuration trial instead of searching "
            "it. For how many trials actually exist, read `total` from "
            "GET /jobs/{id}/trials."
        ),
    )
    output_artifacts: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description=(
            "Free-form artifact descriptor written by the tuning pipeline, outside "
            "this service. Both an object and a bare list of file descriptors occur "
            "in the wild — the publish step records a list — so the union is the real "
            "shape of the column, not laxity. Typing this dict-only made the whole "
            "detail response 500 on any published job. See AssetService._map, which "
            "tolerates the same two shapes when serving the result report."
        ),
    )
    is_stale: bool = Field(
        default=False,
        description=(
            "True when the live configuration's behavioural settings no longer match "
            "what this job snapshotted at submit — config_data, tuner_type, or "
            "rl_tuner_type differs. A cosmetic rename does not set it. Computed at read "
            "time on the detail responses only; never present on the JobSummary list shape."
        ),
    )


class JobRead(JobDetail):
    """A job as returned by ``GET /jobs/{id}`` and ``POST /jobs`` — the full record.

    The default shape of ``GET /jobs/{id}``; a caller passing
    :attr:`JobShape.LEAN` gets :class:`JobDetail` instead.

    Adds the two things :class:`JobDetail` withholds: the nested build ``tasks``
    array, and the ``config_snapshot`` the job captured at submit. Both are wanted
    when a client is rendering a job's own page and neither is wanted by the
    build-id lookup, which is why the split exists.

    Still carries no trial list — see :class:`JobDetail` and
    ``GET /jobs/{id}/trials``.
    """

    tasks: list[GbTaskRead] = Field(
        default_factory=list,
        description=(
            "Build tasks for this job. Nested rather than flattened: the "
            "autotunex_jobs view emitted one row per task."
        ),
    )
    config_snapshot: dict[str, Any] | None = None
