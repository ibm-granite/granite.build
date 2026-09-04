# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Seams and value objects for the in-process ``local`` job runner.

This is the ``local`` analogue of ``services/launch/protocols.py``: it declares
the frozen context assembled from a job's persisted data (:class:`LocalRunContext`),
the two Protocol seams the runner depends on (:class:`LocalTrainer` and
:class:`TrialSink`), the log value object the sink transports
(:class:`LogRecord`), and the reward-function injection helper
(:func:`inject_reward_function`).

The seams are ``typing.Protocol`` rather than ABCs so a test double needs no base
class while mypy still checks it structurally — the same dependency-inversion
shape used across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID


@dataclass(frozen=True)
class LocalRunContext:
    """Everything :class:`LocalTrainer` needs, read from the job and its snapshot.

    The local analogue of ``LaunchContext``. Assembled from persisted data only
    (``config_snapshot``, not the live configuration) so a run reflects what the
    job recorded at submit time.

    ``tuning_algo`` / ``rl_algo`` come from ``job.tuning_type`` and the snapshot's
    ``rl_tuner_type`` (``"none"`` — the value the pipeline treats as "unused" —
    when absent), matching how the granite.build path maps ``--tuning_algo`` /
    ``--rl_algo``. ``train_file`` / ``validation_file`` follow this repo's
    local-storage layout (``dataset_storage_dir / <dataset_id> / <file>``), and
    ``output_dir`` is ``local_output_dir / <job_id>``.
    """

    job_id: UUID
    model: str
    model_source: str
    experiment_name: str
    tuning_algo: str
    rl_algo: str
    config_name: str
    config_data: dict[str, Any]
    train_file: Path
    validation_file: Path
    output_dir: Path
    seed: int | None
    autotune: bool
    cleanup: bool
    save_history: bool
    reward_function_code: str | None
    reward_function_name: str | None


@dataclass(frozen=True)
class LogRecord:
    """A single log line captured during a local run, bound for ``log_entries``.

    Mirrors that table's columns: every field is optional because a captured log
    line may carry a trial context, an iteration, or an epoch — or none of them.
    """

    trial_id: str | None
    level: str | None
    filename: str | None
    message: str | None
    iteration: int | None
    epoch: float | None


@dataclass(frozen=True)
class TrainingMetricRecord:
    """One per-step metrics row captured during a local run."""

    trial_id: str | None
    global_step: int
    epoch: float | None
    loss: float | None
    grad_norm: float | None
    learning_rate: float | None
    split: str
    extra: dict[str, Any] | None


class TrialSink(Protocol):
    """Where a trainer reports trial lifecycle events during a local run.

    The methods are synchronous because they are called from Ray-callback worker
    threads. The persisting implementation (``DbTrialSink``) bridges each call to
    the async repositories on the event loop; a fake satisfying this Protocol is
    used to drive the runner in tests without Ray.
    """

    def trial_started(self, trial_id: str, config: dict[str, Any] | None) -> None:
        """Record that ``trial_id`` began, with its concrete parameter ``config``."""
        ...

    def trial_result(self, trial_id: str, metric: str, metrics: dict[str, Any] | None) -> None:
        """Record a result for ``trial_id`` under the objective ``metric``."""
        ...

    def trial_completed(self, trial_id: str) -> None:
        """Record that ``trial_id`` finished successfully."""
        ...

    def trial_error(self, trial_id: str) -> None:
        """Record that ``trial_id`` failed."""
        ...

    def log(self, record: LogRecord) -> None:
        """Persist a captured log line."""
        ...

    def training_metric(self, record: TrainingMetricRecord) -> None:
        """Persist one per-step training-metrics row."""
        ...


class LocalTrainer(Protocol):
    """Runs the HPO pipeline in-process, reporting progress to a ``TrialSink``.

    The only implementation, ``AutotuneLocalTrainer``, imports Ray and ``autotune``
    lazily; CI drives ``LocalJobRunner`` against a fake trainer instead.
    """

    def run(self, ctx: LocalRunContext, sink: TrialSink) -> None:
        """Run the pipeline described by ``ctx``, forwarding events to ``sink``."""
        ...


def inject_reward_function(
    config_data: dict[str, Any],
    *,
    output_dir: Path,
    code: str,
    name: str | None,
) -> None:
    """Write an online-RL reward function to disk and point the config at it.

    Writes ``code`` to ``<output_dir>/reward_function.py`` (creating ``output_dir``
    if needed), then rewrites the snapshotted config in place:
    ``config_data["training_rl_config"]["reward_function_path"]["default"]`` becomes
    the written path, and — when ``name`` is given —
    ``["reward_function_name"]["default"]`` becomes ``name``. A faithful port of the
    2025 local runner.

    The config rewrites are guarded: each ``["default"]`` is set only when the
    nested ``training_rl_config`` mapping and the target key already exist, so a
    config that does not declare a reward function is left structurally unchanged
    (the file is still written).

    Args:
        config_data: The snapshotted configuration, mutated in place.
        output_dir: Directory the reward-function file is written into.
        code: The reward-function source to persist.
        name: The reward function's name, or ``None`` to leave the name untouched.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_path = output_dir / "reward_function.py"
    reward_path.write_text(code)

    rl_config = config_data.get("training_rl_config")
    if not isinstance(rl_config, dict):
        return

    path_entry = rl_config.get("reward_function_path")
    if isinstance(path_entry, dict):
        path_entry["default"] = str(reward_path)

    if name is not None:
        name_entry = rl_config.get("reward_function_name")
        if isinstance(name_entry, dict):
            name_entry["default"] = name
