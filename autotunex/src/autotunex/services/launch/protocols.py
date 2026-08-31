# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The TuningLauncher seam.

A launcher turns a job's persisted data into a cluster submission and returns
the handle the cluster gives back. It is the "submit a whole job" seam, distinct
from the per-trial ``TrainingBackend`` in ``services/protocols.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID


@dataclass(frozen=True)
class LaunchContext:
    """Everything a launcher needs, read from the job and its snapshot.

    Assembled from persisted data only (``config_snapshot``, not the live
    configuration) so a launch reflects what the job recorded at submit time.
    ``config_name`` is the snapshotted configuration's display name, which names
    the in-container config file (``/tmp/<config_name>.yaml``); ``data_format``
    is the dataset's file extension (e.g. ``jsonl``), used to build the
    ``--train_file`` / ``--validation_file`` paths.
    """

    job_id: UUID
    model: str
    model_source: str
    experiment_name: str
    tuning_type: str | None
    rl_tuner_type: str | None
    config_name: str
    config_data: dict[str, Any]
    dataset_name: str
    dataset_uri: str | None
    data_format: str
    autotune: bool
    seed: int | None
    reward_function_code: str | None
    reward_function_name: str | None


@dataclass(frozen=True)
class LaunchHandle:
    """The cluster's identifiers for an accepted submission.

    ``build_id`` is what the reconcile follow-on will poll; ``pr_url`` is the
    build's human-facing link, if the CLI emits one. Either may be ``None`` when
    the CLI output does not carry it.
    """

    build_id: UUID | None
    pr_url: str | None


class TuningLauncher(Protocol):
    """Submits a job to a cluster and returns its handle. Must not block long."""

    async def launch(self, ctx: LaunchContext) -> LaunchHandle:
        """Submit the job described by ``ctx``; return the cluster's handle."""
        ...


class BuildCanceller(Protocol):
    """Cancels a submitted granite.build build. The cancel counterpart to launch."""

    async def cancel(self, build_id: UUID) -> None:
        """Request cancellation of ``build_id``. Must not block long."""
        ...
