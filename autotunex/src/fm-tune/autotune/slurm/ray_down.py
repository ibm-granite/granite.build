# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tear down a multi-node Ray cluster previously started by ``ray_up_srun``."""

from __future__ import annotations

import logging
from typing import Any, Dict

from autotune.cluster import stop_ray_cluster
from autotune.lsf.log_utils import banner, phase_timer
from autotune.lsf.ray_down import _kill_blaunch_pids, _ray_shutdown_with_timeout

logger = logging.getLogger(__name__)

# srun needs a moment to forward SIGTERM to the remote task, wait for
# worker_entry's ``ray stop``, and collect the step's exit; 10s (the LSF value)
# cuts that short on a busy node and SIGKILLs srun mid-forward.
SRUN_TERM_GRACE_S = 30.0


def stop_multinode_ray_cluster_srun(cluster_info: Dict[str, Any]) -> None:
    """Best-effort teardown of an srun cluster. Safe to call from a ``finally`` block.

    Sequence: ``ray.shutdown`` → SIGTERM the worker process groups (the local
    worker and one ``srun`` per remote node; srun forwards the signal to
    ``worker_entry``, which runs ``ray stop``) → SIGKILL survivors →
    ``stop_ray_cluster`` on the head.

    Nothing reaches into remote nodes beyond that: ``proctrack/cgroup`` kills
    whatever is left in every step when the batch step exits, which is also why
    there is no counterpart to the LSF path's self-``bkill``.
    """
    if not cluster_info:
        return

    banner("ray_down_srun: tearing down cluster")
    worker_pids = list(cluster_info.get("worker_pids") or [])
    temp_dir = cluster_info.get("temp_dir")
    logger.info(
        f"[ray_down_srun] head={cluster_info.get('head_address')!r} "
        f"workers={cluster_info.get('worker_hosts')} pids={worker_pids} temp_dir={temp_dir!r}"
    )

    try:
        with phase_timer("ray.shutdown"):
            _ray_shutdown_with_timeout(timeout_s=30.0)
    except Exception as e:
        logger.warning(f"[ray_down_srun] ray.shutdown phase raised: {e}")

    try:
        with phase_timer("kill_worker_pids"):
            _kill_blaunch_pids(worker_pids, timeout_s=SRUN_TERM_GRACE_S)
    except Exception as e:
        logger.warning(f"[ray_down_srun] killing worker pids raised: {e}")

    if temp_dir:
        try:
            with phase_timer("stop_ray_cluster"):
                stop_ray_cluster(temp_dir)
        except Exception as e:
            logger.warning(f"[ray_down_srun] stop_ray_cluster({temp_dir}) raised: {e}")

    banner("ray_down_srun: complete")
