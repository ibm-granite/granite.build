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
"""Tear down a multi-node Ray cluster previously started by ``ray_up_blaunch``."""

from __future__ import annotations

import errno
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from typing import Any, Dict, List

import ray

from autotune.cluster import stop_ray_cluster
from autotune.lsf.log_utils import banner, phase_timer

logger = logging.getLogger(__name__)


def _ray_shutdown_with_timeout(timeout_s: float = 30.0) -> None:
    """Run ``ray.shutdown()`` in a thread with a hard timeout.

    Ray's gRPC client can wedge during shutdown when GCS is dying or already
    dead.  We don't want that to keep the LSF driver job alive — give it a
    bounded window and move on.
    """
    done = threading.Event()
    err: List[BaseException] = []

    def _target() -> None:
        try:
            ray.shutdown()
        except BaseException as e:  # noqa: BLE001 — best-effort
            err.append(e)
        finally:
            done.set()

    t = threading.Thread(target=_target, name="ray-shutdown", daemon=True)
    t.start()
    if not done.wait(timeout=timeout_s):
        logger.warning(
            f"[ray_down] ray.shutdown did not return within {timeout_s}s; abandoning it. "
            "The thread is daemonized and will not block process exit."
        )
        return
    if err:
        logger.warning(f"[ray_down] ray.shutdown raised: {err[0]}")


def _kill_blaunch_pids(worker_pids: List[int], timeout_s: float = 10.0) -> None:
    """Tear down the local ``blaunch`` (and head-host worker) process trees.

    Each PID in ``worker_pids`` was started via ``_popen_detached``, which
    sets ``start_new_session=True`` — so each PID is the leader of its own
    process group / session. We signal the **whole group** so that any
    helper procs blaunch / bash spawned die too.

    Why this is needed: when ``python main.py`` exits cleanly, ``Popen``
    objects are GC'd but their children are NOT signaled — Python does not
    kill subprocesses on interpreter shutdown. The blaunch process keeps
    talking to RES on the remote host (where ``worker_entry`` is parked
    on ``signal.pause()``), and the local-worker bash keeps wait()'ing on
    its python child. Both also still hold their stdout/stderr fds, which
    keeps the LSF batch-job log writer open and prevents LSF from reaping
    the job. Sending SIGTERM here breaks the deadlock; SIGKILL is the
    backstop because some blaunch builds ignore SIGTERM.

    Sequence per worker: SIGTERM → poll up to ``timeout_s`` for exit →
    SIGKILL anything still alive → final waitpid to reap zombies.
    """
    if not worker_pids:
        return
    alive = []
    for pid in worker_pids:
        try:
            os.killpg(pid, signal.SIGTERM)
            alive.append(pid)
            logger.info(f"[ray_down] sent SIGTERM to blaunch/worker pgid={pid}")
        except OSError as e:
            if e.errno == errno.ESRCH:
                logger.info(f"[ray_down] pgid={pid} already gone")
            else:
                logger.warning(f"[ray_down] SIGTERM pgid={pid} failed: {e}")

    # Wait briefly for the children to exit; SIGKILL anything that lingers.
    deadline = time.monotonic() + timeout_s
    while alive and time.monotonic() < deadline:
        still_alive = []
        for pid in alive:
            try:
                wpid, _ = os.waitpid(pid, os.WNOHANG)
                if wpid == 0:
                    still_alive.append(pid)
            except OSError as e:
                if e.errno == errno.ECHILD:
                    pass  # already reaped
                else:
                    logger.warning(f"[ray_down] waitpid({pid}) raised: {e}")
        alive = still_alive
        if alive:
            time.sleep(0.5)

    if alive:
        logger.warning(f"[ray_down] {len(alive)} blaunch/worker pgid(s) survived SIGTERM; sending SIGKILL: {alive}")
        for pid in alive:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError as e:
                if e.errno != errno.ESRCH:
                    logger.warning(f"[ray_down] SIGKILL pgid={pid} failed: {e}")
        # One final reap so we don't leave zombies for the (about-to-exit) parent.
        for pid in alive:
            try:
                os.waitpid(pid, os.WNOHANG)
            except OSError:
                pass


def _remote_ray_stop_via_blaunch(remote_hosts: List[str], conda_env: str, timeout_s: float = 20.0) -> None:
    """Run ``ray stop`` on each remote worker host via a one-shot ``blaunch``.

    Why: ``_kill_blaunch_pids`` SIGTERMs the **local** blaunch process (the
    one we Popen'd from the driver). On many LSF builds, blaunch does NOT
    propagate that signal to RES, so the remote ``worker_entry`` keeps
    sitting on ``signal.pause()`` and the remote ``ray`` daemons keep
    running. Those processes live in the LSF job's cgroup, which spans
    every allocated host — and LSF refuses to mark the job ``DONE`` until
    that cgroup is empty.

    Telling each remote host to ``ray stop`` synchronously, via a fresh
    ``blaunch`` invocation, is the most reliable way to drain the cgroup
    on the remote side without relying on signal forwarding. ``ray stop``
    on the remote host kills all ray daemons; the still-paused
    ``worker_entry`` process is then orphaned and will be cleaned up when
    the LSF job is reaped — *or* killed by the subsequent
    ``_kill_blaunch_pids`` SIGTERM/SIGKILL.

    Best-effort: per-host failures (timeout, blaunch error) are logged and
    skipped. The next phase (kill_blaunch_pids) will catch anything left.
    """
    if not remote_hosts:
        return
    inner = f"source ~/.bashrc && conda activate {shlex.quote(conda_env)} && ray stop --force"
    for host in remote_hosts:
        cmd = ["blaunch", "-z", host, "bash", "-lc", inner]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            logger.info(
                f"[ray_down] remote ray stop on {host!r} rc={proc.returncode} "
                f"stdout={proc.stdout.strip()!r} stderr={proc.stderr.strip()!r}"
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"[ray_down] remote ray stop on {host!r} timed out after {timeout_s}s")
        except Exception as e:
            logger.warning(f"[ray_down] remote ray stop on {host!r} raised: {e}")


def _self_bkill() -> None:
    """Last-resort: ``bkill $LSB_JOBID`` to force LSF to reap our own job.

    Why this exists: even after we ``ray stop`` everywhere we can reach,
    SIGKILL the local blaunch session groups, and ``stop_ray_cluster``
    on the head host, the LSF job's cgroup on the **remote** host can
    still contain processes we don't have a clean way to reach (e.g. a
    ``worker_entry`` whose blaunch we SIGKILL'd before RES propagated
    the kill, or a hung ``ray stop`` that ate the channel). LSF refuses
    to mark the job ``DONE`` until the cgroup is empty on every host.

    ``bkill $LSB_JOBID`` from inside the job is a documented LSF idiom
    for forcing teardown: the master batch daemon SIGKILLs every process
    in the cgroup on every allocated host. Cosmetic downside — the job
    exits ``EXIT`` instead of ``DONE`` — is acceptable here because all
    useful work has already completed by the time we reach this step
    (models / logs / outputs are on disk).
    """
    jobid = os.environ.get("LSB_JOBID")
    if not jobid:
        logger.info("[ray_down] LSB_JOBID not set; skipping self-bkill")
        return
    try:
        proc = subprocess.run(
            ["bkill", jobid],
            capture_output=True,
            text=True,
            timeout=30,
        )
        logger.info(
            f"[ray_down] self-bkill {jobid} rc={proc.returncode} "
            f"stdout={proc.stdout.strip()!r} stderr={proc.stderr.strip()!r}"
        )
    except Exception as e:
        logger.warning(f"[ray_down] self-bkill {jobid} raised: {e}")


def stop_multinode_ray_cluster(cluster_info: Dict[str, Any]) -> None:
    """Best-effort teardown of a blaunch cluster.  Safe to call from a ``finally`` block.

    Sequence: drain the remote-host cgroup via a remote ``ray stop`` →
    ``ray.shutdown`` → SIGTERM/SIGKILL the local blaunch + colocated-worker
    process groups → ``stop_ray_cluster`` on the head → ``bkill $LSB_JOBID``.

    There are no separate worker ``bsub`` jobs to kill; LSF's RES tears down
    blaunch'd children when the outer driver job exits, and the closing
    self-bkill forces LSF to reap the job's cgroup on every allocated host.
    """
    if not cluster_info:
        return

    banner("ray_down: tearing down cluster")
    worker_pids = list(cluster_info.get("worker_pids") or [])
    worker_hosts = list(cluster_info.get("worker_hosts") or [])
    job_group = cluster_info.get("job_group") or ""
    temp_dir = cluster_info.get("temp_dir")

    logger.info(
        f"[ray_down] head={cluster_info.get('head_address')!r} "
        f"workers={worker_hosts} job_group={job_group!r} temp_dir={temp_dir!r}"
    )

    # Drain the remote-host cgroup BEFORE we shut down Ray and SIGTERM the
    # local blaunch processes.  If we kill the local blaunch first, we lose
    # the channel to RES on the remote host(s) — the remote `worker_entry`
    # keeps sitting on signal.pause(), and the remote ray daemons keep
    # running.  Both keep the LSF job's cgroup non-empty, which prevents LSF
    # from reaping the job.
    remote_hosts = list(cluster_info.get("remote_worker_hosts") or [])
    conda_env = cluster_info.get("conda_env") or ""
    if remote_hosts and conda_env:
        try:
            with phase_timer("remote_ray_stop"):
                _remote_ray_stop_via_blaunch(remote_hosts, conda_env)
        except Exception as e:
            logger.warning(f"[ray_down] _remote_ray_stop_via_blaunch raised: {e}")
    elif remote_hosts and not conda_env:
        logger.warning(
            "[ray_down] handle has remote_worker_hosts but no conda_env; "
            "skipping remote ray stop. The LSF job may not exit cleanly."
        )

    try:
        with phase_timer("ray.shutdown"):
            _ray_shutdown_with_timeout(timeout_s=30.0)
    except Exception as e:
        logger.warning(f"[ray_down] ray.shutdown phase raised: {e}")

    # The blaunch / local-bash children are direct children of this Python
    # process. We MUST signal them ourselves before returning — otherwise
    # the outer ``bash -lc`` shell waits on them forever and LSF never
    # reaps the job (the user has to ``bkill`` manually). SIGTERM
    # propagates through blaunch → RES → remote ``worker_entry``, which
    # catches it and runs ``ray stop`` cleanly.
    try:
        with phase_timer("kill_blaunch_pids"):
            _kill_blaunch_pids(worker_pids)
    except Exception as e:
        logger.warning(f"[ray_down] _kill_blaunch_pids raised: {e}")

    if temp_dir:
        try:
            with phase_timer("stop_ray_cluster"):
                stop_ray_cluster(temp_dir)
        except Exception as e:
            logger.warning(f"[ray_down] stop_ray_cluster({temp_dir}) raised: {e}")

    # Final step: self-bkill so LSF tears down the cgroup on every allocated
    # host.  Without this the job stays in RUN state for an indeterminate time
    # waiting for orphaned processes on remote hosts that we don't have a
    # reliable way to reach.  Done last so cleaner shutdown paths (remote ray
    # stop, killpg, etc.) get a chance first; the bkill is the backstop.
    # Trade-off: the job exits ``EXIT`` instead of ``DONE``, which is cosmetic
    # — by the time we reach this step all useful work (models, logs, results)
    # is already on disk.
    try:
        with phase_timer("self_bkill"):
            _self_bkill()
    except Exception as e:
        logger.warning(f"[ray_down] _self_bkill raised: {e}")

    banner("ray_down: complete")
