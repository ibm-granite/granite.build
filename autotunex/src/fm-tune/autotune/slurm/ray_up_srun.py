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
"""Stand up a multi-node Ray cluster inside a single Slurm allocation via srun.

The Slurm counterpart of :mod:`autotune.lsf.ray_up_blaunch`, with the same flow:

    1. The user submits ``main.py --scheduler slurm`` via ``sbatch`` requesting
       **N GPU nodes** in one allocation (``-N N --gres=gpu:G --exclusive``).
       The batch step (and so the driver) runs on the first allocated node.
    2. :func:`start_multinode_ray_cluster_srun`:
        - expands ``SLURM_JOB_NODELIST`` into the host list
        - starts ``ray start --head`` on the current node (CPU-only)
        - starts a Ray worker on the head node directly (``Popen``; the batch
          step already owns that node's GPUs and CPUs)
        - starts one Ray worker per remaining node as its own ``srun`` step
        - waits until Ray sees all expected GPUs and worker nodes
    3. ``main.py`` runs HPO / training as usual against the cluster.
    4. :func:`autotune.slurm.ray_down.stop_multinode_ray_cluster_srun` tears
       it down.

Three things differ from LSF and are the reason this is not just a different
launcher string:

  * **No shell, no conda activation.** ``srun`` exports the driver's whole
    environment (``--export=ALL`` is the default), so workers run
    ``sys.executable -m autotune.lsf.worker_entry`` directly. ``--conda_env`` is
    not needed.
  * **A step gets one CPU unless told otherwise.** ``task/cgroup`` confines a
    step to the CPUs it asked for, so without ``--cpus-per-task`` the remote
    raylet and every Ray worker on that node would share one core. The count
    comes from ``SLURM_JOB_CPUS_PER_NODE`` per host.
  * **Teardown is the scheduler's job.** ``srun`` forwards SIGTERM to its task
    (``worker_entry`` then runs ``ray stop``), and ``proctrack/cgroup`` reaps
    every step when the batch step exits, so there is no remote-``ray stop`` fan
    out and no self-``scancel`` (the LSF path's self-``bkill`` turns the job
    state to EXIT; here the job ends COMPLETED with main.py's exit code).

The scheduler-agnostic pieces (head start, driver ``ray.init``, worker
spawning, the attach wait, the NCCL/RDMA env, ``worker_entry`` itself) are
reused from :mod:`autotune.lsf`.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import ray

from autotune.lsf.log_utils import banner, dump_env_fingerprint, log_argv, phase_timer, summarize_ray_nodes
from autotune.lsf.ray_up_blaunch import (
    DEFAULT_BRINGUP_DEADLINE_S,
    DEFAULT_FLEET,
    RayUpTimeoutError,
    _default_ib_hca,
    _gpu_model_for_fleet,
    _partition_hosts,
    _ray_init_with_timeout,
    _rdma_env,
    _short_hostname,
    _start_head,
    _wait_for_workers_blaunch,
)
from autotune.tmpdir import resolve_job_id, resolve_ray_temp_dir

logger = logging.getLogger(__name__)

WORKER_MODULE = "autotune.lsf.worker_entry"


def _expand_hostlist(expr: str) -> List[str]:
    """Expand a Slurm hostlist expression (``p1-r01-n[1-4],p2-r03-n[1,3-4]``).

    Pure-Python fallback for :func:`_read_slurm_nodelist` when ``scontrol`` is
    unavailable. Handles comma-separated entries, one bracket group per entry,
    ranges and zero padding (``n[01-03]`` -> ``n01 n02 n03``), which covers what
    Slurm emits for ``SLURM_JOB_NODELIST``.
    """
    hosts: List[str] = []
    # Split on commas that are not inside brackets.
    for entry in re.findall(r"[^,\[]+(?:\[[^\]]*\][^,\[]*)*", expr):
        entry = entry.strip()
        if not entry:
            continue
        m = re.fullmatch(r"([^\[]*)\[([^\]]*)\](.*)", entry)
        if not m:
            hosts.append(entry)
            continue
        prefix, body, suffix = m.groups()
        for part in body.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                width = len(lo)
                for i in range(int(lo), int(hi) + 1):
                    hosts.append(f"{prefix}{i:0{width}d}{suffix}")
            else:
                hosts.append(f"{prefix}{part}{suffix}")
    return hosts


def _read_slurm_nodelist() -> List[str]:
    """Return the ordered, deduped host list of this Slurm allocation.

    Prefers ``scontrol show hostnames`` (authoritative for every hostlist form),
    falling back to :func:`_expand_hostlist`. Slurm orders the list so that the
    first host is the one running the batch step, i.e. this driver.

    Raises:
        RuntimeError: if ``SLURM_JOB_NODELIST`` is unset or empty, i.e. we are
            not inside a Slurm allocation.
    """
    expr = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST") or ""
    if not expr.strip():
        raise RuntimeError(
            "could not determine the Slurm host list: SLURM_JOB_NODELIST is not set. "
            "--scheduler slurm must run inside an sbatch/salloc allocation "
            "(e.g. sbatch -N 2 --gres=gpu:8 --exclusive ...)."
        )
    hosts: Optional[List[str]] = None
    try:
        proc = subprocess.run(["scontrol", "show", "hostnames", expr], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and proc.stdout.strip():
            hosts = proc.stdout.split()
        else:
            logger.warning(f"[ray_up_srun] scontrol show hostnames rc={proc.returncode}: {proc.stderr.strip()!r}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"[ray_up_srun] scontrol unavailable ({e}); expanding {expr!r} in Python")
    if hosts is None:
        hosts = _expand_hostlist(expr)

    seen: set = set()
    ordered = [h for h in hosts if not (h in seen or seen.add(h))]
    if not ordered:
        raise RuntimeError(f"SLURM_JOB_NODELIST={expr!r} expanded to no hosts")
    return ordered


def _cpus_per_node(hosts: List[str]) -> Dict[str, int]:
    """Map each host to the CPUs Slurm allocated on it.

    ``SLURM_JOB_CPUS_PER_NODE`` is run-length encoded in host-list order:
    ``96(x2),48`` means 96 on the first two hosts and 48 on the third. Missing or
    malformed values fall back to ``SLURM_CPUS_ON_NODE`` (this node's count) or
    ``os.cpu_count()``, applied to every host.
    """
    fallback = int(os.environ.get("SLURM_CPUS_ON_NODE") or os.cpu_count() or 1)
    raw = os.environ.get("SLURM_JOB_CPUS_PER_NODE", "")
    counts: List[int] = []
    for tok in filter(None, (t.strip() for t in raw.split(","))):
        m = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", tok)
        if not m:
            logger.warning(f"[ray_up_srun] cannot parse SLURM_JOB_CPUS_PER_NODE={raw!r}; using {fallback}")
            counts = []
            break
        counts.extend([int(m.group(1))] * int(m.group(2) or 1))
    if len(counts) != len(hosts):
        if raw:
            logger.warning(
                f"[ray_up_srun] SLURM_JOB_CPUS_PER_NODE={raw!r} covers {len(counts)} host(s), "
                f"allocation has {len(hosts)}; using {fallback} CPUs per host"
            )
        counts = [fallback] * len(hosts)
    return dict(zip(hosts, counts))


def _worker_args(*, head_address: str, gpus_per_worker: int, cpus: int) -> List[str]:
    return [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--head_address",
        head_address,
        "--num_gpus",
        str(gpus_per_worker),
        "--num_cpus",
        str(cpus),
    ]


def _build_srun_cmd(*, host: str, head_address: str, gpus_per_worker: int, cpus: int) -> List[str]:
    """Build the ``srun`` argv for one remote worker (one step per host).

    One step per host (rather than one multi-node step) gives one ``Popen`` per
    worker, so a dead node is detected and named individually and every host gets
    its own log file — the same reason the LSF path uses ``blaunch -z host``.

      * ``--overlap``: the step may share resources with other steps, so the
        workers never wait on each other and a later diagnostic ``srun`` into the
        allocation is not blocked behind them.
      * ``--cpus-per-task``: see the module docstring; the default is one CPU.
      * ``--mem=0``: all of the job's memory on that node.
      * ``--mpi=none``: this cluster defaults to PMIx, which a non-MPI Python
        process does not need and only slows step launch.
      * No ``--gres``: a step inherits every GRES the job holds on its node by
        default, and naming one would clash with jobs submitted with
        ``--gpus``/``--gpus-per-node`` instead of ``--gres``. ``worker_entry``'s
        GPU-count guard fails the worker clearly if inheritance ever gives less.
    """
    return [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={host}",
        f"--cpus-per-task={cpus}",
        "--mem=0",
        "--overlap",
        "--mpi=none",
        "--kill-on-bad-exit=1",
        f"--job-name=ray_worker_{host}",
        *_worker_args(head_address=head_address, gpus_per_worker=gpus_per_worker, cpus=cpus),
    ]


def _popen_worker(argv: List[str], log_path: str, env: Dict[str, str]) -> subprocess.Popen:
    """Spawn a worker in its own session with stdout/stderr redirected to ``log_path``.

    Same rationale as :func:`autotune.lsf.ray_up_blaunch._popen_detached`: a new
    session lets teardown signal the whole process group, and redirecting at spawn
    keeps the child off the batch step's output file. Takes an explicit ``env``
    because the workers are not launched through a shell that could add the RDMA
    variables on the command line.
    """
    log_fh = open(log_path, "ab", buffering=0)
    try:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
    finally:
        log_fh.close()


def start_multinode_ray_cluster_srun(
    num_workers: int,
    gpus_per_worker: int,
    *,
    fleet: str = DEFAULT_FLEET,
    ib_hca: Optional[str] = None,
    ib_ifname: Optional[str] = None,
    wait_timeout_s: int = 1800,
    log_dir: Optional[str] = None,
    bringup_deadline_s: int = DEFAULT_BRINGUP_DEADLINE_S,
    rl_algo: str = "none",
) -> Dict[str, Any]:
    """Bring up a Ray cluster inside a single Slurm allocation via ``srun``.

    Arguments mirror :func:`autotune.lsf.ray_up_blaunch.start_multinode_ray_cluster_blaunch`
    minus ``conda_env`` (srun propagates the environment) and ``cores_per_worker``
    (read per host from the allocation). ``num_workers`` must equal the number of
    allocated nodes.

    Returns:
        Handle dict for :func:`autotune.slurm.ray_down.stop_multinode_ray_cluster_srun`.
        ``scheduler`` is ``"slurm"``.

    Raises:
        RayUpTimeoutError: bring-up did not complete within ``bringup_deadline_s``
            (partial teardown already attempted).
        RuntimeError: not inside a Slurm allocation, node count mismatch, or a
            worker exited during bring-up.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1 for the srun path")
    if not log_dir:
        raise ValueError("log_dir is required (e.g. f'{output_dir}/logs'); see main.py for the expected wiring")

    resolved_ib_hca = ib_hca if ib_hca else _default_ib_hca(fleet)
    resolved_gpu_model = _gpu_model_for_fleet(fleet)

    banner("ray_up_srun: starting single-allocation cluster")
    dump_env_fingerprint()
    log_dir_abs = os.path.abspath(log_dir)
    logger.info(
        f"[ray_up_srun] kwargs: num_workers={num_workers} gpus_per_worker={gpus_per_worker} "
        f"gpu_model={resolved_gpu_model!r} fleet={fleet!r} ib_hca={resolved_ib_hca!r} "
        f"ib_ifname={ib_ifname!r} wait_timeout_s={wait_timeout_s} log_dir={log_dir_abs!r} "
        f"rl_algo={rl_algo!r} SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')!r}"
    )
    logger.info(f"[ray_up_srun] bring-up deadline: {bringup_deadline_s}s")

    t0 = time.monotonic()
    deadline = t0 + float(bringup_deadline_s)

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    def check_deadline(stage: str) -> None:
        if remaining() <= 0:
            raise TimeoutError(f"bring-up deadline ({bringup_deadline_s}s) exceeded at stage={stage}")

    rdma_env = _rdma_env(ib_hca=resolved_ib_hca, ib_ifname=ib_ifname)
    for k, v in rdma_env.items():
        os.environ.setdefault(k, v)
    # Explicit values win over whatever the driver inherited, matching the LSF
    # path, where the rdma env is prefixed onto each worker's command line.
    worker_env = {**os.environ, **rdma_env}

    # Fail fast on a wrong allocation shape before starting Ray.
    host_list = _read_slurm_nodelist()
    head_host, all_worker_hosts, remote_worker_hosts = _partition_hosts(
        host_list, socket.gethostname(), num_workers=num_workers
    )
    # SLURM_JOB_CPUS_PER_NODE is in SLURM_JOB_NODELIST order, so map it onto
    # host_list *before* _partition_hosts rotates the head to the front --
    # mapping it onto the rotated list hands each host another host's count on
    # a mixed-size allocation. Keyed by short name, like all_worker_hosts.
    cpus_by_host = {_short_hostname(h): n for h, n in _cpus_per_node(host_list).items()}
    logger.info(
        f"[ray_up_srun] head_host={head_host!r} all_workers={all_worker_hosts} "
        f"remote_workers={remote_worker_hosts} cpus_per_host={cpus_by_host}"
    )

    job_id = resolve_job_id()
    head_temp_dir = resolve_ray_temp_dir(f"job_{job_id}")

    partial_info: Dict[str, Any] = {
        "scheduler": "slurm",
        "head_address": None,
        "head_ip": None,
        "head_port": None,
        "dashboard_port": None,
        "temp_dir": head_temp_dir,
        "num_gpus": 0,
        "worker_hosts": list(all_worker_hosts),
        "remote_worker_hosts": list(remote_worker_hosts),
        "worker_pids": [],
        "job_group": None,
        "rdma_env": rdma_env,
    }
    popen_records: List[Tuple[str, subprocess.Popen, str]] = []

    try:
        check_deadline("start_head")
        with phase_timer("start_head"):
            head_ip, head_port, dash_port = _start_head(head_temp_dir)
        head_address = f"{head_ip}:{head_port}"
        partial_info.update(head_address=head_address, head_ip=head_ip, head_port=head_port, dashboard_port=dash_port)
        logger.info(f"[ray_up_srun] head_address={head_address} dashboard=http://{head_ip}:{dash_port}")

        job_group = f"ray_nodes_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        partial_info["job_group"] = job_group
        run_log_dir = os.path.join(log_dir_abs, job_group)
        os.makedirs(run_log_dir, exist_ok=True)
        logger.info(f"[ray_up_srun] job_group={job_group} run_log_dir={run_log_dir}")

        check_deadline("ray_init")
        with phase_timer("ray_init_driver"):
            ok = _ray_init_with_timeout(address=head_address, timeout_s=remaining())
        if not ok:
            raise TimeoutError(f"ray.init({head_address!r}) did not return within remaining budget")

        check_deadline("spawn_workers")
        for host in all_worker_hosts:
            cpus = cpus_by_host[host]
            if host == head_host:
                argv = _worker_args(head_address=head_address, gpus_per_worker=gpus_per_worker, cpus=cpus)
            else:
                argv = _build_srun_cmd(host=host, head_address=head_address, gpus_per_worker=gpus_per_worker, cpus=cpus)
            log_path = os.path.join(run_log_dir, f"{host}.log")
            log_argv(f"worker-{host}", argv)
            logger.info(f"[ray_up_srun] worker {host!r} log file: {log_path}  (tail -f to watch)")
            proc = _popen_worker(argv, log_path, worker_env)
            popen_records.append((host, proc, log_path))
            partial_info["worker_pids"].append(proc.pid)
        logger.info(
            f"[ray_up_srun] spawned {len(popen_records)} worker(s) "
            f"(1 local + {len(remote_worker_hosts)} srun): pids={partial_info['worker_pids']}"
        )

        phase_budget = int(min(wait_timeout_s, remaining()))
        if phase_budget <= 0:
            raise TimeoutError("no budget left for the worker-attach wait")
        _wait_for_workers_blaunch(
            expected_total_gpus=num_workers * gpus_per_worker,
            expected_nodes=num_workers + 1,  # +1 for the CPU-only head node
            popen_records=popen_records,
            timeout_s=phase_budget,
        )

        final_resources = ray.cluster_resources()
        summarize_ray_nodes()
        banner("ray_up_srun: cluster ready")
        partial_info["num_gpus"] = int(final_resources.get("GPU", 0))
        logger.info(
            f"[ray_up_srun] head={head_address} GPUs={partial_info['num_gpus']} "
            f"workers={all_worker_hosts} job_group={job_group}"
        )
        logger.info(f"[ray_up_srun] bring-up complete in {time.monotonic() - t0:.1f}s of {bringup_deadline_s}s budget")
        return partial_info

    except BaseException as exc:
        # Unlike LSF, a worker that died during bring-up (RuntimeError) also gets
        # torn down here: the srun steps are children of this process and would
        # otherwise hold the remaining nodes until the batch step exits.
        elapsed = time.monotonic() - t0
        logger.error(f"[ray_up_srun] bring-up failed after {elapsed:.1f}s: {exc!r}")
        try:
            from autotune.slurm.ray_down import stop_multinode_ray_cluster_srun

            stop_multinode_ray_cluster_srun(partial_info)
        except Exception as td_exc:  # noqa: BLE001 — never mask the original error
            logger.warning(f"[ray_up_srun] partial teardown raised (ignored): {td_exc}")
        if isinstance(exc, TimeoutError):
            raise RayUpTimeoutError(str(exc), cluster_info=partial_info) from exc
        raise
