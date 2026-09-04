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
"""Stand up a multi-node Ray cluster on LSF with GPU-Direct RDMA.

The expected flow is:

    1. The user submits ``main.py`` itself via ``bsub`` on a **CPU-only** node
       (the driver / Ray head needs no GPUs of its own).
    2. ``main.py`` calls :func:`start_multinode_ray_cluster`, which:
        - starts ``ray start --head`` locally (CPU-only, schedules nothing)
        - submits ``num_workers`` GPU ``bsub`` jobs, each carrying
          ``gpus_per_worker`` GPUs and joining the head
        - waits until every worker has reached LSF ``RUN`` and registered
          with Ray
    3. ``main.py`` then runs HPO / training as usual against the cluster.
       All GPU work happens on workers; the head stays CPU-only.
    4. On exit, ``ray_down.stop_multinode_ray_cluster`` is called with the
       handle returned here.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import ray

from autotune.cluster import release_sockets, reserve_ports
from autotune.logging_setup import bridge_runtime_env
from autotune.lsf.log_utils import (
    banner,
    bjobs_long,
    dump_env_fingerprint,
    log_argv,
    phase_timer,
    summarize_ray_nodes,
)

logger = logging.getLogger(__name__)


# Hard cap for the whole bring-up (head start + worker bsubs + driver ray.init
# + worker-attach wait). Beyond this, ``start_multinode_ray_cluster`` aborts,
# tears down whatever it brought up, and raises ``RayUpTimeoutError`` so the
# caller can finish the run gracefully without HPO.
DEFAULT_BRINGUP_DEADLINE_S = 1200  # 20 minutes


class RayUpTimeoutError(TimeoutError):
    """Raised when ``start_multinode_ray_cluster`` exceeds its bring-up deadline.

    Carries the partial cluster handle (``cluster_info``) so the caller can
    inspect / log what was set up before the timeout.  Partial teardown has
    already been attempted by the time this is raised.
    """

    def __init__(self, message: str, cluster_info: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.cluster_info = cluster_info or {}


def _ray_init_with_timeout(address: str, timeout_s: float) -> bool:
    """Run ``ray.init(address=...)`` in a thread with a hard timeout.

    Returns ``True`` on success, ``False`` if the call did not return within
    ``timeout_s``.  Mirrors the pattern used in
    ``autotune.lsf.ray_down._ray_shutdown_with_timeout`` — Ray's gRPC client
    can wedge if the head's GCS / dashboard agent is unhealthy, and we don't
    want that to pin the bring-up beyond the deadline.
    """
    if timeout_s <= 0:
        return False
    done = threading.Event()
    err: List[BaseException] = []

    def _target() -> None:
        try:
            ray.init(
                address=address,
                log_to_driver=True,
                logging_level=logging.INFO,
                **bridge_runtime_env(),
            )
        except BaseException as e:  # noqa: BLE001 — best-effort
            err.append(e)
        finally:
            done.set()

    t = threading.Thread(target=_target, name="ray-init", daemon=True)
    t.start()
    if not done.wait(timeout=timeout_s):
        return False
    if err:
        raise err[0]
    return True


def _rdma_env(ib_hca: str = "mlx5_0", ib_ifname: Optional[str] = None) -> Dict[str, str]:
    """Return the NCCL/RDMA env-var set required for inter-node GDR over IB.

    Notes:
        * ``NCCL_IB_HCA`` defaults to ``mlx5_0`` only. Some A100 fleet nodes
          have ``mlx5_1`` Down — listing it forces NCCL to attempt the dark
          NIC. Pass a different value if both NICs are healthy on your fleet.
        * ``NCCL_SOCKET_IFNAME`` is **not** set by default. NCCL's bootstrap
          (the rendezvous before any RDMA) must run over a healthy TCP
          interface on every rank. A CPU-only head usually does not have
          IPoIB (``ib0``/``ib1``) configured, so pinning the bootstrap to
          IPoIB causes ``Bootstrap : no socket interface found``. With this
          var unset, NCCL auto-picks the routable Ethernet interface — the
          same one Ray uses for GCS/TCPStore. Pass ``ib_ifname=...``
          explicitly only if you want to override that.
        * The remaining values come from internal RDMA tuning measurements.
    """
    env = {
        "NCCL_NET_GDR_LEVEL": "PXB",
        "NCCL_IB_HCA": ib_hca,
        "NCCL_IB_CUDA_SUPPORT": "1",
        "NCCL_IB_GDR_LEVEL": "5",
        "NCCL_IB_DISABLE": "0",
        "NCCL_DEBUG": "WARN",
        # Throughput tunings — better link utilization on a shared HCA, also
        # safe (and beneficial) once dual-rail RDMA becomes available.
        "NCCL_IB_QPS_PER_CONNECTION": "4",
        "NCCL_IB_SPLIT_DATA_ON_QPS": "1",
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_MIN_NCHANNELS": "4",
    }
    if ib_ifname:
        env["NCCL_SOCKET_IFNAME"] = ib_ifname
    return env


def _start_head(temp_dir: str) -> Tuple[str, int, int]:
    """Run ``ray start --head`` and return ``(head_ip, head_port, dashboard_port)``."""
    # Reserve all ports atomically: head, dashboard, node-manager, object-manager,
    # dashboard-agent grpc, dashboard-agent listen, metrics-export, ray-client-server.
    ports, sockets = reserve_ports(8)
    head_port, dash_port, nmp, omp, dagp, dalp, mep, rcsp = ports

    cmd = [
        "ray",
        "start",
        "--head",
        "--port",
        str(head_port),
        "--dashboard-port",
        str(dash_port),
        "--node-manager-port",
        str(nmp),
        "--object-manager-port",
        str(omp),
        "--dashboard-agent-grpc-port",
        str(dagp),
        "--dashboard-agent-listen-port",
        str(dalp),
        "--metrics-export-port",
        str(mep),
        "--ray-client-server-port",
        str(rcsp),
        "--include-dashboard",
        "True",
        # Bind the dashboard to loopback only. Ray's dashboard / job-submission
        # API has no authentication, so exposing it on 0.0.0.0 is the "ShadowRay"
        # unauthenticated-RCE footgun on a shared fabric. SSH-tunnel to the head
        # node if you need the UI.
        "--dashboard-host",
        "127.0.0.1",
        "--num-cpus",
        "0",
        "--num-gpus",
        "0",
        "--object-store-memory",
        "10000000000",
        "--temp-dir",
        temp_dir,
        "--disable-usage-stats",
    ]
    log_argv("ray-start-head", cmd)
    # Hold reserved sockets open until the very last moment, then release them
    # so Ray can bind.  Closing earlier opens a TOCTOU window where another
    # process (or even a fast Ray subprocess race) can steal the port.
    release_sockets(sockets)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"ray start --head failed (rc={proc.returncode})")
        logger.error(f"stdout:\n{proc.stdout}")
        logger.error(f"stderr:\n{proc.stderr}")
        raise RuntimeError(f"ray start --head failed: {proc.stderr.strip()}")
    logger.info(f"ray head started:\n{proc.stdout}")

    hostname = socket.gethostname()
    head_ip = socket.gethostbyname(hostname)
    logger.info(
        f"[ray_up] head: hostname={hostname} ip={head_ip} port={head_port} "
        f"dashboard_port={dash_port} temp_dir={temp_dir}"
    )
    return head_ip, head_port, dash_port


def _build_worker_bsub(
    *,
    idx: int,
    head_address: str,
    job_group: str,
    queue: str,
    user_group: str,
    cores_per_worker: int,
    gpus_per_worker: int,
    mem_per_worker_gb: int,
    gpu_model: str,
    gpu_mode: str,
    conda_env: str,
    log_dir: str,
    rdma_env: Dict[str, str],
) -> Tuple[List[str], str]:
    """Build the ``bsub`` argv for a single worker job.

    Returns ``(argv, abs_log_path)`` so the caller can echo the log location.

    ``gpu_mode`` is auto-derived by the caller (``start_multinode_ray_cluster``)
    from ``rl_algo``:

      * ``"exclusive_process"`` for SFT/PEFT/offline-RL runs. LSF reserves
        the GPUs exclusively at the scheduler level, which by itself prevents
        any other GPU job from co-locating on the same host.
      * ``"shared"`` for online RL runs (verl/vLLM spawns multiple processes
        per GPU which requires shareable mode). Because shared GPUs do *not*
        keep LSF from co-scheduling other jobs on the host, we additionally
        pass ``-x`` (host-exclusive job) so the worker still owns the whole
        physical box. Same end result as the SFT path; different mechanism.
    """
    log_path = os.path.abspath(os.path.join(log_dir, f"worker_{idx}.log"))

    env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in rdma_env.items())
    inner_cmd = (
        f"source ~/.bashrc && conda activate {shlex.quote(conda_env)} && "
        f"env {env_prefix} python -m autotune.lsf.worker_entry "
        f"--head_address {head_address} "
        f"--num_gpus {gpus_per_worker} "
        f"--num_cpus {cores_per_worker}"
    )

    if gpu_mode == "shared":  # for VERL
        gpu_str = f"num={gpus_per_worker}:gmodel={gpu_model}"
    else:  # for SFT/PEFT/offline RL
        gpu_str = f"num={gpus_per_worker}:mode=exclusive_process:gmodel={gpu_model}"

    argv = [
        "bsub",
        "-q",
        queue,
        "-U",
        user_group,
        "-J",
        job_group,
        "-P",
        job_group,
        "-n",
        str(cores_per_worker),
        "-gpu",
        gpu_str,
        "-R",
        f"rusage[mem={mem_per_worker_gb}G]",
        "-R",
        "span[hosts=1]",
    ]
    # For verl (shared GPU mode) we lose LSF's GPU-level scheduling
    # exclusivity, so add ``-x`` to claim the whole host for this job.
    # Without this an unrelated GPU job could co-schedule on the same box
    # and steal devices that verl's actor/critic/ref/rollout actors use.
    if gpu_mode == "shared":
        argv.append("-x")
    argv += [
        "-o",
        log_path,
        "bash",
        "-lc",
        inner_cmd,
    ]
    return argv, log_path


def _submit_worker(cmd: List[str], idx: int) -> str:
    """Submit one worker ``bsub`` and return the LSF job id (as a string)."""
    log_argv(f"bsub-worker-{idx}", cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(f"bsub worker-{idx} failed (rc={proc.returncode})")
        logger.error(f"stdout:\n{proc.stdout}")
        logger.error(f"stderr:\n{proc.stderr}")
        raise RuntimeError(f"bsub failed (rc={proc.returncode}): {proc.stderr.strip()}")
    # Typical LSF stdout: "Job <12345> is submitted to queue <normal>."
    m = re.search(r"Job <(\d+)>", proc.stdout)
    if not m:
        raise RuntimeError(f"could not parse LSF job id from stdout: {proc.stdout!r}")
    job_id = m.group(1)
    logger.info(f"submitted worker-{idx} job_id={job_id}: {proc.stdout.strip()}")
    return job_id


def _bjobs_status(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """Return ``{job_id: {"status": ..., "host": ...}}``.

    Missing jobs map to ``{"status": "GONE", "host": "-"}``.  ``bjobs`` columns
    are parsed positionally: ``JOBID USER STAT QUEUE FROM_HOST EXEC_HOST ...``.
    """
    out: Dict[str, Dict[str, str]] = {jid: {"status": "GONE", "host": "-"} for jid in job_ids}
    if not job_ids:
        return out
    proc = subprocess.run(
        ["bjobs", "-noheader"] + job_ids,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in out:
            out[parts[0]]["status"] = parts[2]
            if len(parts) >= 6:
                out[parts[0]]["host"] = parts[5]
    return out


def _log_bjobs_table(statuses: Dict[str, Dict[str, str]], elapsed: float) -> None:
    """Log a per-job status table (one row per worker)."""
    logger.info(f"[ray_up] bjobs snapshot (elapsed={elapsed:.0f}s):")
    logger.info(f"[ray_up]   {'JOBID':<10} {'STAT':<6} {'HOST':<24}")
    for jid, info in statuses.items():
        logger.info(f"[ray_up]   {jid:<10} {info['status']:<6} {info['host']:<24}")


def _wait_for_workers(
    *,
    expected_total_gpus: int,
    worker_job_ids: List[str],
    timeout_s: int,
) -> None:
    """Poll until (a) every worker job is RUN and (b) Ray sees all GPUs."""
    deadline = time.monotonic() + timeout_s
    poll_s = 15
    started_at = time.monotonic()
    long_dump_after_s = 60.0
    long_dumped: set = set()

    # Phase 1: wait for every worker bsub to reach RUN.
    with phase_timer("wait_workers_run"):
        while True:
            statuses = _bjobs_status(worker_job_ids)
            elapsed = time.monotonic() - started_at
            running = {j for j, info in statuses.items() if info["status"] == "RUN"}
            not_running = {j: info for j, info in statuses.items() if info["status"] != "RUN"}
            _log_bjobs_table(statuses, elapsed)
            logger.info(
                f"[ray_up] worker bjobs: RUN={len(running)}/{len(worker_job_ids)} non_running={len(not_running)}"
            )
            # Dump bjobs -l once for any worker that's been non-RUN > long_dump_after_s.
            if elapsed >= long_dump_after_s:
                for jid in not_running:
                    if jid not in long_dumped:
                        long_dumped.add(jid)
                        logger.info(f"[ray_up] bjobs -l {jid} (non-RUN > {long_dump_after_s:.0f}s):\n{bjobs_long(jid)}")
            if not not_running:
                break
            terminal = {j: i["status"] for j, i in not_running.items() if i["status"] in {"EXIT", "DONE", "GONE"}}
            if terminal:
                for jid in terminal:
                    logger.error(f"[ray_up] terminal status for {jid}: bjobs -l:\n{bjobs_long(jid)}")
                raise RuntimeError(f"worker job(s) failed before reaching RUN: {terminal}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for worker jobs to RUN after {timeout_s}s. statuses={statuses}")
            time.sleep(poll_s)

    # Phase 2: wait for Ray to see all GPUs.
    with phase_timer("wait_workers_attached"):
        while True:
            try:
                gpus = float(ray.cluster_resources().get("GPU", 0.0))
            except Exception as e:
                logger.warning(f"[ray_up] ray.cluster_resources() failed: {e}")
                gpus = 0.0
            logger.info(f"[ray_up] ray.cluster_resources GPU={gpus} / expected={expected_total_gpus}")
            summarize_ray_nodes()
            if gpus >= expected_total_gpus:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"timed out waiting for Ray to see {expected_total_gpus} GPUs (saw {gpus}) after {timeout_s}s"
                )
            time.sleep(poll_s)


def start_multinode_ray_cluster(
    num_workers: int,
    gpus_per_worker: int,
    conda_env: str,
    *,
    cores_per_worker: int = 32,
    mem_per_worker_gb: int = 640,
    queue: str = "normal",
    user_group: str = "infusion",
    gpu_model: str = "NVIDIAA100_SXM4_80GB",
    ib_hca: str = "mlx5_0",
    ib_ifname: Optional[str] = None,
    wait_timeout_s: int = 1800,
    log_dir: str = "logs",
    bringup_deadline_s: int = DEFAULT_BRINGUP_DEADLINE_S,
    rl_algo: str = "none",
) -> Dict[str, Any]:
    """Start a multi-node Ray cluster on LSF with RDMA-enabled NCCL.

    The current process becomes the **Ray head** (and the user's driver).
    The head is CPU-only — it schedules zero workloads itself.
    ``num_workers`` additional LSF jobs are submitted, each providing one
    GPU worker node with ``gpus_per_worker`` GPUs; all training happens on
    those workers.

    Args:
        num_workers: number of GPU worker nodes to add. Must be >= 0; a value
            of 0 brings up a head with no workers (no useful work possible —
            a warning is logged).
        gpus_per_worker: GPUs requested for each worker.
        conda_env: conda env path/name workers must activate before ``ray start``.
        cores_per_worker: cores per worker (default 32).
        mem_per_worker_gb: per-worker memory in GB (default 640).
        queue: LSF queue (default ``normal``).
        user_group: LSF user group passed via ``-U`` (default ``infusion``).
        gpu_model: ``-gpu gmodel=...`` value (default A100 80GB).
        ib_hca: NCCL InfiniBand HCA list (``NCCL_IB_HCA``). Default is
            ``mlx5_0`` because ``mlx5_1`` is Down on the A100 fleet nodes we
            have evidence for; widen to ``mlx5_0,mlx5_1`` if both are Active
            in your fleet (verify via ``ibstat`` in the worker pre-flight log).
        ib_ifname: Optional override for ``NCCL_SOCKET_IFNAME`` (the TCP
            interface NCCL uses for **bootstrap**, before RDMA kicks in).
            Default is ``None`` — NCCL auto-picks the routable Ethernet
            interface, which is what we want when the head is CPU-only and
            doesn't have IPoIB configured. Set explicitly (e.g. ``"ib0"``)
            only if every rank has the named IPoIB iface up.
        wait_timeout_s: hard cap on the wait-for-workers phase. Always
            clamped by the remaining ``bringup_deadline_s`` budget.
        log_dir: parent directory for ``-o`` worker logs (default ``logs``;
            created if missing). The actual log files land in a per-run
            subdirectory: ``<log_dir>/<job_group>/worker_<i>.log``, where
            ``job_group`` is the auto-generated ``ray_nodes_<timestamp>``.
            This keeps logs from successive runs separate.
        bringup_deadline_s: hard cap (in seconds) on the *entire* bring-up
            from entry to "cluster ready" — covers ``ray start --head``, the
            worker ``bsub`` submissions, the driver-side ``ray.init``, and
            the worker-attach wait. If exceeded, this function partially
            tears down whatever was set up and raises
            :class:`RayUpTimeoutError` so the caller can finish the run
            gracefully (e.g. log a warning and exit 0). Default
            :data:`DEFAULT_BRINGUP_DEADLINE_S` (10 minutes).
        rl_algo: name of the RL algorithm for this run (``"none"`` /
            ``"dpo"`` / ``"kto"`` / ``"ppo"`` / ``"grpo"`` / ``"dapo"``).
            Used **only** to auto-select the LSF GPU exclusivity mode in
            the worker ``bsub``: online RL (``ppo``/``grpo``/``dapo``) gets
            ``mode=shared`` because verl/vLLM spawns multiple processes per
            GPU; everything else gets ``mode=exclusive_process`` so LSF
            actually reserves the GPUs and won't double-book the same host
            for two workers. Default ``"none"``.

    Returns:
        A handle dict to pass to ``ray_down.stop_multinode_ray_cluster``.

    Raises:
        RayUpTimeoutError: if bring-up does not complete within
            ``bringup_deadline_s``. The exception carries the partial
            cluster handle on ``.cluster_info``; partial teardown has
            already been attempted.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    banner("ray_up: starting multi-node cluster")
    dump_env_fingerprint()
    log_dir_abs = os.path.abspath(log_dir)
    logger.info(
        f"[ray_up] kwargs: num_workers={num_workers} gpus_per_worker={gpus_per_worker} "
        f"cores_per_worker={cores_per_worker} mem_per_worker_gb={mem_per_worker_gb} "
        f"queue={queue!r} user_group={user_group!r} gpu_model={gpu_model!r} "
        f"ib_hca={ib_hca!r} ib_ifname={ib_ifname!r} "
        f"wait_timeout_s={wait_timeout_s} log_dir={log_dir_abs!r} conda_env={conda_env!r} "
        f"rl_algo={rl_algo!r}"
    )
    logger.info(f"[ray_up] bring-up deadline: {bringup_deadline_s}s")

    # Auto-select LSF GPU exclusivity from rl_algo. Online RL (verl/vLLM)
    # spawns multiple processes per GPU and needs shared mode; everything
    # else (SFT/PEFT/offline-RL) wants exclusive_process so LSF reserves the
    # GPUs and won't co-locate two worker bsubs on the same host.
    from autotune.constants import AUTOTUNE_ONLINE_RL  # local import: avoid heavy package init at module load

    is_online_rl = rl_algo in AUTOTUNE_ONLINE_RL
    gpu_mode = "shared" if is_online_rl else "exclusive_process"
    gpu_mode_reason = (
        f"rl_algo={rl_algo!r} is online RL — verl needs shared GPUs"
        if is_online_rl
        else f"rl_algo={rl_algo!r} is not online RL"
    )
    logger.info(f"[ray_up] gpu_mode={gpu_mode} ({gpu_mode_reason})")
    # Host-level exclusivity policy. With exclusive_process, LSF refuses to
    # co-schedule another GPU job on the same host already.  With shared, we
    # add ``-x`` to the worker bsub so the worker still owns the whole host.
    logger.info(f"[ray_up] host_exclusive={'-x (whole-host)' if is_online_rl else 'GPU-scheduler enforced'}")
    if num_workers == 0:
        logger.warning(
            "[ray_up] num_workers=0 — head will come up with no GPU workers attached. "
            "No training can run on this cluster. Pass --num_nodes >= 1 to add workers."
        )

    # Hard deadline tracking. ``remaining()`` is checked at every major step
    # below; if it ever hits 0 we raise TimeoutError, which the outer except
    # catches and re-raises as RayUpTimeoutError after partial teardown.
    t0 = time.monotonic()
    deadline = t0 + float(bringup_deadline_s)

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    def check_deadline(stage: str) -> None:
        if remaining() <= 0:
            raise TimeoutError(f"bring-up deadline ({bringup_deadline_s}s) exceeded at stage={stage}")

    rdma_env = _rdma_env(ib_hca=ib_hca, ib_ifname=ib_ifname)
    # Apply on the head process so the driver also benefits.
    for k, v in rdma_env.items():
        os.environ.setdefault(k, v)

    job_id = os.environ.get("LSB_JOBID", "0")
    temp_dir = f"/tmp/ray/job_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    # Partial cluster info — populated as we go so a timeout at any point
    # can hand a usable handle to teardown (and onwards to the caller).
    partial_info: Dict[str, Any] = {
        "head_address": None,
        "head_ip": None,
        "head_port": None,
        "dashboard_port": None,
        "temp_dir": temp_dir,
        "num_gpus": 0,
        "worker_job_ids": [],
        "job_group": None,
        "rdma_env": rdma_env,
    }

    try:
        check_deadline("start_head")
        with phase_timer("start_head"):
            head_ip, head_port, dash_port = _start_head(temp_dir)
        head_address = f"{head_ip}:{head_port}"
        partial_info.update(head_address=head_address, head_ip=head_ip, head_port=head_port, dashboard_port=dash_port)
        logger.info(f"[ray_up] head_address={head_address} dashboard=http://{head_ip}:{dash_port} temp_dir={temp_dir}")

        job_group = f"ray_nodes_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        partial_info["job_group"] = job_group
        logger.info(f"[ray_up] job_group={job_group}")

        # Per-run subdir under the configured log root so reruns don't clobber
        # earlier worker logs.  Final shape: <log_dir>/<job_group>/worker_<i>.log
        run_log_dir = os.path.join(log_dir_abs, job_group)
        os.makedirs(run_log_dir, exist_ok=True)
        logger.info(f"[ray_up] worker log dir: {run_log_dir}")

        # Best-effort topology + IB stack dumps for postmortem RDMA debugging.
        # These are expected to be absent on a CPU-only head, so missing-binary
        # / non-zero-rc paths are DEBUG (not WARNING).
        try:
            topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=15)
            if topo.returncode == 0:
                logger.info(f"[ray_up] nvidia-smi topo -m:\n{topo.stdout}")
            else:
                logger.debug(f"[ray_up] nvidia-smi topo -m rc={topo.returncode} stderr={topo.stderr.strip()}")
        except FileNotFoundError:
            logger.debug("[ray_up] nvidia-smi not on PATH (CPU-only head?) — skipping topo dump")
        except Exception as e:
            logger.debug(f"nvidia-smi topo skipped: {e}")
        try:
            ib = subprocess.run(["ibstat"], capture_output=True, text=True, timeout=15)
            if ib.returncode == 0:
                logger.info(f"[ray_up] ibstat:\n{ib.stdout}")
            else:
                logger.debug(f"[ray_up] ibstat rc={ib.returncode} stderr={ib.stderr.strip()}")
        except FileNotFoundError:
            logger.debug("[ray_up] ibstat not on PATH (CPU-only head?) — skipping IB dump")
        except Exception as e:
            logger.debug(f"ibstat skipped: {e}")

        worker_job_ids: List[str] = []
        partial_info["worker_job_ids"] = worker_job_ids  # share by reference
        for i in range(num_workers):
            check_deadline(f"submit_worker_{i}")
            argv, log_path = _build_worker_bsub(
                idx=i,
                head_address=head_address,
                job_group=job_group,
                queue=queue,
                user_group=user_group,
                cores_per_worker=cores_per_worker,
                gpus_per_worker=gpus_per_worker,
                mem_per_worker_gb=mem_per_worker_gb,
                gpu_model=gpu_model,
                gpu_mode=gpu_mode,
                conda_env=conda_env,
                log_dir=run_log_dir,
                rdma_env=rdma_env,
            )
            logger.info(f"[ray_up] worker-{i} log file: {log_path}  (tail -f to watch)")
            worker_job_ids.append(_submit_worker(argv, idx=i))

        logger.info(f"[ray_up] submitted {len(worker_job_ids)} worker job(s): {worker_job_ids}")

        # Connect the driver to its own head before polling cluster_resources().
        # Bounded by the remaining bring-up budget so a wedged GCS / dashboard
        # agent can't pin us indefinitely.
        check_deadline("ray_init")
        with phase_timer("ray_init_driver"):
            ok = _ray_init_with_timeout(address=head_address, timeout_s=remaining())
        if not ok:
            raise TimeoutError(f"ray.init({head_address!r}) did not return within remaining budget")

        if num_workers > 0:
            expected_total = num_workers * gpus_per_worker
            # Clamp the per-phase wait to whatever budget is left.
            phase_budget = int(min(wait_timeout_s, remaining()))
            if phase_budget <= 0:
                raise TimeoutError("no budget left for _wait_for_workers")
            _wait_for_workers(
                expected_total_gpus=expected_total,
                worker_job_ids=worker_job_ids,
                timeout_s=phase_budget,
            )

        final_resources = ray.cluster_resources()
        summarize_ray_nodes()
        banner("ray_up: cluster ready")
        elapsed = time.monotonic() - t0
        logger.info(
            f"[ray_up] head={head_address} dashboard=http://{head_ip}:{dash_port} "
            f"GPUs={int(final_resources.get('GPU', 0))} workers={worker_job_ids} job_group={job_group}"
        )
        logger.info(f"[ray_up] bring-up complete in {elapsed:.1f}s of {bringup_deadline_s}s budget")

        return {
            "head_address": head_address,
            "head_ip": head_ip,
            "head_port": head_port,
            "dashboard_port": dash_port,
            "temp_dir": temp_dir,
            "num_gpus": int(final_resources.get("GPU", 0)),
            "worker_job_ids": worker_job_ids,
            "job_group": job_group,
            "rdma_env": rdma_env,
        }

    except TimeoutError as exc:
        elapsed = time.monotonic() - t0
        logger.error(f"[ray_up] bring-up timed out after {elapsed:.1f}s (deadline={bringup_deadline_s}s): {exc}")
        # Best-effort partial teardown so workers we already submitted are
        # killed and the head process is stopped.  Local import avoids the
        # ray_up <-> ray_down import cycle at module load.
        try:
            from autotune.lsf.ray_down import stop_multinode_ray_cluster

            stop_multinode_ray_cluster(partial_info)
        except Exception as td_exc:  # noqa: BLE001 — never mask the timeout
            logger.warning(f"[ray_up] partial teardown raised (ignored): {td_exc}")
        raise RayUpTimeoutError(str(exc), cluster_info=partial_info) from exc
