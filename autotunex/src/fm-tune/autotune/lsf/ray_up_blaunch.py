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
"""Stand up a multi-node Ray cluster inside a single LSF allocation via blaunch.

The expected flow is:

    1. The user submits ``main.py`` via ``bsub`` requesting **N GPU hosts**
       in one allocation (``-R "span[hosts=N]"``). The driver process lands
       on the first allocated host.
    2. ``main.py`` calls :func:`start_multinode_ray_cluster_blaunch`, which:
        - reads the LSF host list (``LSB_DJOB_HOSTFILE`` / ``LSB_HOSTS``)
        - starts ``ray start --head`` on the current host (CPU-only,
          schedules nothing)
        - starts a Ray worker on the **head host** directly via
          ``subprocess.Popen`` (no blaunch needed — we are already there)
          which registers that host's GPUs
        - blaunches a Ray worker on each of the other ``N-1`` allocated
          hosts so that all ``N`` hosts contribute their GPUs to the cluster
        - waits until Ray sees all expected GPUs and worker nodes
    3. ``main.py`` runs HPO / training as usual against the cluster.
    4. On exit, :func:`autotune.lsf.ray_down.stop_multinode_ray_cluster`
       tears the cluster down (no ``bkill`` of separate worker jobs needed —
       LSF's RES tears down blaunch'd children when the outer job exits).

Bringing the whole cluster up inside one allocation means a single queue
wait covering every host, at the cost of some CPU contention between the
head and the colocated worker on the first host.
"""

from __future__ import annotations

import logging
import os
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
    dump_env_fingerprint,
    log_argv,
    phase_timer,
    summarize_ray_nodes,
)

logger = logging.getLogger(__name__)


# Hard cap for the whole bring-up (head start + blaunch'd workers + driver
# ray.init + worker-attach wait). Beyond this,
# ``start_multinode_ray_cluster_blaunch`` aborts, tears down whatever it brought
# up, and raises ``RayUpTimeoutError`` so the caller can finish the run
# gracefully without HPO.
DEFAULT_BRINGUP_DEADLINE_S = 1200  # 20 minutes


class RayUpTimeoutError(TimeoutError):
    """Raised when bring-up exceeds its deadline.

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


# Per-fleet default for ``NCCL_IB_HCA``. Resolved via ``_default_ib_hca(fleet)``
# so callers can pick the right rail set without hardcoding hostnames.
#
# A100 fleet: only ``mlx5_0`` is reliably Active across the fleet (``mlx5_1``
# is Down on several boxes we have evidence for).
#
# H100 fleet (8-rail): each node has 10 × 400 Gb/s NDR HCAs (per ``ibstat``);
# 8 are the per-GPU compute rails (one HCA per GPU on the same PCIe complex),
# and ``mlx5_8``/``mlx5_9`` are storage/management. Listing all 8 compute
# rails lets NCCL spread an all-reduce / all-gather across them — single-rail
# use leaves ~7× of the inter-node bandwidth on the floor for collectives big
# enough to be link-bound.
_DEFAULT_IB_HCA_BY_FLEET: Dict[str, str] = {
    "a100": "mlx5_0",
    "h100": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7",
}
DEFAULT_FLEET = "a100"

# Fleet → NVIDIA GPU model string. Purely informational (used only in the
# bring-up log line); the real GPU resource request lives in the outer bsub
# ``-gpu gmodel=...``. The fleet implies the GPU type, so this replaces the old
# standalone --gpu_type CLI arg.
_GPU_MODEL_BY_FLEET: Dict[str, str] = {
    "a100": "NVIDIAA100_SXM4_80GB",
    "h100": "NVIDIAH10080GBHBM3",
}


def _default_ib_hca(fleet: str) -> str:
    """Return the per-fleet default for ``NCCL_IB_HCA``.

    Unknown fleet names fall back to ``DEFAULT_FLEET`` (A100, single rail) and
    log a warning so the operator notices a typo before blaming NCCL.
    """
    if fleet not in _DEFAULT_IB_HCA_BY_FLEET:
        logger.warning(
            f"[rdma] unknown fleet={fleet!r}; falling back to {DEFAULT_FLEET!r} "
            f"(NCCL_IB_HCA={_DEFAULT_IB_HCA_BY_FLEET[DEFAULT_FLEET]!r}). "
            f"Known fleets: {sorted(_DEFAULT_IB_HCA_BY_FLEET)}"
        )
        return _DEFAULT_IB_HCA_BY_FLEET[DEFAULT_FLEET]
    return _DEFAULT_IB_HCA_BY_FLEET[fleet]


def _gpu_model_for_fleet(fleet: str) -> str:
    """Return the NVIDIA GPU model string for a fleet (logging only).

    Unknown fleet names fall back to ``DEFAULT_FLEET``'s model and log a warning,
    mirroring :func:`_default_ib_hca`.
    """
    if fleet not in _GPU_MODEL_BY_FLEET:
        logger.warning(
            f"[rdma] unknown fleet={fleet!r}; using {DEFAULT_FLEET!r} GPU model "
            f"{_GPU_MODEL_BY_FLEET[DEFAULT_FLEET]!r} for logging. "
            f"Known fleets: {sorted(_GPU_MODEL_BY_FLEET)}"
        )
        return _GPU_MODEL_BY_FLEET[DEFAULT_FLEET]
    return _GPU_MODEL_BY_FLEET[fleet]


def _rdma_env(ib_hca: str = "mlx5_0", ib_ifname: Optional[str] = None) -> Dict[str, str]:
    """Return the NCCL/RDMA env-var set required for inter-node GDR over IB.

    Notes:
        * ``NCCL_IB_HCA`` is fleet-dependent. A100 = single rail (``mlx5_0``);
          H100 (8-rail) = 8 compute rails (``mlx5_0..mlx5_7``); see
          ``_DEFAULT_IB_HCA_BY_FLEET``. Override per-call via the ``ib_hca``
          arg (``main.py`` exposes ``--ib_hca`` and ``--fleet``).
        * ``NCCL_SOCKET_IFNAME`` is **not** set by default. NCCL's bootstrap
          (the rendezvous before any RDMA) must run over a healthy TCP
          interface on every rank. A CPU-only head usually does not have
          IPoIB (``ib0``/``ib1``) configured, so pinning the bootstrap to
          IPoIB causes ``Bootstrap : no socket interface found``. With this
          var unset, NCCL auto-picks the routable Ethernet interface — the
          same one Ray uses for GCS/TCPStore. Pass ``ib_ifname=...``
          explicitly only if you want to override that.
        * The remaining values come from internal RDMA tuning measurements.
          The throughput-tuning vars (``NCCL_IB_QPS_PER_CONNECTION=4``,
          ``NCCL_IB_SPLIT_DATA_ON_QPS=1``, ``NCCL_BUFFSIZE=8388608``,
          ``NCCL_MIN_NCHANNELS=4``) were specifically chosen to be safe on
          single-rail and beneficial once multi-rail RDMA is in use, so they
          are correct for both the A100 and H100 fleets without modification.
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
        f"[ray_up_blaunch] head: hostname={hostname} ip={head_ip} port={head_port} "
        f"dashboard_port={dash_port} temp_dir={temp_dir}"
    )
    return head_ip, head_port, dash_port


def _read_lsf_hostfile() -> List[str]:
    """Return the deduped list of physical hosts in this LSF allocation.

    Order is preserved (first host is the one the outer ``bsub`` landed on,
    which is also where this driver runs).

    Sources, in order of preference:
      * ``LSB_DJOB_HOSTFILE`` — newline-delimited file LSF writes for
        distributed jobs. Under LSF 10 this is canonically one host per
        line; under LSF 9 it may contain duplicates (one per slot). Either
        way, dedup-then-take-first is correct.
      * ``LSB_HOSTS`` — space-separated list, one entry per slot.

    Raises:
        RuntimeError: if neither variable is set (i.e. we are not running
            inside an LSF distributed-job allocation). Naming both vars in
            the message keeps misuse-outside-LSF debuggable.
    """
    seen: set = set()
    ordered: List[str] = []

    hostfile = os.environ.get("LSB_DJOB_HOSTFILE")
    raw: Optional[str] = None
    if hostfile and os.path.isfile(hostfile):
        try:
            with open(hostfile) as fh:
                raw = fh.read()
        except OSError as e:
            logger.warning(f"[ray_up_blaunch] failed reading LSB_DJOB_HOSTFILE={hostfile!r}: {e}")
            raw = None

    if raw is None:
        raw = os.environ.get("LSB_HOSTS", "")

    for tok in raw.split():
        h = tok.strip()
        if not h or h in seen:
            continue
        seen.add(h)
        ordered.append(h)

    if not ordered:
        raise RuntimeError(
            "could not determine LSF host list: neither LSB_DJOB_HOSTFILE nor "
            "LSB_HOSTS yielded any hostnames. Are you running inside an LSF "
            "distributed-job allocation? (blaunch mode requires a multi-host bsub)"
        )
    return ordered


def _short_hostname(name: str) -> str:
    """Strip the DNS domain from a hostname (everything after the first dot).

    LSF identifies hosts by their unqualified name in ``LSB_DJOB_HOSTFILE``
    and ``LSB_HOSTS`` (e.g. ``gpu-node-03``), but ``socket.gethostname()`` on
    some clusters returns the FQDN (``gpu-node-03.cluster.example.net``).
    Compare on the short form so the blaunch path works on both naming
    conventions.
    """
    return name.split(".", 1)[0]


def _partition_hosts(
    host_list: List[str],
    head_hostname: str,
    num_workers: int,
) -> Tuple[str, List[str], List[str]]:
    """Split the LSF host list into head + worker roles.

    Hostnames are normalized to their short form (no DNS domain) for the
    membership / position checks, since LSF uses short names while
    ``socket.gethostname()`` may return a FQDN. The returned values are the
    short form, which is also what ``blaunch -z`` expects.

    Returns ``(head, all_worker_hosts, remote_worker_hosts)`` where:
      * ``head`` is the current host's short name.
      * ``all_worker_hosts`` is the full host list (short names), with
        ``head`` first (rotated if LSF gave us a different ordering).
        Length ``num_workers``. Every host runs a Ray worker, including
        the head host (colocated with the head process).
      * ``remote_worker_hosts`` is ``all_worker_hosts[1:]`` — the hosts we
        reach via ``blaunch``. Length ``num_workers - 1``.

    Raises:
        RuntimeError: if ``head_hostname`` is not in ``host_list``, or if
            ``len(host_list) != num_workers``.
    """
    head_short = _short_hostname(head_hostname)
    short_list = [_short_hostname(h) for h in host_list]

    if head_short not in short_list:
        raise RuntimeError(
            f"current host {head_hostname!r} (short={head_short!r}) is not in LSF host list "
            f"{host_list!r}; blaunch path requires the driver to be running on one of the "
            f"allocated hosts"
        )
    rotated = list(short_list)
    if rotated[0] != head_short:
        idx = rotated.index(head_short)
        logger.warning(
            f"[ray_up_blaunch] head host {head_short!r} not at index 0 in LSF list "
            f"(was at {idx}); rotating so head is first"
        )
        rotated = rotated[idx:] + rotated[:idx]

    if len(rotated) != num_workers:
        raise RuntimeError(
            f"LSF allocation has {len(rotated)} host(s) ({rotated}) but num_workers={num_workers} "
            "was requested. Adjust the outer bsub's `span[hosts=N]` so it matches --num_nodes, "
            "or change --num_nodes to match the allocation."
        )

    return rotated[0], rotated, rotated[1:]


def _build_inner_cmd(
    *,
    head_address: str,
    gpus_per_worker: int,
    cores_per_worker: int,
    conda_env: str,
    rdma_env: Dict[str, str],
) -> str:
    """Build the inner bash -lc string used by both local and remote workers.

    Stdout/stderr redirection happens at ``Popen`` time (see ``_spawn_*``),
    NOT in the inner string. This is critical for clean LSF teardown: if
    children inherited the driver's fd 1/2, they would keep the LSF batch
    job's log fds open even after our SIGKILL — the bash shell exits but
    LSF reads "writer still open" and won't reap the job. Redirecting at
    spawn means the child process's fd 1/2 point at the per-host log file
    from the start, never at the driver's stdout.
    """
    env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in rdma_env.items())
    return (
        f"source ~/.bashrc && conda activate {shlex.quote(conda_env)} && "
        f"exec env {env_prefix} python -m autotune.lsf.worker_entry "
        f"--head_address {head_address} "
        f"--num_gpus {gpus_per_worker} "
        f"--num_cpus {cores_per_worker}"
    )


def _build_blaunch_cmd(
    *,
    host: str,
    head_address: str,
    gpus_per_worker: int,
    cores_per_worker: int,
    conda_env: str,
    rdma_env: Dict[str, str],
) -> List[str]:
    """Build the ``blaunch`` argv for a single remote worker.

    Uses ``-z host`` (one host per blaunch invocation) rather than
    ``-u hostfile`` so we get one ``Popen`` per worker, which lets us
    detect per-worker non-zero exits via ``Popen.poll()`` and capture
    distinct per-host log paths.
    """
    inner = _build_inner_cmd(
        head_address=head_address,
        gpus_per_worker=gpus_per_worker,
        cores_per_worker=cores_per_worker,
        conda_env=conda_env,
        rdma_env=rdma_env,
    )
    return ["blaunch", "-z", host, "bash", "-lc", inner]


def _build_local_cmd(
    *,
    head_address: str,
    gpus_per_worker: int,
    cores_per_worker: int,
    conda_env: str,
    rdma_env: Dict[str, str],
) -> List[str]:
    """Build the argv for the head-host-local worker (no blaunch needed)."""
    inner = _build_inner_cmd(
        head_address=head_address,
        gpus_per_worker=gpus_per_worker,
        cores_per_worker=cores_per_worker,
        conda_env=conda_env,
        rdma_env=rdma_env,
    )
    return ["bash", "-lc", inner]


def _popen_detached(argv: List[str], log_path: str) -> subprocess.Popen:
    """Spawn a worker child with stdout/stderr → log file and a new session.

    Two properties matter for clean LSF teardown:

    1. **Redirect at spawn time.** The child's fd 1/2 point at the per-host
       log file, never at the driver's inherited stdout/stderr. Otherwise
       the child holds open a writer on LSF's batch-job log pipe; even
       after we SIGKILL the child, anything reading "EOF on log" stays
       blocked until the *child's* fds close — and the LSF wrapper above
       us reaps the job from that signal.
    2. **New session (``start_new_session=True``).** Each child is the
       leader of its own process group / session, so we can signal the
       whole subtree with ``os.killpg(pid, ...)``. blaunch itself spawns
       helpers; SIGTERM to just the blaunch PID is not always enough.
    """
    log_fh = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        # Popen dup'd the fd into the child; close our handle so we don't
        # keep the file open in the driver process either.
        log_fh.close()
    return proc


def _spawn_local_worker(
    *,
    head_host: str,
    head_address: str,
    gpus_per_worker: int,
    cores_per_worker: int,
    conda_env: str,
    rdma_env: Dict[str, str],
    run_log_dir: str,
) -> Tuple[str, subprocess.Popen, str]:
    """Spawn the worker that colocates with the Ray head on host[0].

    Returns ``(host, popen, log_path)``. The Popen is not waited on —
    the worker runs until SIGTERM (the same lifecycle as the bsub path).
    """
    log_path = os.path.abspath(os.path.join(run_log_dir, f"{head_host}.log"))
    argv = _build_local_cmd(
        head_address=head_address,
        gpus_per_worker=gpus_per_worker,
        cores_per_worker=cores_per_worker,
        conda_env=conda_env,
        rdma_env=rdma_env,
    )
    log_argv(f"local-worker-{head_host}", argv)
    logger.info(f"[ray_up_blaunch] head-host worker log file: {log_path}  (tail -f to watch)")
    proc = _popen_detached(argv, log_path)
    return head_host, proc, log_path


def _spawn_remote_workers(
    *,
    remote_host_list: List[str],
    head_address: str,
    gpus_per_worker: int,
    cores_per_worker: int,
    conda_env: str,
    rdma_env: Dict[str, str],
    run_log_dir: str,
) -> List[Tuple[str, subprocess.Popen, str]]:
    """Spawn one blaunch'd worker per remote host."""
    records: List[Tuple[str, subprocess.Popen, str]] = []
    for host in remote_host_list:
        log_path = os.path.abspath(os.path.join(run_log_dir, f"{host}.log"))
        argv = _build_blaunch_cmd(
            host=host,
            head_address=head_address,
            gpus_per_worker=gpus_per_worker,
            cores_per_worker=cores_per_worker,
            conda_env=conda_env,
            rdma_env=rdma_env,
        )
        log_argv(f"blaunch-worker-{host}", argv)
        logger.info(f"[ray_up_blaunch] worker {host!r} log file: {log_path}  (tail -f to watch)")
        proc = _popen_detached(argv, log_path)
        records.append((host, proc, log_path))
    return records


def _tail_log(log_path: str, n: int = 200) -> str:
    try:
        proc = subprocess.run(
            ["tail", "-n", str(n), log_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout
    except Exception as e:
        return f"<tail failed: {e}>"


def _wait_for_workers_blaunch(
    *,
    expected_total_gpus: int,
    expected_nodes: int,
    popen_records: List[Tuple[str, subprocess.Popen, str]],
    timeout_s: int,
) -> None:
    """Poll until Ray sees all expected GPUs and worker nodes are alive.

    Also polls every worker ``Popen`` each iteration: if any has exited
    non-zero, log the tail of its log file and raise immediately so we
    don't sit out the full timeout when a worker is dead.
    """
    deadline = time.monotonic() + timeout_s
    poll_s = 15
    started_at = time.monotonic()

    with phase_timer("wait_workers_attached"):
        while True:
            # Per-worker liveness check first — fail fast on a dead worker.
            for host, proc, log_path in popen_records:
                rc = proc.poll()
                if rc is not None and rc != 0:
                    logger.error(
                        f"[ray_up_blaunch] worker on {host!r} exited rc={rc}; tail of {log_path}:\n"
                        f"{_tail_log(log_path)}"
                    )
                    raise RuntimeError(f"worker on {host!r} exited rc={rc}")
                if rc is not None and rc == 0:
                    # A worker exiting cleanly during bring-up is also wrong
                    # (workers should block on signal.pause until SIGTERM).
                    logger.error(
                        f"[ray_up_blaunch] worker on {host!r} exited rc=0 unexpectedly; tail of {log_path}:\n"
                        f"{_tail_log(log_path)}"
                    )
                    raise RuntimeError(f"worker on {host!r} exited rc=0 during bring-up")

            try:
                gpus = float(ray.cluster_resources().get("GPU", 0.0))
            except Exception as e:
                logger.warning(f"[ray_up_blaunch] ray.cluster_resources() failed: {e}")
                gpus = 0.0
            try:
                alive_nodes = sum(1 for n in ray.nodes() if n.get("Alive"))
            except Exception as e:
                logger.warning(f"[ray_up_blaunch] ray.nodes() failed: {e}")
                alive_nodes = 0

            elapsed = time.monotonic() - started_at
            logger.info(
                f"[ray_up_blaunch] elapsed={elapsed:.0f}s "
                f"GPU={gpus}/{expected_total_gpus} "
                f"alive_nodes={alive_nodes}/{expected_nodes}"
            )
            summarize_ray_nodes()

            if gpus >= expected_total_gpus and alive_nodes >= expected_nodes:
                return

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"timed out waiting for cluster to assemble after {timeout_s}s "
                    f"(GPU={gpus}/{expected_total_gpus} alive_nodes={alive_nodes}/{expected_nodes})"
                )
            time.sleep(poll_s)


def start_multinode_ray_cluster_blaunch(
    num_workers: int,
    gpus_per_worker: int,
    conda_env: str,
    *,
    cores_per_worker: int = 32,
    fleet: str = DEFAULT_FLEET,
    ib_hca: Optional[str] = None,
    ib_ifname: Optional[str] = None,
    wait_timeout_s: int = 1800,
    log_dir: Optional[str] = None,
    bringup_deadline_s: int = DEFAULT_BRINGUP_DEADLINE_S,
    rl_algo: str = "none",
) -> Dict[str, Any]:
    """Bring up a Ray cluster inside a single LSF allocation via blaunch.

    This does not submit any separate worker ``bsub`` jobs. It assumes the
    caller is already running inside a multi-host LSF allocation
    (``-R "span[hosts=N]"``) and starts:

      * ``ray start --head`` on the current host (CPU-only).
      * One Ray worker on the current host directly (no blaunch — registers
        this host's GPUs).
      * One Ray worker on each of the other allocated hosts via ``blaunch``.

    The handle returned has the same shape as the bsub-path handle so
    :func:`autotune.lsf.ray_down.stop_multinode_ray_cluster` works for both,
    with two additions:

      * ``worker_job_ids`` is always ``[]`` (there are no separate LSF jobs).
      * ``worker_hosts`` lists all N hosts running a Ray worker (head host
        first), and ``worker_pids`` lists the corresponding PIDs.

    Args:
        num_workers: number of GPU worker nodes (one per allocated host).
            Must equal the number of hosts in the LSF allocation.
        gpus_per_worker: GPUs per host (each host registers this many GPUs
            with Ray).
        conda_env: conda env path/name workers must activate before
            ``ray start``.
        cores_per_worker: cores reserved by each Ray worker (default 32).
            Note: on the head host, the Ray head process itself competes
            for cores with the colocated worker; size accordingly.
        fleet: name of the deployment fleet (e.g. ``"a100"``,
            ``"h100"``). Selects the default ``NCCL_IB_HCA`` rail list via
            ``_default_ib_hca(fleet)`` and the GPU model string (logged only)
            via ``_gpu_model_for_fleet(fleet)``. Default ``DEFAULT_FLEET``.
        ib_hca: explicit override for ``NCCL_IB_HCA`` (e.g. ``"mlx5_0,mlx5_1"``).
            ``None`` means "use the fleet default."
        ib_ifname: optional override for ``NCCL_SOCKET_IFNAME``. See
            ``_rdma_env``.
        wait_timeout_s: cap on the wait-for-workers phase (clamped by the
            overall ``bringup_deadline_s`` budget).
        log_dir: parent dir for per-worker log files. **Required** —
            ``main.py`` passes ``f'{output_dir}/logs'`` so the logs are
            co-located with the rest of the run's artifacts and get
            wiped by ``--cleanup``. Final shape:
            ``<log_dir>/<job_group>/<host>.log``.
        bringup_deadline_s: hard cap on the entire bring-up.
        rl_algo: name of the RL algorithm. Only logged here — GPU mode and
            host-exclusivity are decided in the outer ``bsub``, since under
            single-allocation we cannot vary them per worker.

    Returns:
        Handle dict for ``stop_multinode_ray_cluster``.

    Raises:
        RayUpTimeoutError: bring-up did not complete within
            ``bringup_deadline_s`` (partial teardown already attempted).
        RuntimeError: if not running inside a multi-host LSF allocation,
            or if the host count doesn't match ``num_workers``, or if any
            worker process exits during bring-up.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1 for blaunch path")
    if not log_dir:
        # Required so worker logs land under the user-specified output_dir
        # (where ``cleanup()`` knows to wipe them). main.py threads this
        # through; future call sites must do the same.
        raise ValueError("log_dir is required (e.g. f'{output_dir}/logs'); see main.py for the expected wiring")

    # Resolve fleet → default rail list, unless the caller passed an explicit
    # override. ``_default_ib_hca`` warns on unknown fleet names.
    resolved_ib_hca = ib_hca if ib_hca else _default_ib_hca(fleet)
    resolved_gpu_model = _gpu_model_for_fleet(fleet)

    banner("ray_up_blaunch: starting single-allocation cluster")
    dump_env_fingerprint()
    log_dir_abs = os.path.abspath(log_dir)
    logger.info(
        f"[ray_up_blaunch] kwargs: num_workers={num_workers} gpus_per_worker={gpus_per_worker} "
        f"cores_per_worker={cores_per_worker} gpu_model={resolved_gpu_model!r} "
        f"fleet={fleet!r} ib_hca={resolved_ib_hca!r} ib_ifname={ib_ifname!r} "
        f"wait_timeout_s={wait_timeout_s} log_dir={log_dir_abs!r} conda_env={conda_env!r} "
        f"rl_algo={rl_algo!r}"
    )
    logger.info(f"[ray_up_blaunch] bring-up deadline: {bringup_deadline_s}s")

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

    # Resolve LSF host list and partition into head + workers up-front so we
    # fail fast before starting Ray if the allocation shape is wrong.
    host_list = _read_lsf_hostfile()
    head_hostname = socket.gethostname()
    head_host, all_worker_hosts, remote_worker_hosts = _partition_hosts(
        host_list, head_hostname, num_workers=num_workers
    )
    logger.info(
        f"[ray_up_blaunch] head_host={head_host!r} all_workers={all_worker_hosts} remote_workers={remote_worker_hosts}"
    )

    job_id = os.environ.get("LSB_JOBID", "0")
    head_temp_dir = f"/tmp/_ray/job_{job_id}"
    os.makedirs(head_temp_dir, exist_ok=True)

    partial_info: Dict[str, Any] = {
        "head_address": None,
        "head_ip": None,
        "head_port": None,
        "dashboard_port": None,
        "temp_dir": head_temp_dir,
        "num_gpus": 0,
        "worker_job_ids": [],
        "worker_hosts": list(all_worker_hosts),
        "remote_worker_hosts": list(remote_worker_hosts),
        "worker_pids": [],
        "job_group": None,
        "rdma_env": rdma_env,
        # Stashed so ray_down can fan out a remote `ray stop` via blaunch
        # to drain the LSF cgroup on remote hosts (otherwise LSF won't
        # reap the job — see _remote_ray_stop_via_blaunch).
        "conda_env": conda_env,
    }

    popen_records: List[Tuple[str, subprocess.Popen, str]] = []

    try:
        check_deadline("start_head")
        with phase_timer("start_head"):
            head_ip, head_port, dash_port = _start_head(head_temp_dir)
        head_address = f"{head_ip}:{head_port}"
        partial_info.update(head_address=head_address, head_ip=head_ip, head_port=head_port, dashboard_port=dash_port)
        logger.info(
            f"[ray_up_blaunch] head_address={head_address} "
            f"dashboard=http://{head_ip}:{dash_port} temp_dir={head_temp_dir}"
        )

        job_group = f"ray_nodes_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        partial_info["job_group"] = job_group
        run_log_dir = os.path.join(log_dir_abs, job_group)
        os.makedirs(run_log_dir, exist_ok=True)
        logger.info(f"[ray_up_blaunch] job_group={job_group} run_log_dir={run_log_dir}")

        # Connect the driver to its own head before spawning workers so we
        # have a Ray client ready to poll cluster_resources during bring-up.
        check_deadline("ray_init")
        with phase_timer("ray_init_driver"):
            ok = _ray_init_with_timeout(address=head_address, timeout_s=remaining())
        if not ok:
            raise TimeoutError(f"ray.init({head_address!r}) did not return within remaining budget")

        # Local worker on the head host (no blaunch — we are already here).
        check_deadline("spawn_local_worker")
        local_record = _spawn_local_worker(
            head_host=head_host,
            head_address=head_address,
            gpus_per_worker=gpus_per_worker,
            cores_per_worker=cores_per_worker,
            conda_env=conda_env,
            rdma_env=rdma_env,
            run_log_dir=run_log_dir,
        )
        popen_records.append(local_record)
        partial_info["worker_pids"].append(local_record[1].pid)

        # Remote workers via blaunch.
        check_deadline("spawn_remote_workers")
        remote_records = _spawn_remote_workers(
            remote_host_list=remote_worker_hosts,
            head_address=head_address,
            gpus_per_worker=gpus_per_worker,
            cores_per_worker=cores_per_worker,
            conda_env=conda_env,
            rdma_env=rdma_env,
            run_log_dir=run_log_dir,
        )
        popen_records.extend(remote_records)
        for _, proc, _ in remote_records:
            partial_info["worker_pids"].append(proc.pid)

        logger.info(
            f"[ray_up_blaunch] spawned {len(popen_records)} worker(s) "
            f"(1 local + {len(remote_records)} remote): pids={partial_info['worker_pids']}"
        )

        expected_total = num_workers * gpus_per_worker
        # +1 for the head node itself (CPU-only Ray node).
        expected_nodes = num_workers + 1
        phase_budget = int(min(wait_timeout_s, remaining()))
        if phase_budget <= 0:
            raise TimeoutError("no budget left for _wait_for_workers_blaunch")
        _wait_for_workers_blaunch(
            expected_total_gpus=expected_total,
            expected_nodes=expected_nodes,
            popen_records=popen_records,
            timeout_s=phase_budget,
        )

        final_resources = ray.cluster_resources()
        summarize_ray_nodes()
        banner("ray_up_blaunch: cluster ready")
        elapsed = time.monotonic() - t0
        logger.info(
            f"[ray_up_blaunch] head={head_address} dashboard=http://{head_ip}:{dash_port} "
            f"GPUs={int(final_resources.get('GPU', 0))} "
            f"workers={all_worker_hosts} job_group={job_group}"
        )
        logger.info(f"[ray_up_blaunch] bring-up complete in {elapsed:.1f}s of {bringup_deadline_s}s budget")

        return {
            "head_address": head_address,
            "head_ip": head_ip,
            "head_port": head_port,
            "dashboard_port": dash_port,
            "temp_dir": head_temp_dir,
            "num_gpus": int(final_resources.get("GPU", 0)),
            "worker_job_ids": [],
            "worker_hosts": list(all_worker_hosts),
            "remote_worker_hosts": list(remote_worker_hosts),
            "worker_pids": [proc.pid for _, proc, _ in popen_records],
            "job_group": job_group,
            "rdma_env": rdma_env,
            "conda_env": conda_env,
        }

    except TimeoutError as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            f"[ray_up_blaunch] bring-up timed out after {elapsed:.1f}s (deadline={bringup_deadline_s}s): {exc}"
        )
        try:
            from autotune.lsf.ray_down import stop_multinode_ray_cluster

            stop_multinode_ray_cluster(partial_info)
        except Exception as td_exc:  # noqa: BLE001 — never mask the timeout
            logger.warning(f"[ray_up_blaunch] partial teardown raised (ignored): {td_exc}")
        raise RayUpTimeoutError(str(exc), cluster_info=partial_info) from exc
