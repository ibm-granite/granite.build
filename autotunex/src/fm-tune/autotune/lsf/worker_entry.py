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
"""Ray worker entry point launched by each LSF worker job.

Run as: ``python -m autotune.lsf.worker_entry --head_address <ip:port> ...``

The worker process must outlive ``ray start`` so the LSF job stays alive and
Ray keeps the node attached.  We block on ``signal.pause()`` after starting
Ray and shut it down cleanly on ``SIGTERM`` (which ``bkill`` sends).
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time

from autotune.cluster import release_sockets, reserve_ports
from autotune.logging_setup import setup_logging
from autotune.lsf.log_utils import banner, dump_env_fingerprint, log_argv

setup_logging()
logger = logging.getLogger(__name__)


def _ray_stop() -> None:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(["ray", "stop"], check=False, capture_output=True, text=True)
        dt = time.monotonic() - t0
        logger.info(
            f"ray stop done in {dt:.1f}s rc={proc.returncode} "
            f"stdout={proc.stdout.strip()!r} stderr={proc.stderr.strip()!r}"
        )
    except Exception as e:
        dt = time.monotonic() - t0
        logger.warning(f"ray stop failed after {dt:.1f}s: {e}")


def _on_sigterm(signum, frame):
    name = signal.Signals(signum).name if isinstance(signum, int) else str(signum)
    logger.info(f"Received {name} ({signum}); stopping ray worker.")
    _ray_stop()
    sys.exit(0)


def _pre_flight() -> None:
    """Best-effort host diagnostics so a stuck worker is debuggable from its log."""
    for label, cmd in (
        ("nvidia-smi -L", ["nvidia-smi", "-L"]),
        ("ibstat", ["ibstat"]),
        ("ip -br link show", ["ip", "-br", "link", "show"]),
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if proc.returncode == 0:
                logger.info(f"[preflight] {label}:\n{proc.stdout}")
            else:
                logger.warning(f"[preflight] {label} rc={proc.returncode} stderr={proc.stderr.strip()!r}")
        except FileNotFoundError:
            logger.warning(f"[preflight] {cmd[0]!r} not found on PATH")
        except Exception as e:
            logger.warning(f"[preflight] {label} raised: {e}")


def _visible_gpu_count() -> int | None:
    """Return the number of GPUs this process can actually use, or None if unknown.

    Ray's ``--num-gpus`` is a *declaration*, not a probe: ``ray start`` advertises
    whatever count it is given without checking it against the host. If that count
    exceeds the GPUs CUDA can enumerate, the mismatch surfaces much later and far
    less legibly — Ray Train hands the highest-ranked worker an out-of-range
    ordinal and ``torch.cuda.set_device(local_rank)`` aborts with
    ``device >= 0 && device < num_gpus INTERNAL ASSERT FAILED ... device=7, num_gpus=7``
    during NCCL backend setup, before training starts. We probe up front instead.

    Honours ``CUDA_VISIBLE_DEVICES`` (the count Ray's workers will inherit). Tries
    the authoritative source first — the CUDA runtime via torch, which is exactly
    what ``set_device`` checks — then falls back to ``nvidia-smi -L``. Returns
    ``None`` only when neither probe works, so the caller can warn-and-proceed
    rather than block a launch on a flaky probe.
    """
    # Prefer CUDA_VISIBLE_DEVICES if the scheduler/caller pinned a device set:
    # that masks the real count and is what the worker processes will see.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        # Empty string => zero visible GPUs; otherwise count non-empty entries.
        entries = [d for d in cvd.split(",") if d.strip() != ""]
        return len(entries)

    # No mask: ask the CUDA runtime directly (same enumeration set_device uses).
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception as e:
        logger.warning(f"[gpu-guard] torch device probe failed: {e}")

    # Fallback: parse `nvidia-smi -L` (one line per GPU).
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            return sum(1 for line in proc.stdout.splitlines() if line.strip().startswith("GPU "))
        logger.warning(f"[gpu-guard] nvidia-smi -L rc={proc.returncode} stderr={proc.stderr.strip()!r}")
    except FileNotFoundError:
        logger.warning("[gpu-guard] nvidia-smi not found on PATH")
    except Exception as e:
        logger.warning(f"[gpu-guard] nvidia-smi probe raised: {e}")

    return None


def _assert_gpu_count(requested: int) -> None:
    """Fail fast if this host can't supply the ``--num-gpus`` Ray is about to claim.

    Raises ``RuntimeError`` when the probe succeeds and finds fewer GPUs than
    requested — naming the host and both counts so the operator knows exactly
    which node is short and by how much. A probe that returns ``None`` (unknown)
    or a count >= requested is allowed through; over-provisioning is harmless,
    only under-provisioning crashes the run later.
    """
    visible = _visible_gpu_count()
    host = socket.gethostname()
    if visible is None:
        logger.warning(
            f"[gpu-guard] could not determine visible GPU count on {host}; "
            f"proceeding with requested --num-gpus={requested} unchecked."
        )
        return
    if visible < requested:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        raise RuntimeError(
            f"[gpu-guard] host {host} exposes {visible} usable GPU(s) but Ray was "
            f"asked for --num-gpus={requested}. The 8th/highest-ranked worker would "
            f"later abort in torch.cuda.set_device(local_rank) during NCCL setup "
            f"('device >= 0 && device < num_gpus'). Likely causes: a degraded/"
            f"off-the-bus GPU on this node, a CUDA_VISIBLE_DEVICES mask "
            f"(={cvd!r}), or an LSF allocation with fewer GPUs than --gpus_per_node. "
            f"Check `nvidia-smi -L` on {host}, or reduce --gpus_per_node to {visible}."
        )
    logger.info(f"[gpu-guard] {host}: {visible} GPU(s) visible >= requested {requested}; OK.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_address", required=True, help="Ray head address, e.g. 10.0.0.1:6379")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_cpus", type=int, required=True)
    parser.add_argument(
        "--obj_store_mem",
        type=int,
        default=30_000_000_000,
        help="Object store memory in bytes (default 30 GB).",
    )
    args = parser.parse_args()

    banner("worker_entry: starting")
    dump_env_fingerprint()
    logger.info(
        f"[worker] args: head_address={args.head_address!r} num_gpus={args.num_gpus} "
        f"num_cpus={args.num_cpus} obj_store_mem={args.obj_store_mem}"
    )

    # Note: NCCL/RDMA env vars (NCCL_IB_HCA etc.) reach this process via the
    # ``env K=V ... python -m autotune.lsf.worker_entry`` prefix that
    # ``ray_up_blaunch._build_inner_cmd`` constructs from ``_rdma_env(...)``.
    # We do NOT set defaults here — that
    # would bake in stale values (e.g. ``mlx5_0`` only) and silently override
    # multi-rail fleet defaults from the caller.
    _pre_flight()

    # Fail fast (before ray start) if this host can't back the requested GPU
    # count — turns a cryptic mid-NCCL-setup set_device abort into a clear,
    # node-named error. See _assert_gpu_count for the full rationale.
    try:
        _assert_gpu_count(args.num_gpus)
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    # Per-worker temp dir.  Keyed on (LSB_JOBID, hostname, pid) so that:
    #   * In bsub-per-worker mode, every worker has its own LSB_JOBID anyway —
    #     the host+pid suffix is just extra disambiguation (harmless).
    #   * In blaunch mode, every worker child inherits the SAME outer LSB_JOBID,
    #     so we MUST add hostname+pid; otherwise the Ray head's worker on
    #     host[0] and any other worker that happens to share a tmpfs view
    #     would collide on the plasma-store Unix-socket path.
    worker_lsb_jobid = os.environ.get("LSB_JOBID") or str(os.getpid())
    host = socket.gethostname()
    worker_temp_dir = f"/tmp/ray/job_{worker_lsb_jobid}_{host}_{os.getpid()}"
    os.makedirs(worker_temp_dir, exist_ok=True)
    logger.info(f"[worker] temp_dir={worker_temp_dir}")

    # Reserve five ports: node-manager, object-manager, dashboard-agent grpc,
    # dashboard-agent listen, metrics-export.  Sockets are held open until just
    # before ``ray start`` so the bind window is as small as possible.
    ports, sockets = reserve_ports(5)
    nmp, omp, dagp, dalp, mep = ports

    logger.info(
        f"Starting ray worker: head={args.head_address} num_gpus={args.num_gpus} "
        f"num_cpus={args.num_cpus} ports(nmp,omp,dagp,dalp,mep)={ports}"
    )

    cmd = [
        "ray",
        "start",
        "--address",
        args.head_address,
        "--num-gpus",
        str(args.num_gpus),
        "--num-cpus",
        str(args.num_cpus),
        "--object-store-memory",
        str(args.obj_store_mem),
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
        "--temp-dir",
        worker_temp_dir,
        "--disable-usage-stats",
    ]
    log_argv("ray-start-worker", cmd)
    # Release the held sockets immediately before the spawn (see ray_up_blaunch._start_head).
    release_sockets(sockets)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        logger.error(f"ray start failed in {dt:.1f}s (rc={proc.returncode})")
        logger.error(f"stdout:\n{proc.stdout}")
        logger.error(f"stderr:\n{proc.stderr}")
        return proc.returncode
    logger.info(f"ray start succeeded in {dt:.1f}s:\n{proc.stdout}")

    # One-shot post-attach confirmation: `ray status` against the head we joined.
    try:
        st = subprocess.run(
            ["ray", "status", "--address", args.head_address],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if st.returncode == 0:
            logger.info(f"ray status:\n{st.stdout}")
        else:
            logger.warning(f"ray status rc={st.returncode} stderr={st.stderr.strip()!r}")
    except Exception as e:
        logger.warning(f"ray status failed: {e}")

    # Block forever; LSF will SIGTERM us via bkill on teardown.
    logger.info("Worker attached. Blocking until SIGTERM/SIGINT.")
    signal.pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
