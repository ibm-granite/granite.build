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

import logging
import math
import os
import socket
import subprocess
from typing import Dict, List, Optional, Tuple

import ray

from autotune.device import detect_accelerator, object_store_bytes
from autotune.logging_setup import bridge_runtime_env

logger = logging.getLogger(__name__)


def resolve_local_ray_gpus() -> int:
    """GPU count to advertise to a local Ray cluster.

    On CUDA this is the pre-MPS probe verbatim: honour CUDA_VISIBLE_DEVICES,
    else torch.cuda.device_count(), else parse nvidia-smi. On any non-CUDA
    accelerator, Ray cannot schedule the device as a resource, so return 0.
    """
    accel = detect_accelerator()
    if accel.kind != "cuda":
        return 0

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        return len([d for d in cuda_visible.split(",") if d])
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            return len(result.stdout.strip().splitlines())
        except Exception:
            return 0


def configure_ray_data_context() -> None:
    """Enable Ray Data's rich progress UI and disable the old ray_tqdm bar.

    Must be called after ray.init(). Safe to call multiple times. Silently
    no-ops on Ray versions that don't expose these flags.
    """
    try:
        from ray.data import DataContext

        ctx = DataContext.get_current()
        if hasattr(ctx, "enable_rich_progress_bars"):
            ctx.enable_rich_progress_bars = True
        if hasattr(ctx, "use_ray_tqdm"):
            ctx.use_ray_tqdm = False
    except Exception as e:
        logger.debug(f"Could not configure Ray Data context: {e}")


def compute_ray_data_sizing(
    num_workers: int,
    concurrency_override: Optional[int],
    num_cpus_override: Optional[float],
) -> Tuple[int, float]:
    """Compute the (concurrency, num_cpus_per_task) budget for Ray Data tokenization.

    Ray Data's ``map_batches`` fans tokenization out across the cluster, but it
    competes with the GPU training workers (each reserves ``{"CPU": 1}`` via
    ScalingConfig) for CPU slots. By default we let tokenization use every CPU
    not reserved by this trial's workers: ``floor(total_cluster_cpus) - num_workers``.

    ``concurrency_override`` / ``num_cpus_override`` (from config or CLI) take
    precedence when set. The result is always clamped to at least 1.

    Note for concurrent HPO: each trial computes this independently from the
    full cluster CPU count, unaware of sibling trials. ``max_concurrent_trials``
    bounds the trial count; for large sweeps prefer an explicit override.
    """
    try:
        total_cpus = int(math.floor(ray.cluster_resources().get("CPU", 0)))
    except Exception:
        total_cpus = 0

    if concurrency_override is not None:
        concurrency = max(1, int(concurrency_override))
    else:
        concurrency = max(1, total_cpus - int(num_workers))

    num_cpus = float(num_cpus_override) if num_cpus_override is not None else 1.0
    return concurrency, num_cpus


def ray_data_block_target(concurrency: int, row_count: int) -> int:
    """Block count for repartition: one block per concurrent task, clamped to rows.

    Ray Data launches at most one stateless map task per input block, so the
    dataset must be split into >= ``concurrency`` blocks to use that many CPUs.
    Clamped to ``row_count`` so tiny datasets (e.g. small eval splits) are not
    over-partitioned into mostly-empty blocks. Never returns less than 1.
    """
    return max(1, min(int(concurrency), int(row_count)))


def reserve_ports(n: int) -> Tuple[List[int], List[socket.socket]]:
    """Reserve *n* free ports atomically by holding sockets open.

    Sockets are bound to ``0.0.0.0:0`` (all interfaces) without
    ``SO_REUSEADDR`` — both choices matter for the multi-node Ray case:

    * Ray's GCS / dashboard binds to ``0.0.0.0`` so worker nodes can reach
      the head.  A reservation on ``127.0.0.1`` does **not** stop another
      process from binding the same port on ``0.0.0.0`` — the kernel treats
      those as distinct addresses.  Binding to ``0.0.0.0`` reserves the port
      across all interfaces.
    * ``SO_REUSEADDR`` is **not** set, so the reservation is exclusive while
      held.  Caller must close the socket immediately before ``ray start``.

    Caller must release the sockets via :func:`release_sockets` *just before*
    the subprocess spawn that will rebind them, to keep the TOCTOU window
    minimal.
    """
    sockets: List[socket.socket] = []
    ports: List[int] = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", 0))
        ports.append(s.getsockname()[1])
        sockets.append(s)
    return ports, sockets


def release_sockets(sockets: List[socket.socket]) -> None:
    """Close all held sockets so that the ports become available."""
    for s in sockets:
        try:
            s.close()
        except OSError:
            pass


def stop_ray_cluster(temp_dir: str) -> None:
    """Stop only the Ray cluster whose processes reference *temp_dir*.

    On a multi-tenant node a bare ``ray stop`` would kill *every* Ray
    cluster.  This function uses ``psutil`` (already an optional dependency)
    to selectively terminate only our processes.  If ``psutil`` is not
    available it falls back to ``ray stop`` with a warning.
    """
    try:
        import psutil
    except ImportError:
        logger.warning(
            "psutil not available — falling back to 'ray stop' which may affect other Ray clusters on this node."
        )
        subprocess.run(
            ["ray", "stop"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return

    matched = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            # Match any Ray process that was started with our temp dir.
            if temp_dir in cmdline_str and (
                "ray" in cmdline_str or "raylet" in cmdline_str or "gcs_server" in cmdline_str
            ):
                matched.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not matched:
        logger.info(f"No Ray processes found for temp_dir={temp_dir}")
        return

    logger.info(f"Sending SIGTERM to {len(matched)} Ray process(es) for temp_dir={temp_dir}")
    for proc in matched:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Wait briefly for graceful shutdown, then SIGKILL anything still alive.
    gone, alive = psutil.wait_procs(matched, timeout=5)
    if alive:
        logger.warning(f"{len(alive)} Ray process(es) survived SIGTERM; sending SIGKILL: {[p.pid for p in alive]}")
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        psutil.wait_procs(alive, timeout=3)
    logger.info(f"Terminated {len(matched)} Ray process(es) for temp_dir={temp_dir}")


def start_local_ray_cluster() -> Dict:
    """Start a job-isolated local Ray cluster.

    Uses LSB_JOBID to create a unique temp directory under /tmp/ray/ so that
    multiple LSF jobs on the same node do not collide.  Reserves two ports
    atomically for Ray's node-manager and object-manager via ``reserve_ports``.

    Returns a dict with ``temp_dir`` (needed by ``stop_local_ray_cluster``)
    and ``num_gpus``.
    """
    # -- Job-specific temp directory ------------------------------------------
    job_id = os.environ.get("LSB_JOBID", "0")
    temp_dir = f"/tmp/_ray/job_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Ray temp dir: {temp_dir}")

    # -- Accelerator + GPU detection ------------------------------------------
    accel = detect_accelerator()
    num_gpus = resolve_local_ray_gpus()
    logger.info(f"Detected accelerator={accel.kind}, {num_gpus} Ray GPU(s)")

    # -- Object store sizing --------------------------------------------------
    # CUDA clusters use the 0.5 RAM proportion (ray.data tokenization path).
    # The single-device (MPS/CPU) driver never touches ray.data, so cap the
    # object store at a small fixed size to leave RAM for Metal.
    object_store_memory = None
    if accel.kind == "cuda":
        if "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION" not in os.environ:
            os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.5"
    else:
        object_store_memory = object_store_bytes()

    # -- Start Ray ------------------------------------------------------------
    ray.init(
        address=None,
        num_gpus=num_gpus,
        object_store_memory=object_store_memory,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=temp_dir,
        log_to_driver=True,
        logging_level=logging.INFO,
        **bridge_runtime_env(),
    )
    logger.info(f"Local Ray cluster started (temp_dir={temp_dir})")

    configure_ray_data_context()

    return {"temp_dir": temp_dir, "num_gpus": num_gpus}


def stop_local_ray_cluster(temp_dir: str) -> None:
    """Gracefully shut down the local Ray cluster for *temp_dir*.

    Calls ``ray.shutdown()`` first, then delegates to ``stop_ray_cluster``
    to kill any lingering Ray processes scoped to the temp dir.
    """
    ray.shutdown()
    stop_ray_cluster(temp_dir)


def ensure_gpu_isolation(requested_gpus: int):
    """
    Make sure this process only sees `requested_gpus` devices.
    If LSF already set CUDA_VISIBLE_DEVICES, we trust it (but warn if > requested_gpus).
    Otherwise, default to the first `requested_gpus` devices.
    """
    already_set = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    if already_set:
        # Count how many devices are visible.
        visible = [d for d in already_set.split(",") if d != ""]
        if len(visible) > requested_gpus:
            logger.warning(
                f"CUDA_VISIBLE_DEVICES is already set to {already_set} "
                f"({len(visible)} GPUs) which exceeds requested {requested_gpus}. "
                "Proceeding but Ray will only schedule up to requested GPUs."
            )
        else:
            logger.info(f"Respecting existing CUDA_VISIBLE_DEVICES={already_set}")
        return

    # Otherwise, pick first `requested_gpus` devices on the machine.
    # NOTE: We assume devices are 0..7; adjust if your node uses a different scheme.
    devices = ",".join(str(i) for i in range(requested_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    logger.info(f"Set CUDA_VISIBLE_DEVICES={devices}")
