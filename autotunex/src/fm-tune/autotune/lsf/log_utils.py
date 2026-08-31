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
"""Debug-logging helpers for the multi-node Ray bring-up / teardown flow.

These helpers exist so that a single failed run produces a self-contained
diagnostic trail in the head log without the user having to SSH into worker
hosts or manually hunt for ``bjobs -l`` output.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import socket
import subprocess
import time
from typing import Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Keys that ``_rdma_env`` populates — used by ``dump_env_fingerprint`` to echo
# whichever values are actually live in ``os.environ``.
_RDMA_ENV_KEYS = (
    "NCCL_NET_GDR_LEVEL",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_CUDA_SUPPORT",
    "NCCL_IB_GDR_LEVEL",
    "NCCL_IB_DISABLE",
    "NCCL_DEBUG",
    "NCCL_IB_QPS_PER_CONNECTION",
    "NCCL_IB_SPLIT_DATA_ON_QPS",
    "NCCL_BUFFSIZE",
    "NCCL_MIN_NCHANNELS",
)


def banner(title: str, char: str = "=", width: int = 78) -> None:
    """Emit a 3-line banner so phase boundaries are easy to grep."""
    line = char * width
    logger.info(line)
    logger.info(f"{char} {title}")
    logger.info(line)


@contextlib.contextmanager
def phase_timer(name: str) -> Iterator[None]:
    """Log ``[start]`` / ``[done in X.Ys]`` / ``[failed in X.Ys: <exc>]`` around a block."""
    logger.info(f"[phase:{name}] start")
    t0 = time.monotonic()
    try:
        yield
    except BaseException as e:
        dt = time.monotonic() - t0
        logger.error(f"[phase:{name}] failed in {dt:.1f}s: {type(e).__name__}: {e}")
        raise
    else:
        dt = time.monotonic() - t0
        logger.info(f"[phase:{name}] done in {dt:.1f}s")


def _run_capture(cmd: List[str], timeout: float = 15.0) -> Optional[str]:
    """Best-effort subprocess capture; returns stdout or ``None`` on any failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        logger.debug(f"_run_capture: {cmd[0]!r} not found on PATH")
        return None
    except Exception as e:
        logger.debug(f"_run_capture {cmd!r} raised: {e}")
        return None
    if proc.returncode != 0:
        logger.debug(f"_run_capture {cmd!r} rc={proc.returncode} stderr={proc.stderr.strip()!r}")
        return None
    return proc.stdout


def dump_env_fingerprint() -> None:
    """Log a one-time snapshot of who/where/what we're running as.

    Always best-effort: any individual lookup that fails is logged at DEBUG and
    skipped, never raises.
    """
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "<unknown>"
    try:
        ip = socket.gethostbyname(hostname) if hostname != "<unknown>" else "<unknown>"
    except Exception:
        ip = "<unknown>"

    logger.info(f"[env] hostname={hostname} ip={ip}")
    logger.info(
        f"[env] LSB_JOBID={os.environ.get('LSB_JOBID', '')!r} "
        f"USER={os.environ.get('USER', '')!r} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
    )

    ray_path = shutil.which("ray") or "<not found>"
    python_path = shutil.which("python") or "<not found>"
    logger.info(f"[env] which ray={ray_path}  which python={python_path}")

    py_ver = _run_capture(["python", "--version"], timeout=5.0)
    ray_ver = _run_capture(["ray", "--version"], timeout=5.0)
    if py_ver:
        logger.info(f"[env] python: {py_ver.strip()}")
    if ray_ver:
        logger.info(f"[env] ray:    {ray_ver.strip()}")

    rdma_present = {k: os.environ[k] for k in _RDMA_ENV_KEYS if k in os.environ}
    logger.info(f"[env] RDMA env in os.environ: {rdma_present}")


def summarize_ray_nodes() -> None:
    """Log one line per node currently registered with Ray (best-effort)."""
    try:
        import ray  # noqa: WPS433 — local import keeps this helper safe to import early
    except Exception as e:
        logger.warning(f"[ray] cannot import ray: {e}")
        return

    try:
        nodes = ray.nodes()
    except Exception as e:
        logger.warning(f"[ray] ray.nodes() failed: {e}")
        return

    if not nodes:
        logger.info("[ray] ray.nodes() returned no nodes")
        return

    logger.info(f"[ray] ray.nodes(): {len(nodes)} node(s)")
    for n in nodes:
        # ray.nodes() entries vary slightly across Ray versions; pull defensively.
        node_id = n.get("NodeID", "?")[:8]
        addr = n.get("NodeManagerAddress", "?")
        host = n.get("NodeManagerHostname", "?")
        alive = n.get("Alive", False)
        res = n.get("Resources", {}) or {}
        gpu = res.get("GPU", 0)
        cpu = res.get("CPU", 0)
        mem_b = res.get("memory", 0)
        mem_gb = (mem_b / (1024**3)) if isinstance(mem_b, (int, float)) else 0.0
        logger.info(
            f"[ray]   node={node_id}.. host={host} addr={addr} alive={alive} GPU={gpu} CPU={cpu} memory={mem_gb:.1f}GB"
        )


def log_argv(label: str, argv: Iterable[str]) -> None:
    """Log a subprocess argv on a single line, shell-quoted enough to copy/paste."""
    parts = []
    for a in argv:
        s = str(a)
        if any(c.isspace() for c in s) or "'" in s or '"' in s:
            esc = s.replace("'", "'\\''")
            parts.append(f"'{esc}'")
        else:
            parts.append(s)
    logger.info(f"[argv:{label}] {' '.join(parts)}")
