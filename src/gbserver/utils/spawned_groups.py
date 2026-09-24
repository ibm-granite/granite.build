#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Process-wide registry of workload session leaders, for last-resort reaping.

Workloads are launched with ``start_new_session=True``, so each becomes its own
session and process-group leader and is therefore NOT reached by signalling the
server's own group. The owning environment reaps its group on the normal paths
(``Bash.cleanup_nohup``), keyed by ``launch_id`` on a per-instance dict.

That registry is unreachable once the owning thread is the problem: if a build
thread wedges, shutdown abandons the join (bounded by
``_SHUTDOWN_JOIN_TIMEOUT_S``) and the workload's process group is left running.
This module is the process-wide fallback so shutdown can still make a real
attempt, without needing the launch_id, the environment instance, or the event
loop the launch happened on.

Registration is best-effort bookkeeping, deliberately kept dependency-free
(stdlib only, no psutil) and safe to call from any thread.
"""

import os
import signal
import threading
import time
from typing import Dict, List, Optional, Tuple

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# pid -> (pgid, label). The pid is the session leader we spawned; pgid is captured
# at registration because it cannot be read back once the process is gone.
_groups: Dict[int, Tuple[int, str]] = {}
_lock = threading.Lock()


def register(pid: int, label: str = "") -> None:
    """Record a spawned session leader so shutdown can reap it as a last resort."""
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError) as exc:
        # Already gone, or no permission: nothing useful to remember.
        logger.debug("spawned_groups.register: pid %s not readable: %s", pid, exc)
        return
    with _lock:
        _groups[pid] = (pgid, label)
    logger.debug("spawned_groups: registered pid %s (pgid %s) %s", pid, pgid, label)


def unregister(pid: int) -> None:
    """Forget a workload the owning environment reaped on its normal path."""
    with _lock:
        _groups.pop(pid, None)


def _collect_zombie(pid: int) -> bool:
    """Reap ``pid`` if it is our terminated child. True once it has been collected.

    A signalled child stays addressable as a zombie until its parent waits on it, so
    a ``killpg`` probe alone reports it as alive forever and ``reap_all`` would
    escalate to SIGKILL for a process that already died. The workload is spawned in
    this process, so we are usually that parent; when we are not, ``waitpid`` raises
    ``ChildProcessError`` and the probe falls back to signalling.
    """
    try:
        reaped, _status = os.waitpid(pid, os.WNOHANG)
        return reaped == pid
    except ChildProcessError:
        # Not our child (or already collected) — nothing to do here.
        return False
    except OSError:
        return False


def _alive(pgid: int, pid: int) -> bool:
    """True while any member of the group is still running.

    Collects ``pid`` first if it is our zombie child, so a process that has already
    terminated is not mistaken for a survivor (see _collect_zombie).
    """
    _collect_zombie(pid)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but not ours to signal — treat as alive; we cannot reap it.
        return True
    except OSError:
        return False
    # The group still answers. If the leader is a zombie we could not collect (not
    # our child) the group is effectively finished, but we cannot distinguish that
    # from a live member, so report alive and let the caller give up on it.
    return True


def reap_all(grace_seconds: float = 5.0) -> List[int]:
    """SIGTERM every registered group, then SIGKILL whatever survives the grace.

    A best-effort last resort for shutdown paths that cannot rely on the owning
    thread: it never raises, and it gives up rather than blocking, so a group we
    are not allowed to signal cannot wedge shutdown.

    Only groups we actually spawned are signalled, and only when the pid is still
    that group's leader (``pgid == pid``). A recycled pid whose group no longer
    matches is skipped, so this cannot kill an unrelated process that happens to
    have inherited the number.

    Args:
        grace_seconds: How long to wait after SIGTERM before escalating.

    Returns:
        The pids still alive after the SIGKILL attempt (empty when all were reaped).
    """
    with _lock:
        entries = list(_groups.items())
    if not entries:
        return []

    targets: List[Tuple[int, int, str]] = []
    for pid, (pgid, label) in entries:
        try:
            # Re-read the group: if this pid is no longer its own leader, the pid
            # was recycled (or setsid never took) and signalling the recorded pgid
            # could hit something unrelated.
            current = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            with _lock:
                _groups.pop(pid, None)
            continue
        if current != pgid or current != pid:
            logger.debug(
                "spawned_groups: skipping pid %s (pgid now %s, recorded %s)",
                pid,
                current,
                pgid,
            )
            with _lock:
                _groups.pop(pid, None)
            continue
        targets.append((pid, pgid, label))

    if not targets:
        return []

    logger.warning(
        "reaping %d leaked workload process group(s) during shutdown: %s",
        len(targets),
        ", ".join(f"pid={p} {lbl}".strip() for p, _, lbl in targets),
    )
    for _pid, pgid, _label in targets:
        _signal(pgid, signal.SIGTERM)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    remaining = [(p, g, l) for p, g, l in targets if _alive(g, p)]
    while remaining and time.monotonic() < deadline:
        time.sleep(0.2)
        remaining = [(p, g, l) for p, g, l in remaining if _alive(g, p)]

    for _pid, pgid, _label in remaining:
        _signal(pgid, signal.SIGKILL)
    # One short settle so SIGKILL is reflected before we report what survived.
    if remaining:
        time.sleep(0.2)

    survivors = []
    for pid, pgid, label in targets:
        if _alive(pgid, pid):
            survivors.append(pid)
            logger.error(
                "workload pid %s (pgid %s) %s survived SIGKILL; giving up",
                pid,
                pgid,
                label,
            )
        with _lock:
            _groups.pop(pid, None)
    return survivors


def _signal(pgid: int, sig: int) -> None:
    """Signal a process group, tolerating a group that is already gone."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError) as exc:
        logger.debug("spawned_groups: killpg(%s, %s) failed: %s", pgid, sig, exc)


def tracked_pids() -> List[int]:
    """Currently-registered pids (for tests and diagnostics)."""
    with _lock:
        return sorted(_groups)
