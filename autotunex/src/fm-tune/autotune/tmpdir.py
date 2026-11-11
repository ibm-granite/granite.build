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

"""Scheduler-agnostic Ray temp-directory resolution.

Ray keeps its session state — crucially the plasma store's Unix domain socket —
under a temp directory. Picking that directory badly breaks jobs in ways that are
tedious to diagnose, so all of fm-tune routes through here.

**Why not a shared ``/tmp/_ray`` parent.** A path like ``/tmp/_ray/job_<id>`` has a
*shared intermediate directory*: whichever user first creates ``/tmp/_ray`` owns it,
at their umask. Observed in the wild on an LSF node::

    drwxrwxr-x 3 otheruser otheruser /tmp/_ray    # mode 0775, created Sep 7

Every user outside the ``otheruser`` group then gets ``PermissionError: [Errno
13]`` creating anything beneath it — permanently, for that node, until someone
removes the directory. ``/tmp`` itself was ``drwxrwxrwt`` (world-writable, sticky),
i.e. perfectly healthy; only the intermediate level was poisoned. Jobs failed or
succeeded purely on which node the scheduler picked.

The job id already provides isolation, so the shared level buys nothing. Including
the **uid** in the parent (``/tmp/_ray-<uid>/...``, mode ``0700``) removes the
collision class outright: nobody else can own your parent, and yours can never
become someone else's blocker.

**Why ``TMPDIR`` comes first.** Most GPU schedulers already hand out a private,
auto-cleaned scratch directory, and honouring it is the idiomatic answer on each of
them without fm-tune needing to know which one it is on:

===================  ==========================================
LSF                  ``TMPDIR`` (per-job, auto-cleaned)
Slurm                ``TMPDIR``, or ``SLURM_TMPDIR`` at many sites
PBS / Torque         ``TMPDIR``
Kubernetes / Docker  ``TMPDIR`` if set; the container ``/tmp`` is already private
===================  ==========================================

**Why the length check.** Ray places the plasma store's ``AF_UNIX`` socket inside
this directory, and those paths cap near 108 bytes (``sun_path``). A long candidate
— a deep ``$HOME``, say — fails later inside Ray with a far more confusing error
than a permission denial, so candidates are length-checked before being accepted
and a too-long one is skipped like an unwritable one.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import socket
import stat
import sys
import tempfile
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Environment variables that may carry a scheduler job id, most specific first.
# AUTOTUNE_JOB_ID is fm-tune's own override; the rest cover LSF, Slurm, PBS/Torque
# and SGE-style schedulers. LSB_JOBID alone would silently collapse to a constant
# off-LSF, letting two concurrent jobs on one node collide on the plasma socket.
JOB_ID_ENV_VARS = (
    "AUTOTUNE_JOB_ID",
    "LSB_JOBID",
    "SLURM_JOB_ID",
    "PBS_JOBID",
    "JOB_ID",
)

# Explicit override for the parent directory, checked before TMPDIR.
RAY_TMPDIR_ENV = "FMTUNE_RAY_TMPDIR"

# Scheduler-provided scratch directories, in preference order.
SCRATCH_ENV_VARS = ("TMPDIR", "SLURM_TMPDIR")

# Last-resort root, hardcoded rather than taken from ``tempfile.gettempdir()``.
# gettempdir() itself honours ``TMPDIR``, so relying on it would collapse the
# final candidate onto the scheduler scratch dir — leaving no fallback at all
# precisely when ``TMPDIR`` is the unwritable one.
FALLBACK_ROOT = "/tmp"

# Cap for the temp dir itself. An AF_UNIX path is at most 107 characters (108
# bytes including the NUL), and Ray appends up to 65 beneath what we return:
#   <temp_dir>/session_<26-char timestamp>_<pid>/sockets/plasma_store
# with a 7-digit pid (Linux pid_max is 4194304). 107 - 65 = 42. The cap used to be
# 44, which a real LSF worker suffix (w_<job>_<host>_<pid> under /tmp/_ray-<uid>)
# overflowed by one byte; Ray then fails far more confusingly than a permission
# denial. An over-long *suffix* is shortened instead of rejected (see _fit_suffix).
# macOS caps sun_path at 104 bytes (103 characters), 4 fewer than Linux, so the cap
# there is 103 - 65 = 38; a UUID job id otherwise yields a 104-byte socket path.
MAX_TEMP_DIR_LEN = 38 if sys.platform == "darwin" else 42

# Owner-only, so the directory is private and can never become a shared-parent
# problem for another user the way /tmp/_ray did.
_PRIVATE_MODE = 0o700


def resolve_job_id(default: Optional[str] = None) -> str:
    """A job identifier for naming per-job state, across schedulers.

    Probes :data:`JOB_ID_ENV_VARS` in order and falls back to ``default`` — or the
    pid, which is always unique on the node — so off-scheduler runs still get a
    distinct directory instead of everyone sharing ``job_0``.
    """
    for var in JOB_ID_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return default if default is not None else str(os.getpid())


def _describe(path: str) -> str:
    """``owner:group mode`` for ``path``, for error messages. Best effort."""
    try:
        st = os.stat(path)
    except OSError:
        return "<cannot stat>"
    try:
        import pwd

        owner = pwd.getpwuid(st.st_uid).pw_name
    except Exception:
        owner = str(st.st_uid)
    try:
        import grp

        group = grp.getgrgid(st.st_gid).gr_name
    except Exception:
        group = str(st.st_gid)
    return f"{owner}:{group} {stat.filemode(st.st_mode)}"


def _blocking_parent(path: str) -> Optional[str]:
    """The nearest existing ancestor of ``path``, i.e. what actually denied us."""
    parent = os.path.dirname(path.rstrip(os.sep))
    while parent and parent != os.sep:
        if os.path.exists(parent):
            return parent
        parent = os.path.dirname(parent)
    return os.sep if os.path.exists(os.sep) else None


def _private_base(root: str) -> str:
    """Per-uid parent under ``root``.

    The uid (not ``$USER``, which can be unset in a batch context) keeps the parent
    unambiguously ours.
    """
    return os.path.join(root, f"_ray-{os.getuid()}")


def candidate_roots() -> List[str]:
    """Parent directories to try for Ray state, best first.

    1. :data:`RAY_TMPDIR_ENV` — explicit operator override, used verbatim.
    2. A per-uid directory under each scheduler scratch dir
       (:data:`SCRATCH_ENV_VARS`). Those are usually already private, but the uid
       level costs nothing and protects a site that points ``TMPDIR`` at a shared
       path.
    3. A per-uid directory under the system temp dir (``tempfile.gettempdir()``).
    4. A per-uid directory under :data:`FALLBACK_ROOT`, hardcoded so that a
       poisoned ``TMPDIR`` cannot leave the list without a usable entry (step 3
       follows ``TMPDIR`` and would otherwise duplicate step 2).

    Duplicates are collapsed, so the common case where these coincide still
    yields a single attempt.
    """
    roots: List[str] = []
    override = os.environ.get(RAY_TMPDIR_ENV, "").strip()
    if override:
        roots.append(override)
    for var in SCRATCH_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            roots.append(_private_base(value))
    roots.append(_private_base(tempfile.gettempdir()))
    roots.append(_private_base(FALLBACK_ROOT))

    seen = set()
    unique = []
    for root in roots:
        norm = os.path.normpath(root)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _ensure_private_dir(path: str) -> Optional[str]:
    """Check ``path`` is a real directory we own, and tighten it to 0700.

    Returns ``None`` if usable, else the reason it is not. ``lstat`` (not ``stat``)
    so a symlink planted by another user is seen as one rather than followed. A
    failed chmod rejects the candidate instead of being swallowed: the directory
    could still be group/world-accessible.
    """
    try:
        st = os.lstat(path)
    except OSError as exc:
        return f"cannot lstat {path}: {exc.strerror or exc}"
    if stat.S_ISLNK(st.st_mode):
        return f"{path} is a symlink; refusing to follow it"
    if not stat.S_ISDIR(st.st_mode):
        return f"{path} is not a directory"
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return f"{path} is owned by another user ({_describe(path)})"
    if stat.S_IMODE(st.st_mode) != _PRIVATE_MODE:
        try:
            os.chmod(path, _PRIVATE_MODE)
            st = os.lstat(path)
        except OSError as exc:
            return f"cannot chmod {path} to 0700 ({type(exc).__name__}: {exc.strerror or exc})"
    if stat.S_IMODE(st.st_mode) & 0o077:
        return f"{path} is group/world-accessible after chmod ({_describe(path)})"
    return None


def _try_create(path: str) -> Tuple[bool, Optional[str]]:
    """Create ``path`` (mode 0700) and confirm it is writable.

    Returns ``(ok, reason_if_not)``. ``exist_ok=True`` alone is not enough: it
    suppresses "already exists" but says nothing about an unwritable *parent*,
    which is exactly the ``/tmp/_ray`` failure. So the write is verified directly.
    """
    if len(path) > MAX_TEMP_DIR_LEN:
        return False, f"path too long ({len(path)} > {MAX_TEMP_DIR_LEN}); would break Ray's AF_UNIX plasma socket"
    try:
        os.makedirs(path, mode=_PRIVATE_MODE, exist_ok=True)
    except OSError as exc:
        blocker = _blocking_parent(path)
        detail = f"{type(exc).__name__}: {exc.strerror or exc}"
        if exc.errno in (errno.EACCES, errno.EPERM) and blocker:
            detail += f"; blocked by {blocker} ({_describe(blocker)})"
        return False, detail
    # The per-uid path is predictable, so another local user can pre-create it (or
    # plant a symlink there) before we run; makedirs(exist_ok=True) would accept
    # either. Verify our own levels: the `_ray-<uid>` parent (never a
    # scheduler-provided scratch dir or an operator's explicit override) and the
    # leaf. makedirs applies `mode` to the leaf only, and umask masks it, so both
    # are also chmod'ed to 0700 explicitly — the parent would otherwise inherit the
    # default (often 0755) and become exactly the shared directory this module
    # exists to avoid.
    parent = os.path.dirname(path)
    owned = [path]
    if os.path.basename(parent) == f"_ray-{os.getuid()}":
        owned.insert(0, parent)
    for level in owned:
        reason = _ensure_private_dir(level)
        if reason:
            return False, reason
    # makedirs succeeds on a pre-existing dir we cannot write into, so probe it.
    if not os.access(path, os.W_OK | os.X_OK):
        return False, f"exists but is not writable ({_describe(path)})"
    return True, None


def _fit_suffix(root: str, suffix: str) -> str:
    """``<root>/<suffix>``, with the suffix shortened if the path exceeds the cap.

    The shortened form keeps a readable prefix plus 8 hex chars of the suffix's
    SHA-1, so it is deterministic and distinct suffixes stay distinct. A root that
    leaves no room even for that is returned unchanged, and ``_try_create``
    rejects it as too long like any other unusable candidate.
    """
    path = os.path.join(root, suffix)
    if len(path) <= MAX_TEMP_DIR_LEN:
        return path
    digest = hashlib.sha1(suffix.encode()).hexdigest()[:8]
    room = MAX_TEMP_DIR_LEN - len(os.path.join(root, "")) - len(digest) - 1
    if room < 1:
        return path
    return os.path.join(root, f"{suffix[:room]}-{digest}")


def resolve_ray_temp_dir(suffix: str, *, job_id: Optional[str] = None) -> str:
    """Create and return a Ray temp directory named ``<root>/<suffix>``.

    Walks :func:`candidate_roots` and returns the first usable path, so a single
    node with a poisoned ``/tmp`` parent no longer kills the job. Every rejected
    candidate is logged with *why* — owner and mode of whatever blocked it — since
    a bare ``PermissionError: [Errno 13]`` gives no hint that the cause is
    node-local state rather than a bug.

    Raises ``RuntimeError`` naming the host and every attempt if none work; that
    message is meant to be actionable (usually: ask an admin to remove a stale
    shared directory on that node).
    """
    del job_id  # callers bake the id into `suffix`; kept for call-site clarity
    failures = []
    for root in candidate_roots():
        path = _fit_suffix(root, suffix)
        ok, reason = _try_create(path)
        if ok:
            if failures:
                logger.warning(
                    "[tmpdir] using %s after %d unusable candidate(s): %s",
                    path,
                    len(failures),
                    "; ".join(failures),
                )
            else:
                logger.info("[tmpdir] Ray temp dir: %s", path)
            return path
        failures.append(f"{path} -> {reason}")
        logger.warning("[tmpdir] candidate unusable: %s -> %s", path, reason)

    host = socket.gethostname()
    raise RuntimeError(
        f"[tmpdir] No writable Ray temp directory on host {host!r}. Tried:\n  "
        + "\n  ".join(failures)
        + f"\nSet {RAY_TMPDIR_ENV} to a short, writable, node-local path, or ask an "
        "administrator to remove the stale shared directory named above (a directory "
        "owned by another user/group blocks every other user on that node)."
    )
