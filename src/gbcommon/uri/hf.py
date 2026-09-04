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

"""URI referring to models/datasets/spaces/etc. in HuggingFace Hub"""

import json
import os
import re
import shutil
import sys
import threading
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Iterator, List, Optional, Self

from huggingface_hub import (
    HfApi,
    repo_exists,
    revision_exists,
    scan_cache_dir,
    snapshot_download,
)

from gbcommon.types.testing import is_hf_mocked, standalone_rg_environment
from gbcommon.uri.uri import URI
from gbcommon.utils.fs_lock import SharedFileSystemLock
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import GB_ENVIRONMENT
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# Prefix applied to space names when defining the resource group name.
_GB_RG_SPACE_NAME_PREFIX = "gbspace-"

# Standalone registers "public", "standalone", and "local" as aliases for one
# space dir (gbserver/commands/utils.py), but only "public" has a resource group
# provisioned, so the other two derive from it.
_GB_RG_SPACE_NAME_ALIASES = {"standalone": "public", "local": "public"}
HF_HOST = "huggingface.co"
HF_URI_SCHEME = "hf"

URLSEGMENT_MODELS = "models"
URLSEGMENT_DATASETS = "datasets"
URLSEGMENT_SPACES = "spaces"
URLSEGMENT_BUCKETS = "buckets"

DEFAULT_REVISION = "main"

# hfpull download serialization + self-healing (issue #320).
#
# huggingface_hub wraps each file's ``.incomplete`` write in its own
# cross-process ``WeakFileLock`` inside the destination, so concurrent
# ``snapshot_download`` calls into a shared cache do not corrupt one another's
# per-file writes -- *within a single node*. The failure in #320 was a
# ``FileNotFoundError`` on an ``.incomplete`` file, i.e. the download directory
# was removed out from under a live writer, which no lock can prevent.
#
# pull() therefore does two things:
#   1. Serialization with liveness-aware reclaim: a cross-process lock keyed to
#      the destination so two of our own pullers do not work the same tree at
#      once. A waiter blocks for as long as the holder is *alive and making
#      progress* (so a legitimately long pull is never abandoned mid-write) and
#      reclaims the lock only once the holder has made no progress for the
#      reclaim window (a crashed/evicted holder). It falls through *unlocked*
#      only when the lock filesystem itself is unusable (a read-only mount),
#      relying then on HF's own per-file locks rather than failing the build.
#      Because the reclaim is a heuristic (a live holder that stalls past the
#      window -- e.g. a long HF 429 back-off -- can be misjudged dead), the
#      holder *fences* its own work: it re-checks ownership before any
#      destructive self-heal and after its download, and if it has been evicted
#      it discards the now-unlocked result and re-waits for the lock instead of
#      mutating a tree the new owner is writing. The new owner's self-heal then
#      cleans up any brief write overlap -- so a mis-reclaim costs a re-wait and
#      some owner-side cleanup, and never dueling destructive self-heal. It is
#      not, however, an absolute guarantee against corruption: the fence runs
#      *after* ``snapshot_download`` returns, so if a mis-reclaimed-but-live
#      holder and the new owner both write the same ``.incomplete`` cross-node
#      (HF's per-file lock is node-local; see below), a byte-corruption that
#      happens to preserve the expected file size passes HF's size-based
#      consistency check and is not caught. That residual needs a live holder to
#      stall past the reclaim window and a size-preserving overlap -- rare, and
#      the narrow, HF-internal liveness signal below makes it rarer -- but it is
#      a residual, not an impossibility.
#   2. Self-healing: if a download hits the "removed dir" or "poisoned
#      .incomplete" states, retry once with ``force_download=True`` (which
#      unlinks the bad ``.incomplete``) and, failing that, drop HF's scratch
#      download dir and retry -- turning a manual ``rm -rf`` into automatic
#      recovery. This is the primary #320 fix and is filesystem-agnostic. The
#      self-heal mutates the shared destination, so it only runs while the lock
#      is held; on the unlocked fall-through a recoverable error propagates
#      instead (recovering under a concurrent peer could re-induce #320).
#
# The lock is a ``SharedFileSystemLock`` (gbcommon.utils.fs_lock), which uses
# atomic directory creation (``os.mkdir``) rather than ``flock``/``filelock``.
# Two-node probes showed BSD ``flock(2)`` is node-local on the Blue Vela GPFS
# mount (two nodes held the same exclusive lock at once), so a flock-based lock
# provides no cross-node mutual exclusion there; ``mkdir`` atomicity, by
# contrast, is coherent across nodes on every shared filesystem tested (GPFS and
# the AFM/COS-backed CSI PVC the K8s hfpull steps mount). Each per-revision
# lock directory is removed on release; the shared ``.gb-hfpull-locks``
# container it lives in is intentionally left in place (removing it would race
# with a peer creating a sibling lock), so at most one empty container dir
# remains rather than a lock per pull. mkdir has no kernel auto-release on
# holder death, so the lock runs with a ``ttl`` plus the destination as its
# ``progress_path``: a held lock is reclaimed only after ``ttl`` seconds with no
# writes under ``dest`` (see ``GB_HFPULL_LOCK_TIMEOUT``). A live holder keeps
# writing files, so it is never reclaimed however long its download runs; a dead
# holder's lock is reclaimed within ``ttl`` of its last write, cluster-wide,
# rather than persisting until an operator clears it.
HFPULL_LOCK_TIMEOUT_ENV = "GB_HFPULL_LOCK_TIMEOUT"
HFPULL_FORCE_ENV = "GB_HFPULL_FORCE"
# The lock's no-progress reclaim window (``ttl``), in seconds: how long a held
# lock may show no writes under the destination before a waiter treats the
# holder as dead and reclaims it. Because a *live* holder is detected by its own
# writes (never reclaimed while it keeps writing), this need not exceed a whole
# download -- only the longest plausible gap with no writes for a holder that is
# still alive. Two things widen that gap: the shared FS's attribute-cache
# latency (a waiter on another node can see a stale mtime for a file the holder
# is actively writing, up to tens of seconds on NFS/GPFS) and HF's own stalls
# (a 429 rate-limit ``Retry-After`` back-off, or a congested/retrying transfer,
# can write nothing for minutes). 900s sits comfortably above both, so a
# live-but-stalled holder is very rarely misjudged dead; and if one ever is, the
# ownership fence (see ``_pull_hf_repo``/``pull``) keeps the outcome correct --
# the evicted holder abandons and re-waits rather than corrupting the tree. It
# still bounds the stall behind a genuinely crashed holder to ~15 min rather than
# the indefinite hang of an unreclaimed lock. Overridable via
# GB_HFPULL_LOCK_TIMEOUT. (Kept named "TIMEOUT" for operator continuity: it is
# still "how long before we give up on the current holder.")
DEFAULT_HFPULL_LOCK_TTL_S = 900.0
# Upper bound on the reclaim window (1 year). Nobody needs a longer one, and the
# clamp keeps the Python and shell parses in lockstep: bash 64-bit arithmetic
# would overflow (wrap to garbage/negative) on an absurd GB_HFPULL_LOCK_TIMEOUT
# while Python parses arbitrary integers, silently re-diverging the two windows.
HFPULL_LOCK_TTL_MAX_S = 31_536_000.0
# Poll interval while waiting for a peer to release the mkdir lock.
HFPULL_LOCK_POLL_S = 1.0
# Cap on how many times a pull will re-wait after being evicted mid-download
# (see the ownership fence in ``pull``). A re-wait is normally rare and the retry
# a fast no-op, so hitting this many means sustained contention or a node that
# keeps stalling past the reclaim window -- fail cleanly rather than loop.
HFPULL_MAX_LOCK_REWAITS = 5

# Subdirectory holding hfpull lock directories. It is a dot-prefixed *sibling*
# of the repo's ``<revision>`` dirs (it lives at ``<owner>/<repo>/``), chosen so
# a single container -- not one lock dir per pull -- shares that level, and so
# individual ``<revision>.lock`` dirs never sit directly beside the revision
# dirs. A ``<revision>`` is a git ref or commit hash, never dot-prefixed, so
# ``_hf_repo_id_from_cache_path`` (which strips a trailing hex commit segment)
# does not mistake this container for one; code enumerating ``<owner>/<repo>/*``
# should skip dot-prefixed entries as it would any hidden dir.
HFPULL_LOCK_DIRNAME = ".gb-hfpull-locks"

# huggingface_hub's per-file scratch dir inside a ``local_dir`` snapshot pull.
_HF_DOWNLOAD_SCRATCH = Path(".cache") / "huggingface" / "download"


def _env_flag(name: str) -> bool:
    """True when environment variable *name* is set to a truthy value."""
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _hfpull_lock_ttl() -> float:
    """No-progress reclaim window (seconds) for the hfpull download lock.

    A held lock is reclaimed only after this many seconds with no writes under
    the destination (see ``DEFAULT_HFPULL_LOCK_TTL_S``); a live holder is never
    reclaimed. Overridable via ``GB_HFPULL_LOCK_TIMEOUT``.
    """
    raw = os.getenv(HFPULL_LOCK_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_HFPULL_LOCK_TTL_S
    s = raw.strip()
    # Accept only plain decimals, in lockstep with the LSF/skypilot shell parse,
    # so an operator gets the same window on k8s/CLI and LSF/skypilot (issue
    # #322: divergent parsing silently gave different windows). Scientific
    # notation (e.g. ``1e2``), signs, and ``inf``/``nan`` all fall back to the
    # default rather than being honored on one path and rejected on the other.
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s):
        logger.warning(
            "Invalid %s=%r; using default %ss",
            HFPULL_LOCK_TIMEOUT_ENV,
            raw,
            DEFAULT_HFPULL_LOCK_TTL_S,
        )
        return DEFAULT_HFPULL_LOCK_TTL_S
    whole = int(s.split(".", 1)[0])
    if whole == 0 and not re.fullmatch(r"0+(?:\.0+)?", s):
        # A positive sub-second value rounds up to the 1s poll granularity rather
        # than down to 0 (matches the shell), so it does not collapse to "reclaim
        # any stalled peer immediately".
        return 1.0
    # Clamp to the max so an absurd value stays identical to the shell (whose
    # 64-bit arithmetic would otherwise overflow); see HFPULL_LOCK_TTL_MAX_S.
    return float(min(whole, int(HFPULL_LOCK_TTL_MAX_S)))


def _hfpull_lock_path(dest: Path) -> Path:
    """Lock directory for a pull into *dest*.

    Placed at ``<dest.parent>/.gb-hfpull-locks/<dest.name>.lock`` -- i.e. under a
    dot-prefixed container that is a sibling of *dest* on the shared cache
    filesystem: co-located so peer containers contend on it, but dot-prefixed
    (and one level down, inside the container) so it is distinguishable from the
    revision directories it sits beside. Never placed inside *dest*, which can be
    removed mid-download (the #320 failure mode).
    """
    return dest.parent / HFPULL_LOCK_DIRNAME / f"{dest.name}.lock"


class _HfEvicted(Exception):
    """Raised when our download lock was reclaimed by a peer mid-download.

    A long stall (past ``GB_HFPULL_LOCK_TIMEOUT`` with no writes under the
    progress path) can make a live holder look dead, so a waiter reclaims it.
    When the
    (still-alive) original holder notices -- before a destructive self-heal, or
    after its download returns -- it raises this to abandon the now-unlocked work
    and re-enter the waiter loop, rather than mutating a tree the new owner is
    writing or reporting an unlocked download as authoritative. ``pull`` catches
    it and retries; the new owner's self-heal cleans up any overlap.
    """


@contextmanager
def _hfpull_download_lock(
    dest: Path, progress_path: Path
) -> Iterator[Optional[SharedFileSystemLock]]:
    """Cross-process lock serializing pulls into *dest*, liveness-aware.

    Uses :class:`SharedFileSystemLock` (atomic ``mkdir``, coherent across nodes
    on the shared cache filesystems -- unlike BSD ``flock``; see the module
    comment) with ``timeout=None`` and *progress_path* as its liveness signal. A
    waiter therefore blocks for as long as the holder keeps writing under
    *progress_path* (a live, in-progress pull is never abandoned mid-write) and
    reclaims the lock only after ``GB_HFPULL_LOCK_TIMEOUT`` seconds with no such
    writes (a crashed/evicted holder). It falls through *unlocked* only when the
    lock itself cannot be set up (e.g. a read-only mount), relying then on
    huggingface_hub's own per-file locks rather than failing the build.

    *progress_path* should be a path written **only** by the pull's own workload
    (the caller passes HF's internal scratch dir for repos): with ``timeout=None``
    a waiter trusts "recent writes here" to mean "the holder is alive", so an
    unrelated process writing there would keep a dead holder's lock alive and
    hang waiters. HF's scratch dir is far less exposed to that than the shared
    model dir.

    Yields the :class:`SharedFileSystemLock` when this process holds it, or
    ``None`` when it fell through unlocked. The caller uses that both to gate
    destructive self-healing (only safe while holding the lock) and to fence its
    work: ``lock.still_owned()`` detects a peer having reclaimed the lock, so a
    holder evicted mid-download can abandon and re-wait (see ``_pull_hf_repo``
    and ``pull``).
    """
    lock_dir = _hfpull_lock_path(dest)
    lock = SharedFileSystemLock(
        lock_dir,
        timeout=None,
        poll_interval=HFPULL_LOCK_POLL_S,
        ttl=_hfpull_lock_ttl(),
        progress_path=progress_path,
    )
    if lock.acquire():
        try:
            yield lock
        finally:
            lock.release()
        return

    # timeout=None only returns False on an infra failure (the lock filesystem
    # is unusable), never on contention -- a live holder is waited out and a dead
    # one reclaimed. So this is a broken-mount fall-through, not a busy peer.
    logger.warning(
        "hfpull: could not set up download lock %s (lock filesystem unusable); "
        "proceeding and relying on huggingface_hub's per-file locks",
        lock_dir,
    )
    yield None


# The corrupt-cache signatures a force re-download can clear (issue #320),
# matched on message text so the shell hfpull steps can mirror it exactly and so
# an HF version that changes the error type still recovers. The LSF/skypilot
# shell copies keep this identical in ``HFPULL_RECOVERABLE_RE``; the
# ``test_hfpull_shell_serialization`` guard pins the shell regex equal to this
# pattern so the three copies cannot drift apart.
HF_RECOVERABLE_CACHE_ERROR_RE = re.compile(
    r"(FileNotFoundError|No such file or directory).*\.incomplete"
    r"|Consistency check failed"
)


def _is_recoverable_hf_cache_error(e: BaseException) -> bool:
    """True for the corrupt-cache states a force re-download can clear.

    Two hard states (issue #320): a ``FileNotFoundError`` on an ``.incomplete``
    file (the download dir was removed under a live writer), or
    huggingface_hub's ``Consistency check failed`` size mismatch (a poisoned
    ``.incomplete``). ``force_download=True`` unlinks the bad ``.incomplete``
    before re-downloading, clearing both.

    Classified by the error *message* (see ``HF_RECOVERABLE_CACHE_ERROR_RE``)
    rather than the exception type, so the shell hfpull steps -- which only see
    stdout/stderr -- can mirror it verbatim, and so a huggingface_hub version
    that rewords or retypes the error still triggers recovery instead of
    silently reverting to the #320 failure.
    """
    return bool(HF_RECOVERABLE_CACHE_ERROR_RE.search(str(e)))


def _clear_hf_download_scratch(dest: Path) -> None:
    """Delete huggingface_hub's scratch download dir under *dest*.

    ``<dest>/.cache/huggingface/download`` holds only transient
    ``.incomplete`` / ``.metadata`` files, so dropping it forces a clean
    re-download without touching already-materialized model files.
    """
    scratch = dest / _HF_DOWNLOAD_SCRATCH
    shutil.rmtree(scratch, ignore_errors=True)


def _pull_hf_repo(
    *,
    repo_id: str,
    repo_type: str,
    revision: Optional[str],
    dest: Path,
    token: Optional[str],
    endpoint: Optional[str],
    force: bool,
    lock: Optional[SharedFileSystemLock] = None,
) -> None:
    """snapshot_download into *dest*, self-healing a corrupt HF cache.

    On a recoverable corruption (see ``_is_recoverable_hf_cache_error``) retry
    once with ``force_download=True``; if that still fails, drop the scratch
    download dir and retry once more. Non-recoverable errors propagate.

    The self-heal mutates the shared destination -- ``force_download`` re-downloads
    and unlinks files, and the final step ``rm -rf``s the scratch dir -- so it
    only runs while we hold the cross-process download lock (*lock* is not None).
    When the lock was not acquired (best-effort fall-through, ``lock is None``), a
    peer may be writing the same tree concurrently, and recovering under it could
    pull files out from a live writer and re-induce the #320 corruption; there a
    recoverable error propagates instead of self-healing without mutual exclusion.

    We may also have been *evicted* mid-download: a long no-write stall can make a
    live holder look dead, so a peer reclaims the lock. Before each destructive
    self-heal step we therefore re-check ``lock.still_owned()``; if we no longer
    own it we raise :class:`_HfEvicted` rather than mutate a tree the new owner is
    writing (which would re-induce #320 via dueling repairs). The final download
    has no following step to re-fence, so a failure there also re-checks ownership
    and re-waits on eviction instead of surfacing as a build failure. An
    operator-requested ``force`` on the initial attempt is honored regardless.
    """
    lock_held = lock is not None

    def _download(force_download: bool) -> None:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(dest),
            token=token,
            force_download=force_download,
            endpoint=endpoint,
        )

    def _fence_self_heal() -> None:
        # About to mutate the shared tree (force_download / scratch rm -rf).
        # Safe only while we still own the lock; if a peer reclaimed it, abandon
        # to the waiter loop instead of racing the new owner's writes.
        if lock is not None and not lock.still_owned():
            logger.warning(
                "hfpull: download lock for %s was reclaimed before self-heal; "
                "abandoning to re-wait rather than mutate under the new owner",
                repo_id,
            )
            raise _HfEvicted()

    try:
        _download(force)
        return
    except Exception as first:
        if not _is_recoverable_hf_cache_error(first):
            raise
        if not lock_held:
            # Best-effort lock not held: a peer may be writing the shared tree,
            # so self-healing (force_download / scratch rm -rf) could re-induce
            # #320. Propagate rather than recover without mutual exclusion.
            logger.warning(
                "hfpull: HF download cache for %s looks corrupt (%s) but the "
                "download lock was not held (a peer may be writing it); not "
                "self-healing -- propagating",
                repo_id,
                first,
            )
            raise
        logger.warning(
            "hfpull: HF download cache for %s looks corrupt (%s); retrying "
            "with force_download=True",
            repo_id,
            first,
        )

    # Re-fence immediately before each destructive step, so a peer that reclaims
    # the lock in the window since the last check can't have its tree mutated
    # (force_download re-downloads/unlinks). If evicted, abandon to the re-wait.
    _fence_self_heal()
    try:
        _download(True)
        return
    except Exception as second:
        if not _is_recoverable_hf_cache_error(second):
            raise
        logger.warning(
            "hfpull: force_download retry for %s still failed (%s); clearing "
            "scratch download cache %s and retrying once more",
            repo_id,
            second,
            dest / _HF_DOWNLOAD_SCRATCH,
        )

    # Re-fence immediately before the destructive scratch rm -rf + re-download.
    _fence_self_heal()
    _clear_hf_download_scratch(dest)
    try:
        _download(True)
    except Exception:
        # Unlike the two earlier attempts -- each of which falls through to a
        # *following* fenced step that would catch an eviction -- this final
        # download has nothing after it. A peer that reclaims the lock *during*
        # it surfaces here as a raw download error, which would otherwise fail
        # the build instead of counting as a re-wait like every other eviction
        # point. Re-check ownership: if we were evicted, abandon to the re-wait
        # (raise _HfEvicted); if we still hold the lock the failure is genuine
        # and propagates.
        _fence_self_heal()
        raise


class HfType(StrEnum):
    """The different types of HfURI"""

    MODEL = auto()
    DATASET = auto()
    SPACE = auto()
    BUCKET = auto()


@dataclass(frozen=True)
class _HfParts:
    host: str
    owner: str
    repo: str
    revision: str
    hf_type: Optional[HfType] = None
    path_in_repo: str = ""  # path within the repo, e.g. "checkpoints/model.bin"


_HF_TYPE_TO_SEGMENT: dict[HfType, str] = {
    HfType.MODEL: URLSEGMENT_MODELS,
    HfType.DATASET: URLSEGMENT_DATASETS,
    HfType.SPACE: URLSEGMENT_SPACES,
    HfType.BUCKET: URLSEGMENT_BUCKETS,
}
_HF_SEGMENT_TO_TYPE: dict[str, HfType] = {v: k for k, v in _HF_TYPE_TO_SEGMENT.items()}

# Maps HfType to the repo_type string expected by huggingface_hub APIs.
# HfType.BUCKET has no huggingface_hub equivalent, so it is absent (callers default to "model").
_HF_TYPE_TO_REPO_TYPE: dict[HfType, str] = {
    HfType.MODEL: "model",
    HfType.DATASET: "dataset",
    HfType.SPACE: "space",
}

# Error categories returned by _log_hf_api_error.
HF_ERR_RATE_LIMIT = "rate_limit"
HF_ERR_SERVER = "server"
HF_ERR_AUTH = "auth"
HF_ERR_NOT_FOUND = "not_found"
HF_ERR_OTHER = "other"


def _classify_hf_error(exc: Exception) -> tuple[str, Optional[int]]:
    """Classify an HF API exception into a category tag and HTTP status.

    Args:
        exc: The exception raised by a huggingface_hub call.

    Returns:
        A ``(category, status_code)`` tuple. ``status_code`` is ``None`` for
        non-HTTP exceptions. ``category`` is one of the ``HF_ERR_*`` constants.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        return HF_ERR_OTHER, None
    if status == 429:
        return HF_ERR_RATE_LIMIT, status
    if 500 <= status <= 599:
        return HF_ERR_SERVER, status
    if status in (401, 403):
        return HF_ERR_AUTH, status
    if status == 404:
        return HF_ERR_NOT_FOUND, status
    return HF_ERR_OTHER, status


def _log_hf_api_error(
    op: str, target: str, exc: Exception, not_found_is_benign: bool = False
) -> str:
    """Log an HF API failure at a severity matched to its cause.

    Makes transient/rate-limit conditions stand out in logs instead of being
    flattened into one generic error line. ``429`` (rate limit) and ``5xx``
    (server) log at WARNING since they are typically retriable; ``401``/``403``
    (auth) log at ERROR; ``404`` logs at ERROR unless ``not_found_is_benign``
    (e.g. an :meth:`HfURI.exists` probe of an absent repo), in which case it is
    a DEBUG line; everything else logs at ERROR.

    Args:
        op: Short name of the operation (e.g. ``"push"``, ``"delete"``).
        target: The repo/bucket id or URI the operation acted on.
        exc: The raised exception.
        not_found_is_benign: When True, a ``404`` is logged at DEBUG rather than
            ERROR (the resource being absent is an expected outcome).

    Returns:
        One of the ``HF_ERR_*`` category constants.
    """
    category, status = _classify_hf_error(exc)
    request_id = getattr(exc, "request_id", None)
    server_message = getattr(exc, "server_message", None)
    detail = f"status={status}" if status is not None else "no HTTP status"
    if request_id:
        detail += f", request_id={request_id}"
    if server_message:
        detail += f", server_message={server_message!r}"

    if category == HF_ERR_RATE_LIMIT:
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = getattr(response, "headers", {}).get("Retry-After")
        if retry_after:
            detail += f", Retry-After={retry_after}"
        logger.warning(
            "HF %s RATE LIMIT (HTTP 429) for %s: %s: %s", op, target, detail, exc
        )
    elif category == HF_ERR_SERVER:
        logger.warning("HF %s server error for %s: %s: %s", op, target, detail, exc)
    elif category == HF_ERR_AUTH:
        logger.error("HF %s auth error for %s: %s: %s", op, target, detail, exc)
    elif category == HF_ERR_NOT_FOUND:
        if not_found_is_benign:
            logger.debug("HF %s: %s not found: %s", op, target, detail)
        else:
            logger.error("HF %s not found for %s: %s: %s", op, target, detail, exc)
    else:
        logger.error("HF %s failed for %s: %s: %s", op, target, detail, exc)
    return category


class HfURI(URI):
    """
    Hugging Face URI format:

    hf://[<host>/][<type>/]<owner>/<repo>[/<revision>[/<path_in_repo>]]

    Defaults:
      - host -> "huggingface.co"
      - type -> none (MODEL type can be omitted)
      - revision -> "main"
      - path_in_repo -> "" (repo root)

    When a path_in_repo is present the revision must be explicit so the
    parser can unambiguously separate revision from sub-path.

    Examples:
      # Models (type segment optional — omitting it implies MODEL)
      hf:///mistralai/Mistral-7B-Instruct-v0.3          # implicit MODEL, default host
      hf:///models/mistralai/Mistral-7B-Instruct-v0.3   # explicit MODEL, default host
      hf://huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3  # explicit host
      hf://ibm.com/models/mistralai/Mistral-7B-Instruct-v0.3         # custom host
      hf:///ibm-granite/granite-3.0-8b-instruct/v1.0    # explicit revision
      hf:///ibm-granite/granite-3.0-8b-instruct/main/config.json     # path_in_repo

      # Datasets
      hf://huggingface.co/datasets/wikitext/wikitext-103-v1          # explicit host
      hf:///datasets/org/my-dataset                                  # default host
      hf:///datasets/org/my-dataset/v2/data/train.csv  # revision + path_in_repo

      # Spaces
      hf://huggingface.co/spaces/huggingface/diffusers-gallery

      # Buckets
      hf://huggingface.co/buckets/org/test-bucket1

      # Path within repo (revision must be explicit)
      hf://huggingface.co/models/org/model/main/checkpoints/best
    """

    _thread_local = threading.local()

    def __init__(
        self: Self,
        uri: Optional[urllib.parse.ParseResult] = None,
        context: Optional[str] = None,
        secrets: Optional[dict] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.secrets = secrets or {}
        self.config = config or {}
        self.parts: Optional[_HfParts] = None
        super().__init__(uri, context, secrets, **kwargs)

    @staticmethod
    def get_supported_schemes() -> List[str]:
        return [HF_URI_SCHEME]

    def get_metadata(self) -> Any:
        p = self._parts()
        metadata = {
            "uri": self.get_uristr(self),
            "host": (self.uri.netloc if self.uri else HF_HOST),
            "owner": p.owner,
            "repo": p.repo,
            "revision": p.revision,
            "hf_type": p.hf_type,
        }
        # Include optional config values if present
        if self.config:
            if "organization" in self.config:
                metadata["organization"] = self.config["organization"]
            if "resource_group_id" in self.config:
                metadata["resource_group_id"] = self.config["resource_group_id"]
        return metadata

    def _parts(self) -> _HfParts:
        if self.parts:
            return self.parts
        assert self.uri is not None, "self.uri is None"
        parts = self.uri.path.strip("/").split("/")
        host = self.uri.netloc or HF_HOST  # default to huggingface.co if not provided

        # A common mistake is writing "hf://models/owner/repo" (two slashes)
        # when "hf:///models/owner/repo" (three slashes) was intended. With two
        # slashes the type segment ("models", "datasets", ...) is parsed as the
        # host, so the push/pull silently targets a bogus endpoint instead of
        # huggingface.co. Warn so the failure is diagnosable.
        if host in _HF_SEGMENT_TO_TYPE:
            logger.warning(
                "HF URI '%s' uses '%s' as the host, which is almost certainly a "
                "missing-slash typo: use 'hf:///%s/...' (three slashes) so the "
                "host defaults to %s instead of '%s'.",
                self.uri.geturl(),
                host,
                host,
                HF_HOST,
                host,
            )

        # If the first path segment is a known type keyword, consume it.
        # Otherwise the type defaults to MODEL (the "/models/" segment is optional).
        if parts and parts[0] in _HF_SEGMENT_TO_TYPE:
            hf_type = _HF_SEGMENT_TO_TYPE[parts[0]]
            parts = parts[1:]
        else:
            hf_type = HfType.MODEL

        if not parts or len(parts) < 2:
            raise ValueError(f"Malformed HF URI: {self.uri.geturl()}")

        owner = parts[0]
        repo = parts[1]
        if hf_type == HfType.BUCKET:
            revision = ""
            path_in_repo = "/".join(parts[2:]) if len(parts) > 2 else ""
        else:
            revision = parts[2] if len(parts) > 2 else DEFAULT_REVISION
            path_in_repo = "/".join(parts[3:]) if len(parts) > 3 else ""

        self.parts = _HfParts(
            host=host,
            owner=owner,
            repo=repo,
            revision=revision,
            hf_type=hf_type,
            path_in_repo=path_in_repo,
        )
        return self.parts

    def exists(self: Self, force: bool = False) -> bool:
        """Check whether the HF repo/bucket (and revision) actually exists on the Hub.

        Returns ``False`` on any error, but logs the failure with a severity
        matched to its cause: a genuine ``404`` (resource truly absent) is the
        expected negative path and logs at DEBUG, while a rate-limit (``429``),
        server (``5xx``), or auth (``401``/``403``) error logs at WARNING/ERROR.
        The latter are dangerous here: the resource may actually exist but we
        would still report it missing, so they must be visible in logs.
        """
        if is_hf_mocked():
            return True
        repo_id = "<unparsed>"
        try:
            p = self._parts()
            repo_id = f"{p.owner}/{p.repo}"
            token = self._resolve_token()

            if p.hf_type == HfType.BUCKET:
                endpoint = f"https://{p.host}" if p.host != HF_HOST else None
                api = HfApi(endpoint=endpoint, token=token)
                api.bucket_info(bucket_id=repo_id)
                logger.debug("HF bucket %s exists", repo_id)
                return True

            repo_type = None
            if p.hf_type == HfType.DATASET:
                repo_type = "dataset"
            elif p.hf_type == HfType.SPACE:
                repo_type = "space"

            if p.revision and p.revision != DEFAULT_REVISION:
                found = revision_exists(
                    repo_id=repo_id,
                    revision=p.revision,
                    repo_type=repo_type,
                    token=token,
                )
            else:
                found = repo_exists(repo_id=repo_id, repo_type=repo_type, token=token)
            logger.debug("HF exists check for %s returned %s", repo_id, found)
            return found
        except Exception as e:
            _log_hf_api_error("exists", repo_id, e, not_found_is_benign=True)
            return False

    def is_accessible(self: Self) -> bool:
        return self.exists()

    def is_prod_safe(self: Self) -> bool:
        """HF artifacts are always prod-safe.

        HF URIs use the same host (huggingface.co) across all environments, so
        they carry no prod/non-prod distinction and may register in PROD.
        """
        return True

    def get_revision(self) -> str:
        """Return the branch/tag/ref of the repo."""
        return self._parts().revision

    def get_owner(self) -> str:
        """Return the owner of the repo."""
        return self._parts().owner

    def get_repo(self) -> str:
        """Return the repo name."""
        return self._parts().repo

    def get_host(self) -> str:
        """Return the host name."""
        return self._parts().host

    def get_hf_type(self) -> Optional[HfType]:
        """Return the HF resource type if encoded in the URI."""
        return self._parts().hf_type

    @staticmethod
    def space_name_to_resource_group_name(space_name: Optional[str]) -> str:
        """Return the HF resource group name derived from a GB space name.

        By convention, GB space names are prefixed with ``_GB_RG_SPACE_NAME_PREFIX``
        to form the resource group name used in HuggingFace Enterprise.

        For non-production environments (STAGING, DEV), a lowercase environment
        suffix is appended to differentiate resource groups within the same HF
        organization (e.g. ``gbspace-public-staging``).

        Args:
            space_name: The GB space name to convert.

        Returns:
            The corresponding HF resource group name.
        """
        if not space_name:
            return ""
        canonical = _GB_RG_SPACE_NAME_ALIASES.get(
            space_name.strip().lower(), space_name
        )
        name = f"{_GB_RG_SPACE_NAME_PREFIX}{canonical}"
        upper_env = GB_ENVIRONMENT.upper() if GB_ENVIRONMENT else ""
        if upper_env in ("STAGING", "DEV"):
            name = f"{name}-{GB_ENVIRONMENT.lower()}"
        elif upper_env == "STANDALONE" and (rg_env := standalone_rg_environment()):
            # A test run can redirect the standalone group to one it owns; a real
            # standalone user gets the production group (see
            # GBTEST_STANDALONE_ENVIRONMENT).
            name = f"{name}-{rg_env.lower()}"
        logger.debug(
            "Resolved resource group name '%s' from space '%s' (env=%s)",
            name,
            space_name,
            GB_ENVIRONMENT,
        )
        return name

    def get_artifact_type(self) -> ArtifactType:
        """Return the artifact type based on the HF resource type encoded in the URI.

        HfType.MODEL and an unspecified type (None, which defaults to MODEL) both
        map to ArtifactType.MODEL. HfType.DATASET maps to ArtifactType.DATASET.
        HfType.BUCKET maps to ArtifactType.BUCKET. All other types (SPACE)
        return UNDEFINED.

        Returns:
            ArtifactType: The artifact type.
        """
        match self._parts().hf_type:
            case HfType.MODEL | None:
                return ArtifactType.MODEL
            case HfType.DATASET:
                return ArtifactType.DATASET
            case HfType.BUCKET:
                return ArtifactType.BUCKET
            case _:
                return ArtifactType.UNDEFINED

    def get_path_in_repo(self) -> str:
        """Return the sub-path within the repo encoded in the URI, or empty string."""
        return self._parts().path_in_repo

    def custom_str(self) -> str:
        """Return the canonical URI string, always including the type segment."""
        p = self._parts()

        # Always include the type segment so the URI round-trips through _parts()
        path_parts = []
        if p.hf_type:
            path_parts.append(_HF_TYPE_TO_SEGMENT[p.hf_type])

        path_parts.extend([p.owner, p.repo])

        # Include revision if it is non-default and non-empty, OR a path_in_repo follows it
        if p.revision and (p.revision != DEFAULT_REVISION or p.path_in_repo):
            path_parts.append(p.revision)

        if p.path_in_repo:
            path_parts.append(p.path_in_repo)

        path = "/" + "/".join(path_parts)

        # Always include host in the output
        return f"{HF_URI_SCHEME}://{p.host}{path}"

    @staticmethod
    def parse(uri_str: str) -> "HfURI":
        """Factory method from a raw URI string"""
        return HfURI(urllib.parse.urlparse(uri_str))

    @classmethod
    def from_parts(
        cls,
        owner: str,
        repo: str,
        hf_type: Optional[HfType] = None,
        revision: str = DEFAULT_REVISION,
        host: str = HF_HOST,
        path_in_repo: str = "",
    ) -> "HfURI":
        """
        Create an HfURI from individual components.

        Args:
            owner: Repository owner/organization (e.g., "mistralai")
            repo: Repository name (e.g., "Mistral-7B-Instruct-v0.3")
            hf_type: Type of resource (MODEL, DATASET, SPACE) - encoded in URI if provided
            revision: Git reference (branch/tag/commit), defaults to "main"
            host: Hub host, defaults to "huggingface.co"

        Returns:
            HfURI instance

        Examples:
            # Create model URI
            uri = HfURI.from_parts(
                owner="mistralai",
                repo="Mistral-7B-Instruct-v0.3",
                hf_type=HfType.MODEL
            )
            # Result: hf://huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3

            # Create dataset URI with custom revision
            uri = HfURI.from_parts(
                owner="wikitext",
                repo="wikitext-103-v1",
                hf_type=HfType.DATASET,
                revision="v2.0"
            )
            # Result: hf://huggingface.co/datasets/wikitext/wikitext-103-v1/v2.0

            # Create bucket URI
            uri = HfURI.from_parts(
                owner="org",
                repo="test-bucket1",
                hf_type=HfType.BUCKET
            )
            # Result: hf://huggingface.co/buckets/org/test-bucket1

            # Create with custom host
            uri = HfURI.from_parts(
                owner="google",
                repo="bert-base-uncased",
                hf_type=HfType.SPACE,
                host="internal-hub.company.com"
            )
            # Result: hf://internal-hub.company.com/spaces/google/bert-base-uncased
        """
        type_segment = f"/{_HF_TYPE_TO_SEGMENT[hf_type]}" if hf_type else ""
        path = f"{type_segment}/{owner}/{repo}"
        if hf_type != HfType.BUCKET:
            if revision != DEFAULT_REVISION or path_in_repo:
                path += f"/{revision}"
        if path_in_repo:
            path += f"/{path_in_repo}"

        uri_str = f"{HF_URI_SCHEME}://{host}{path}"
        return cls.parse(uri_str)

    # Note: currently not using this (comments can be deleted after PR approval)

    # def get_repo_url_https(self, token: Optional[str] = None) -> str:
    #     """
    #     Construct the HTTPS URL for the Hugging Face repo.
    #     Example: https://huggingface.co/owner/repo
    #     """
    #     p = self._parts()
    #     base_path = f"{p.owner}/{p.repo}"

    #     url = f"https://{HF_HOST}/{base_path}"
    #     if token:
    #         url = f"https://{token}:x-oauth-basic@{HF_HOST}/{base_path}"

    #     return url

    # def get_repo_from_cache(self: Self, token: Optional[str] = None, force: bool = False) -> Path:
    #     """Shallow clone into a temp cache dir, reuse if available."""
    #     if not hasattr(self._thread_local, "repo_cache"):
    #         self._thread_local.repo_cache = Path(tempfile.mkdtemp())

    #     https_url = self.get_repo_url_https(token=token)
    #     p = self._parts()
    #     repo_cache_path = self._thread_local.repo_cache / f"{p.owner}-{p.repo}-{p.revision}"

    #     if repo_cache_path.exists() and not force:
    #         return repo_cache_path

    #     if repo_cache_path.exists():
    #         import shutil
    #         shutil.rmtree(repo_cache_path, ignore_errors=True)

    #     # Repo.clone_from(https_url, repo_cache_path, branch=p.revision, single_branch=True, depth=1)
    #     return repo_cache_path

    def _resolve_token(self) -> Optional[str]:
        """Resolve HF auth token from secrets dict or HF_TOKEN env var.

        Returns:
            Token string if found, or None if absent/blank.
        """
        token = self.secrets.get("HF_TOKEN") if self.secrets else None
        if token is None:
            token = os.getenv("HF_TOKEN")
        return token if token and token.strip() else None

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        """Download the HuggingFace repo or bucket referenced by this URI.

        For repos, uses ``huggingface_hub.snapshot_download`` with ``local_dir``
        so all repo files land directly in *dest*.  For buckets, uses
        ``HfApi.sync_bucket`` to download bucket contents to *dest*.

        Pulls into the same *dest* are serialized behind a cross-process
        ``mkdir`` lock (issue #320): a waiter blocks while the holder keeps
        writing under *dest* and reclaims the lock only after
        ``GB_HFPULL_LOCK_TIMEOUT`` seconds with no such writes (a dead holder),
        proceeding unlocked only if the lock filesystem is unusable rather than
        failing. If this process is *itself* reclaimed mid-download (a long stall
        made it look dead), it discards its partial, now-unlocked work and
        re-waits for the lock rather than treating it as authoritative -- up to
        ``HFPULL_MAX_LOCK_REWAITS`` times, after which it fails cleanly rather
        than loop. Repo downloads also self-heal a corrupt HF cache by retrying
        with ``force_download`` and, if needed, clearing the scratch download
        dir.

        Returns ``True`` immediately without network calls when HF mocking is
        enabled via ``GBTEST_MOCK_HF``.

        Token is resolved from ``self.secrets['HF_TOKEN']`` or the ``HF_TOKEN``
        environment variable.  For non-default hosts the ``endpoint`` kwarg is
        forwarded to the Hub client.

        Args:
            dest: Local directory to download files into.
            force: Re-download even if files already exist locally, unlinking any
                poisoned ``.incomplete`` files (repo pulls only).

        Returns:
            True if the download succeeded, False on any error.
        """
        if is_hf_mocked():
            return True
        try:
            # Normalize so the lock-path derivation below tolerates a str dest,
            # matching the pre-lock behavior (downloads only ever used str(dest)).
            dest = Path(dest)
            p = self._parts()
            repo_id = f"{p.owner}/{p.repo}"
            endpoint = f"https://{p.host}" if p.host != HF_HOST else None
            token = self._resolve_token()
            hf_type = p.hf_type
            repo_type = (
                _HF_TYPE_TO_REPO_TYPE.get(hf_type, "model")
                if hf_type is not None
                else "model"
            )

            # Serialize concurrent hfpull processes sharing this destination
            # (issue #320). ``lock`` is None when we fell through unlocked (an
            # unusable lock filesystem); otherwise it gates the destructive
            # self-heal and lets us fence our own work. If we are evicted
            # mid-download (a long stall made us look dead and a peer reclaimed
            # the lock), ``_HfEvicted`` brings us back here to re-acquire and
            # retry -- our partial work is discarded and the new owner's copy
            # becomes authoritative; the already-downloaded files make the retry
            # a fast no-op. Re-waits are capped (HFPULL_MAX_LOCK_REWAITS) so a
            # reclaim storm fails cleanly instead of looping forever.
            #
            # Liveness (progress_path) watches HF's internal scratch dir for repo
            # pulls -- that is where the download's own writes land, and being
            # HF-internal it is far less likely to be touched by an unrelated
            # process sharing the model cache than the whole dest (also cheaper to
            # scan). Buckets have no such scratch dir, so they watch dest.
            progress_path = (
                dest if p.hf_type == HfType.BUCKET else dest / _HF_DOWNLOAD_SCRATCH
            )
            rewaits = 0
            while True:
                try:
                    with _hfpull_download_lock(dest, progress_path) as lock:
                        if p.hf_type == HfType.BUCKET:
                            bucket_hf_path = f"hf://buckets/{repo_id}"
                            if p.path_in_repo:
                                bucket_hf_path += f"/{p.path_in_repo}"
                            logger.info("Downloading HF bucket %s to %s", repo_id, dest)
                            api = HfApi(endpoint=endpoint, token=token)
                            api.sync_bucket(source=bucket_hf_path, dest=str(dest))
                            logger.debug(
                                "Completed HF pull of bucket %s to %s", repo_id, dest
                            )
                        else:
                            logger.info(
                                "Downloading HF repo %s (type=%s, rev=%s) to %s",
                                repo_id,
                                repo_type,
                                p.revision,
                                dest,
                            )
                            _pull_hf_repo(
                                repo_id=repo_id,
                                repo_type=repo_type,
                                revision=p.revision,
                                dest=dest,
                                token=token,
                                endpoint=endpoint,
                                force=force,
                                lock=lock,
                            )
                            logger.debug(
                                "Completed HF pull of %s (type=%s, rev=%s) to %s",
                                repo_id,
                                repo_type,
                                p.revision,
                                dest,
                            )
                        # Ownership fence: if we held the lock but lost it while
                        # downloading, our result was produced (partly) unlocked
                        # -- discard and re-wait rather than treat a tree the new
                        # owner is still writing as authoritative.
                        if lock is not None and not lock.still_owned():
                            raise _HfEvicted()
                        return True
                except _HfEvicted:
                    rewaits += 1
                    if rewaits > HFPULL_MAX_LOCK_REWAITS:
                        logger.error(
                            "hfpull: download lock for %s was reclaimed %d times "
                            "(exceeds the %d re-wait cap); giving up rather than "
                            "retrying indefinitely -- the destination is under "
                            "sustained contention or this node keeps stalling past "
                            "the reclaim window",
                            repo_id,
                            rewaits,
                            HFPULL_MAX_LOCK_REWAITS,
                        )
                        return False
                    logger.warning(
                        "hfpull: download lock for %s was reclaimed mid-download; "
                        "re-waiting for it and retrying (%d/%d)",
                        repo_id,
                        rewaits,
                        HFPULL_MAX_LOCK_REWAITS,
                    )
        except Exception as e:
            _log_hf_api_error("pull", str(self), e)
            return False

    @staticmethod
    def hfpush_step(
        uri_str: str,
        source_path: str,
        private: bool = True,
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        space_name: Optional[str] = None,
        timeout_seconds: int = 3600,
    ) -> int:
        """Parse a HF URI string, push source_path to it, and return an exit code.

        Handles the push timeout via ``signal.SIGALRM`` so the process does not
        hang indefinitely if the HF API becomes unresponsive.  Intended as the
        single entry point called from the hfpush builtin step script.  Accepts
        any combination of ``resource_group_id``, ``resource_group_name``, and
        ``space_name`` for backward compatibility with older helm templates;
        resolution (and consistency verification when more than one is
        provided) happens inside :meth:`HfURI.push`.

        Args:
            uri_str: HF URI string to push to (e.g. ``hf:///owner/repo``).
            source_path: Local file or directory path to upload.
            private: Whether to create a private repo/bucket if it does not exist.
            resource_group_id: Pre-resolved HF Enterprise resource group id.
            resource_group_name: HF Enterprise resource group name.  Resolved
                to an id via the HF API when ``resource_group_id`` is not
                provided.
            space_name: GB space name.  Converted to a resource group name
                (``"gbspace-<space>"``) and then resolved to an id.
            timeout_seconds: Seconds before the push is aborted. Defaults to 3600.

        Returns:
            0 on success, 1 on failure or timeout.
        """
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"HF push timed out after {timeout_seconds} seconds")

        # Helm templates pass unset values as empty strings via `default ""`.
        resource_group_id = resource_group_id or None
        resource_group_name = resource_group_name or None
        space_name = space_name or None

        uri = HfURI.parse(uri_str)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            uri.push(
                Path(source_path),
                private=private,
                resource_group_id=resource_group_id,
                resource_group_name=resource_group_name,
                space_name=space_name,
            )
            signal.alarm(0)  # cancel alarm on success
            return 0
        except Exception as e:
            print(f"HF push failed: {e}", flush=True)
            return 1

    @staticmethod
    def hfpull_step(uri_str: str, dest: str, force: bool = False) -> int:
        """Parse a HF URI string, pull its contents to dest, and return an exit code.

        Intended as the single entry point called from the hfpull builtin step
        script so the helm template stays minimal.

        Args:
            uri_str: HF URI string to download (e.g. ``hf:///owner/repo``).
            dest: Local filesystem path to download files into.
            force: Re-download even if files already exist locally, unlinking any
                poisoned ``.incomplete`` files. Also enabled by setting
                ``GB_HFPULL_FORCE`` in the step environment, so an operator can
                force a clean re-pull without editing the launch command.

        Returns:
            0 on success, 1 on failure.
        """
        force = force or _env_flag(HFPULL_FORCE_ENV)
        uri = HfURI.parse(uri_str)
        return 0 if uri.pull(Path(dest), force=force) else 1

    def delete(self: Self) -> bool:
        """Delete the resource referenced by this URI from the HuggingFace Hub.

        For repos: if ``path_in_repo`` is set, deletes only that file;
        otherwise deletes the entire repository and cleans the local cache.
        For buckets: if ``path_in_repo`` is set, deletes that file from the
        bucket; otherwise deletes the entire bucket.

        Returns:
            True if deletion succeeded, False on any error.
        """
        if is_hf_mocked():
            return True
        p = self._parts()
        try:
            repo_id = f"{p.owner}/{p.repo}"
            endpoint = f"https://{p.host}" if p.host != HF_HOST else None
            api = HfApi(endpoint=endpoint, token=self._resolve_token())

            if p.hf_type == HfType.BUCKET:
                if p.path_in_repo:
                    logger.info(
                        "Deleting file %s from bucket %s", p.path_in_repo, repo_id
                    )
                    api.batch_bucket_files(bucket_id=repo_id, delete=[p.path_in_repo])
                else:
                    logger.info("Deleting bucket %s", repo_id)
                    api.delete_bucket(bucket_id=repo_id)
                return True

            hf_type = p.hf_type
            repo_type = (
                _HF_TYPE_TO_REPO_TYPE.get(hf_type, "model")
                if hf_type is not None
                else "model"
            )
            if p.path_in_repo:
                logger.info(
                    "Deleting file %s from %s (type=%s, rev=%s)",
                    p.path_in_repo,
                    repo_id,
                    repo_type,
                    p.revision,
                )
                api.delete_file(
                    path_in_repo=p.path_in_repo,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=p.revision,
                )
            else:
                logger.info("Deleting repo %s (type=%s)", repo_id, repo_type)
                api.delete_repo(repo_id=repo_id, repo_type=repo_type)
            self._delete_from_cache(repo_id, repo_type, p.revision)
            logger.debug("Completed HF delete for %s", self)
            return True
        except Exception as e:
            _log_hf_api_error("delete", str(self), e)
            return False

    def _delete_from_cache(self, repo_id: str, repo_type: str, revision: str) -> None:
        """Remove the cached entry for a specific revision from the HF hub cache.

        Only the revision matching the URI's ``revision`` ref is deleted; other
        cached revisions of the same repo are left intact.  Cache cleanup failures
        are logged as warnings and do not propagate.

        Args:
            repo_id: The repo ID in ``owner/repo`` form.
            repo_type: The repo type string (``"model"``, ``"dataset"``, ``"space"``).
            revision: The ref (branch/tag/commit) to remove from the cache.
        """
        try:
            cache_info = scan_cache_dir()
            for cached_repo in cache_info.repos:
                if (
                    cached_repo.repo_id == repo_id
                    and cached_repo.repo_type == repo_type
                ):
                    # Match revisions where the ref set contains our revision, or
                    # where the commit hash starts with the revision string (for
                    # short commit hashes).
                    hashes = [
                        rev.commit_hash
                        for rev in cached_repo.revisions
                        if revision in rev.refs or rev.commit_hash.startswith(revision)
                    ]
                    if hashes:
                        cache_info.delete_revisions(*hashes).execute()
                        logger.info(
                            "Deleted cached revision %s for %s (type=%s)",
                            revision,
                            repo_id,
                            repo_type,
                        )
                    break
        except Exception as e:
            logger.warning("Could not clean HF cache for %s: %s", repo_id, e)

    @classmethod
    def resolve_resource_group_id_for_org(
        cls,
        token: Optional[str],
        organization: str,
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        space_name: Optional[str] = None,
        host: str = HF_HOST,
    ) -> Optional[str]:
        """Resolve an HF Enterprise resource group id from id, name, or space.

        Class-level entry point that does not require an ``HfURI`` instance.

        Handles the full ``space_name -> resource_group_name -> resource_group_id``
        flow in one place.  Any combination of inputs is accepted as long as
        they agree:

        * If ``resource_group_id`` is given with no name or space, it is
          returned as-is (no API call).
        * If ``resource_group_name`` and/or ``space_name`` are given (without
          ``resource_group_id``), the name is resolved via the HF API.
        * If ``resource_group_id`` is given *and* a name/space is also given,
          the name/space is resolved and the result is verified to match the
          explicit id; a ``ValueError`` is raised on mismatch.
        * If both ``resource_group_name`` and ``space_name`` are given, the
          name derived from ``space_name`` must equal the explicit name.

        Args:
            token: HF auth token used to build the temporary ``HfApi``.  May be
                ``None`` for anonymous lookups, though Enterprise resource
                groups typically require authentication.
            organization: HF organization namespace.
            resource_group_id: Explicit resource group id (e.g. from
                build.yaml ``store_push.config.hf.resource_group_id``).
            resource_group_name: Explicit resource group name.
            space_name: GB space name; converted to a resource group name via
                :meth:`space_name_to_resource_group_name`.
            host: HF host (defaults to ``huggingface.co``).

        Returns:
            The resolved resource group id, or ``None`` when no inputs were
            supplied.

        Raises:
            ValueError: If any of the provided inputs disagree, or if name/space
                resolution fails.
        """
        if is_hf_mocked():
            # When this op is mocked there is no live Hub to query; skip the
            # resource-group lookup (which would hit the /resource-groups list
            # endpoint and require a token) and keep any explicit id, else None.
            return resource_group_id
        if not resource_group_id and not resource_group_name and not space_name:
            return None
        derived_name = (
            cls.space_name_to_resource_group_name(space_name) if space_name else None
        )
        if resource_group_name and derived_name and derived_name != resource_group_name:
            raise ValueError(
                f"space-derived resource group name '{derived_name}' does not "
                f"match resource group name '{resource_group_name}'"
            )
        effective_name = resource_group_name or derived_name
        if effective_name is None:
            return resource_group_id
        endpoint = f"https://{host}" if host != HF_HOST else None
        api = HfApi(endpoint=endpoint, token=token)
        resolved_id = cls._resolve_resource_group_id(api, organization, effective_name)
        if not resolved_id:
            raise ValueError(
                f"Could not resolve resource group id for '{effective_name}' "
                f"in organization '{organization}'"
            )
        if resource_group_id and resource_group_id != resolved_id:
            raise ValueError(
                f"resource group id '{resource_group_id}' does not match the "
                f"id '{resolved_id}' resolved from '{effective_name}' in "
                f"organization '{organization}'"
            )
        logger.info(
            "Resolved resource group id '%s' for name '%s' in org '%s'",
            resolved_id,
            effective_name,
            organization,
        )
        return resolved_id

    def resolve_resource_group_id(
        self: Self,
        token: Optional[str],
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        space_name: Optional[str] = None,
    ) -> Optional[str]:
        """Instance-level convenience wrapper around :meth:`resolve_resource_group_id_for_org`.

        Extracts *organization* and *host* from this URI's parts.
        """
        p = self._parts()
        return self.resolve_resource_group_id_for_org(
            token=token,
            organization=p.owner,
            resource_group_id=resource_group_id,
            resource_group_name=resource_group_name,
            space_name=space_name,
            host=p.host,
        )

    @classmethod
    def _resolve_resource_group_id(
        cls, api: HfApi, organization: str, name: str
    ) -> Optional[str]:
        """Look up a resource group ID by name within the given organization.

        Uses the HF Hub enterprise REST API directly since ``huggingface_hub``
        does not expose a ``list_resource_groups`` helper.

        Args:
            api: Authenticated HfApi instance.
            organization: Organization namespace to search.
            name: Resource group name to look up.

        Returns:
            The resource group ID string if found, or None.
        """
        from huggingface_hub.utils._http import get_session, hf_raise_for_status

        try:
            r = get_session().get(
                f"{api.endpoint}/api/organizations/{organization}/resource-groups",
                headers=api._build_hf_headers(),
            )
            hf_raise_for_status(r)
            for group in r.json():
                if group.get("name") == name:
                    return group.get("id") or group.get("resourceGroupId")
            logger.warning(
                "Resource group '%s' not found in organization '%s'", name, organization
            )
        except Exception as e:
            # This REST call is an extra request on the hot push path, so it is a
            # prime rate-limit target; classify it so a 429/5xx here is visible
            # rather than hidden behind a bland "could not list" warning.
            _log_hf_api_error("list_resource_groups", organization, e)
        return None

    # Cap on how many unreadable paths a failure message lists, so a wholly
    # unreadable tree of thousands of files yields a diagnosable error rather
    # than an unreadably long one. The count reported is always the true total.
    _MAX_UNREADABLE_REPORTED = 10

    @staticmethod
    def _validate_non_empty_src(src: Path) -> None:
        """Ensure ``src`` has uploadable, non-zero-length, *readable* content.

        HuggingFace silently skips an upload that would produce an empty commit
        (e.g. a single 0-byte file), so a push of empty content appears to
        succeed while creating nothing on the Hub.  Fail fast instead.

        Readability is checked here too, in the same walk, because
        ``upload_folder`` opens each file only once it is already mid-commit:
        the resulting ``PermissionError`` carries no HTTP status, so
        :meth:`_log_hf_api_error` can only classify it as ``HF_ERR_OTHER`` and
        logs "no HTTP status", which reads like a Hub outage rather than a local
        ``EACCES``. It also aborts on the first bad file, so an operator fixes
        one path at a time. Checking up front names every unreadable file and
        attributes the failure to the filesystem, where the fix is.

        This is a diagnostic, not a guarantee: ``os.access`` is a point-in-time
        check and a push can still race a permission change, so the ``push``
        call sites keep their exception handling.

        Args:
            src: Local file or directory path being pushed.

        Raises:
            ValueError: If ``src`` is a zero-length file, or a directory whose
                regular files are all zero-length (or which contains none).
            PermissionError: If ``src`` itself, or any regular file under it, is
                not readable by the current user.
        """
        if src.is_file():
            if src.stat().st_size == 0:
                raise ValueError(f"refusing to push zero-length file: {src}")
            if not os.access(src, os.R_OK):
                raise PermissionError(f"cannot read file to push: {src}")
            return
        # Directory: require at least one non-empty regular file, and collect
        # every unreadable one in the same walk so the error names them all.
        has_content = False
        unreadable: List[Path] = []
        for f in src.rglob("*"):
            # A directory that cannot be traversed hides its children from
            # rglob entirely, so check those too -- otherwise an unreadable
            # subtree looks simply empty.
            if f.is_dir():
                if not os.access(f, os.R_OK | os.X_OK):
                    unreadable.append(f)
                continue
            if not f.is_file():
                continue
            if not os.access(f, os.R_OK):
                unreadable.append(f)
                continue
            if f.stat().st_size > 0:
                has_content = True
        if unreadable:
            shown = sorted(str(p) for p in unreadable)
            elided = len(shown) - HfURI._MAX_UNREADABLE_REPORTED
            if elided > 0:
                shown = shown[: HfURI._MAX_UNREADABLE_REPORTED]
                shown.append(f"... and {elided} more")
            raise PermissionError(
                f"cannot read {len(unreadable)} path(s) under {src}; "
                "the step that produced this artifact left them unreadable to "
                "this user: " + ", ".join(shown)
            )
        if not has_content:
            raise ValueError(
                f"refusing to push directory with no non-empty files: {src}"
            )

    @staticmethod
    def _hf_repo_id_from_cache_path(path: str) -> Optional[str]:
        """Derive an ``owner/repo`` HF id from a local HF cache path.

        Base models are pulled onto the cluster into a cache laid out as
        ``<cache>/<owner>/<repo>/<revision>``, where the trailing revision is a
        git ref or commit hash. Returns ``owner/repo`` or ``None`` when the path
        has too few components to derive one.

        This assumes the fixed ``owner/repo/revision`` cache layout: it takes
        the last two segments as ``owner/repo`` after dropping a trailing
        commit-hash segment. Inputs that do not follow that layout (e.g. an
        all-hex repo name with no revision, or a non-hex trailing segment) can
        mis-parse, but the cluster cache always produces this layout.
        """
        segments = [seg for seg in path.strip("/").split("/") if seg]
        # Drop a trailing revision/commit-hash segment if present so the last
        # two segments are the owner and repo.
        if segments and re.fullmatch(r"[0-9a-f]{7,64}", segments[-1]):
            segments = segments[:-1]
        if len(segments) < 2:
            return None
        return f"{segments[-2]}/{segments[-1]}"

    @staticmethod
    def _normalize_adapter_base_model(src: Path) -> None:
        """Rewrite a LoRA adapter's local base-model path to its HF repo id.

        When a PEFT/LoRA adapter is trained on the cluster the base model is
        first pulled into a local cache (``<cache>/<owner>/<repo>/<revision>``)
        and the trainer records that local path in ``adapter_config.json`` under
        ``base_model_name_or_path``. Uploaded verbatim, HuggingFace then shows
        the pod-local path (e.g.
        ``/gb-read-write/hfcache/ibm-granite/granite-4.1-3b/ec5a9d...``) instead
        of the ``owner/repo`` id and cannot link the base model.

        This mirrors what the Lakehouse (lhpush) path already does: derive the
        ``owner/repo`` id from the cache layout and write it back before the
        upload. Only local absolute paths are rewritten; values that already
        look like an HF repo id are left untouched.
        """
        config_path = src / "adapter_config.json"
        if not config_path.is_file():
            return
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Could not read %s to normalize base model: %s", config_path, e
            )
            return
        base = config.get("base_model_name_or_path")
        # Only rewrite pod-local absolute paths; a real HF id such as
        # "ibm-granite/granite-4.1-3b" does not start with "/".
        if not isinstance(base, str) or not base.startswith("/"):
            return
        repo_id = HfURI._hf_repo_id_from_cache_path(base)
        if not repo_id:
            logger.warning(
                "Could not derive an HF repo id from base_model_name_or_path "
                "'%s'; leaving it unchanged.",
                base,
            )
            return
        config["base_model_name_or_path"] = repo_id
        try:
            config_path.write_text(json.dumps(config, indent=2))
        except OSError as e:
            logger.warning(
                "Could not write normalized base model to %s: %s", config_path, e
            )
            return
        logger.info(
            "Normalized adapter base_model_name_or_path '%s' -> '%s' for HF upload",
            base,
            repo_id,
        )

    def push(
        self: Self,
        src: Path,
        commit_message: str = "Upload via gbserver",
        private: Optional[bool] = None,
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        space_name: Optional[str] = None,
    ) -> None:
        """Upload a local file or directory to a HuggingFace repo or bucket.

        For repos the destination path is derived from the URI's
        ``path_in_repo`` segment.  Uses ``HfApi.upload_file`` for a single
        file and ``HfApi.upload_folder`` for a directory.

        For buckets, uses ``HfApi.create_bucket`` to ensure the bucket exists,
        then ``HfApi.batch_bucket_files`` for a single file or
        ``HfApi.sync_bucket`` for a directory.  The ``commit_message`` arg is
        ignored for buckets (no Git commits).

        Args:
            src: Local path to a file or directory to upload.
            commit_message: Commit message attached to the upload (repos only).
            private: Whether to create a private repo/bucket (if creating).
            resource_group_id: Optional resource group ID for Enterprise
                access control.  May be combined with ``resource_group_name``
                or ``space_name`` as long as they agree — see
                :meth:`resolve_resource_group_id`.
            resource_group_name: Optional resource group name; resolved to an
                ID via the HF API.
            space_name: GB space name used to derive the resource group name.

        Raises:
            ValueError: If ``src`` does not exist, or is a zero-length file or a
                directory with no non-empty files (HuggingFace would skip the
                resulting empty commit, leaving the push a silent no-op).
            Exception: Any error from the HuggingFace Hub API is re-raised.
        """
        if is_hf_mocked():
            return
        p = self._parts()
        repo_id = f"{p.owner}/{p.repo}"
        endpoint = f"https://{p.host}" if p.host != HF_HOST else None

        api = HfApi(endpoint=endpoint, token=self._resolve_token())
        src = Path(src)
        if not src.exists():
            raise ValueError(f"{src} does not exist")
        self._validate_non_empty_src(src)

        resource_group_id = self.resolve_resource_group_id(
            token=self._resolve_token(),
            resource_group_id=resource_group_id,
            resource_group_name=resource_group_name,
            space_name=space_name,
        )

        if p.hf_type == HfType.BUCKET:
            bucket_id = repo_id
            # Ensure the bucket exists before uploading; abort the push if this
            # fails so we never attempt an upload to a non-existent bucket.
            try:
                api.create_bucket(
                    bucket_id=bucket_id,
                    private=private,
                    resource_group_id=resource_group_id,
                    exist_ok=True,
                )
            except Exception as e:
                _log_hf_api_error("create_bucket", bucket_id, e)
                raise
            logger.debug("Ensured HF bucket %s exists", bucket_id)
            if src.is_file():
                dest_path = p.path_in_repo or src.name
                logger.info(
                    "Uploading file %s to bucket %s/%s", src, bucket_id, dest_path
                )
                try:
                    api.batch_bucket_files(bucket_id=bucket_id, add=[(src, dest_path)])
                except Exception as e:
                    _log_hf_api_error("upload to bucket", bucket_id, e)
                    raise
            else:
                bucket_hf_path = f"hf://buckets/{bucket_id}"
                if p.path_in_repo:
                    bucket_hf_path += f"/{p.path_in_repo}"
                logger.info("Uploading folder %s to bucket %s", src, bucket_id)
                try:
                    api.sync_bucket(source=str(src), dest=bucket_hf_path)
                except Exception as e:
                    _log_hf_api_error("upload to bucket", bucket_id, e)
                    raise
            logger.debug("Completed HF push of %s to bucket %s", src, bucket_id)
            return

        hf_type = p.hf_type
        repo_type = (
            _HF_TYPE_TO_REPO_TYPE.get(hf_type, "model")
            if hf_type is not None
            else "model"
        )

        # Create repository if it doesn't exist; abort the push on failure so we
        # never fall through to an upload against a repo that was not created.
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                resource_group_id=resource_group_id,
                exist_ok=True,
            )
        except Exception as e:
            _log_hf_api_error("create_repo", repo_id, e)
            raise
        logger.debug("Ensured HF repo %s (type=%s) exists", repo_id, repo_type)

        if src.is_file():
            dest_path = p.path_in_repo or src.name
            logger.info(
                "Uploading file %s → %s/%s (type=%s, rev=%s)",
                src,
                repo_id,
                dest_path,
                repo_type,
                p.revision,
            )
            try:
                api.upload_file(
                    path_or_fileobj=src,
                    path_in_repo=dest_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=p.revision,
                    commit_message=commit_message,
                )
            except Exception as e:
                _log_hf_api_error("upload_file", repo_id, e)
                raise
        else:
            # Only folder uploads can carry an adapter_config.json; LoRA
            # adapters are always pushed as a directory, so single-file uploads
            # have nothing to normalize.
            if repo_type == "model":
                self._normalize_adapter_base_model(src)
            logger.info(
                "Uploading folder %s → %s/%s (type=%s, rev=%s)",
                src,
                repo_id,
                p.path_in_repo,
                repo_type,
                p.revision,
            )
            try:
                api.upload_folder(
                    folder_path=str(src),
                    path_in_repo=p.path_in_repo,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=p.revision,
                    commit_message=commit_message,
                )
            except Exception as e:
                _log_hf_api_error("upload_folder", repo_id, e)
                raise
        logger.debug("Completed HF push of %s to %s", src, repo_id)
