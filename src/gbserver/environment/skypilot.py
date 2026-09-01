"""SkyPilot environment backend (unmanaged mode).

Manages build step execution on SkyPilot-provisioned pods/VMs using
sky.launch(). Each step gets its own cluster; pods auto-stop after
idle timeout. The sky SDK is lazy-imported so gbserver does not
require it unless a Skypilot environment is actually configured.
"""

import asyncio
import concurrent.futures
import functools
import glob
import os
import shlex
import threading
import time
import urllib.parse
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Union,
)

from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from gbcommon.uri.uri import URI
from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.spaces.resource_group import (
    apply_hf_step_overlay,
    resolve_hfpush_resource_group_id,
)
from gbserver.types.buildconfig import BuildTargetStepConfig
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.types.errors import WorkloadFailedException
from gbserver.utils.logger import get_logger

if TYPE_CHECKING:
    from gbserver.monitoring.logfile_monitor import LogFileMonitor
    from gbserver.resilience.retry_handler import RetryStrategy

logger = get_logger(__name__)

from gbserver.utils.optional_imports import HAS_SKYPILOT

if HAS_SKYPILOT:
    import sky
    import sky.exceptions
else:
    sky = None  # type: ignore[assignment]

_DEFAULT_POLL_INTERVAL_SECONDS = 300

# Dedicated thread pool for the blocking SkyPilot provisioning submits that a
# cancel may *orphan* — the shielded sky.launch / sky.jobs.launch /
# sky.stream_and_get calls whose futures _abort_shielded_request abandons when
# its drain times out. asyncio.to_thread runs on the event loop's default
# ThreadPoolExecutor (~min(32, cpu+4) workers), shared by every to_thread in
# gbserver; an abandoned provisioning thread blocks until its SDK call returns
# naturally (potentially the full provisioning time), holding a shared slot, so
# a few aborted provisions could starve unrelated offloaded work. Running them
# on their own pool bounds the blast radius to SkyPilot provisioning. Lazily
# created so merely importing this module (or running without SkyPilot) does not
# spin up threads.
_SKY_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_SKY_EXECUTOR_LOCK = threading.Lock()


def _get_sky_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Return the dedicated SkyPilot provisioning thread pool, creating it once.

    :returns: the process-wide ThreadPoolExecutor used for orphanable SkyPilot
        submit/stream calls (double-checked lazy singleton).
    """
    global _SKY_EXECUTOR
    if _SKY_EXECUTOR is None:
        with _SKY_EXECUTOR_LOCK:
            if _SKY_EXECUTOR is None:
                _SKY_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    thread_name_prefix="gb-sky-provision"
                )
    return _SKY_EXECUTOR


async def _sky_submit_to_thread(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run a blocking, possibly-orphaned SkyPilot call on the dedicated pool.

    Mirrors ``asyncio.to_thread`` but targets ``_get_sky_executor()`` instead of
    the loop's shared default executor, so a thread abandoned after a cancel
    (drain timeout) cannot starve unrelated ``to_thread`` work elsewhere in
    gbserver.

    :param func: the blocking SkyPilot SDK call (e.g. ``sky.launch``).
    :param args: positional arguments forwarded to ``func``.
    :param kwargs: keyword arguments forwarded to ``func``.
    :returns: the call's result once the worker thread completes.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_sky_executor(), functools.partial(func, *args, **kwargs)
    )


def _require_skypilot():
    """Raise a clear error if the sky SDK is not installed.

    Pure availability guard — does not start the API server. Callers that
    need the server running should call ``_ensure_skypilot_api_running``.
    """
    if not HAS_SKYPILOT:
        raise ImportError(
            "The 'skypilot' package is required for the Skypilot environment. "
            "Install it with: pip install 'gbserver[skypilot]'"
        )


async def _abort_shielded_request(
    request_id: Any,
    pending_fut: Optional["asyncio.Future"],
    description: str,
    on_abort: Callable[[], Awaitable[None]],
) -> None:
    """Abort an in-flight, shielded SkyPilot request after cancellation.

    Shared by the unmanaged (``_abort_provision``) and managed
    (``_abort_managed_launch``) launchers, which differ only in the final
    reclaim step (``on_abort``): teardown-by-name vs cancel-job-by-name.

    The current task is already flagged for cancellation, so every ``await``
    here would re-raise ``CancelledError`` immediately. We run the reclaim as a
    shielded task and, mirroring ``build/run.py``, clear the pending
    cancellation with ``Task.uncancel()`` so the awaits block normally,
    re-uncancel on each *further* ``CancelledError`` (e.g. a double
    ``gb build cancel``) so a mid-abort cancel can't skip the by-name reclaim,
    and re-arm the cancel in the ``finally`` so it keeps propagating once the
    reclaim is done. The reclaim steps: recover the request_id from the submit
    future if we were cancelled before it returned; tell the API server to abort
    the request (unblocking the shielded thread); drain the still-running thread
    (bounded, so a rejected/ineffective abort does not gate the reclaim on the
    full provisioning time); then run ``on_abort``.

    Args:
        request_id: The id from ``sky.(jobs.)launch``; ``None`` when cancelled
            during the submit (before it returned), then recovered by draining
            ``pending_fut`` so the request can still be aborted server-side.
        pending_fut: The shielded ``to_thread`` future still running on its OS
            thread (the submit or the ``stream_and_get`` wait), drained here.
        description: Resource label used in the log/timeout messages, e.g.
            ``"SkyPilot cluster gb-xyz"``.
        on_abort: Coroutine performing the resource-specific reclaim by the
            deterministic name — the safety net that runs even when there is no
            request_id to abort.
    """

    async def _do_abort() -> None:
        # The reclaim body, run as a shielded task so repeated cancels can't
        # interrupt it before on_abort() (the by-name reclaim) has run.
        local_id, local_fut = request_id, pending_fut
        # Set if a drain below times out: the submit or stream_and_get thread is
        # still running and could create/finish the resource *after* the by-name
        # reclaim returns. Keeping the future lets us retry the reclaim when it
        # settles (closes the post-timeout leak window). At most one of the two
        # drains runs per call, so a single slot suffices.
        orphaned_fut = None
        if local_id is None and local_fut is not None:
            local_id = await Environment._drain_thread_future(
                local_fut, f"{description} submit"
            )
            if local_id is None and not local_fut.done():
                orphaned_fut = local_fut
            local_fut = None  # already drained above
        logger.info(
            "Cancellation requested; aborting %s (request %s)", description, local_id
        )
        if local_id is not None:
            try:
                abort_id = await asyncio.to_thread(sky.api_cancel, local_id)
                await asyncio.to_thread(sky.get, abort_id)
            except Exception as e:
                logger.warning("api_cancel for %s failed: %s", local_id, e)
        if local_fut is not None:
            await Environment._drain_thread_future(local_fut, description)
            if not local_fut.done():
                # Same window on the stream_and_get path: if api_cancel was
                # rejected or ineffective, provisioning continues server-side and
                # can finish after on_abort() returns.
                orphaned_fut = local_fut
        await on_abort()
        if orphaned_fut is not None:
            _reclaim_when_thread_settles(orphaned_fut, description, on_abort)

    abort_task = asyncio.ensure_future(_do_abort())
    current = asyncio.current_task()
    cancels = current.cancelling() if current else 0
    for _ in range(cancels):
        current.uncancel()  # type: ignore[union-attr]
    try:
        while not abort_task.done():
            try:
                await asyncio.shield(abort_task)
            except asyncio.CancelledError:
                # A further cancel arrived mid-abort; suppress it and re-uncancel
                # so the reclaim still runs to completion (matches build/run.py).
                if current and current.cancelling() > 0:
                    cancels += current.cancelling()
                    for _ in range(current.cancelling()):
                        current.uncancel()
    finally:
        if cancels > 0 and current:
            current.cancel()


def _reclaim_when_thread_settles(
    orphaned_fut: "asyncio.Future",
    description: str,
    on_abort: Callable[[], Awaitable[None]],
) -> None:
    """Re-run the by-name reclaim once an orphaned launch/stream thread settles.

    When a drain in ``_abort_shielded_request`` times out, the underlying
    ``sky.(jobs.)launch`` (submit) or ``sky.stream_and_get`` (stream) thread
    keeps running and can create/finish the cluster/job *after* the first by-name
    reclaim has already returned — the exact leak the reclaim exists to prevent,
    with no further attempt. This attaches a done-callback that fires a *second*
    reclaim when the orphaned future settles (i.e. once the resource, if any, has
    actually been created), so it is reaped best-effort. ``on_abort`` is
    idempotent (teardown/cancel-by-name tolerate an already-gone resource), so
    the extra call is safe when nothing leaked.

    Best-effort, with real limits:

    * **Loop already closed.** We only reach here 60s+ after the abort (the drain
      timeout), by which point the build has usually finished. In a one-shot
      ``gb`` invocation the event loop is then closed, so the done-callback never
      fires and *nothing is logged at all* — the resource is leaked silently. The
      ``except RuntimeError`` branch below only catches "no running loop" at
      ``ensure_future`` time, which is close to unreachable (if a callback fires,
      a loop is usually running); it does not cover the loop-already-gone case.
      In long-lived ``gbserver`` the loop outlives the build, so the reclaim does
      fire — this gap mainly affects short-lived CLI processes.
    * **Fire-and-forget.** The second reclaim is scheduled with no strong
      reference retained and its exceptions are never retrieved. This is safe
      only because both ``on_abort`` implementations
      (``_abort_provision._teardown`` and ``_abort_managed_launch._cancel_job``)
      swallow their own errors and complete quickly; a future ``on_abort`` that
      raised, or ran long enough to be GC'd mid-flight, would need a held
      reference and a result-consuming callback here.

    :param orphaned_fut: the still-pending ``to_thread`` submit/stream future.
    :param description: resource label used in the log messages.
    :param on_abort: the idempotent by-name reclaim coroutine to re-run.
    """

    def _on_settled(fut: "asyncio.Future") -> None:
        if fut.cancelled():
            return
        if fut.exception() is not None:
            # The launch itself failed → no resource was created; nothing to reap.
            logger.info(
                "Orphaned %s thread failed after abort timeout (%s); "
                "no second reclaim needed",
                description,
                fut.exception(),
            )
            return
        logger.warning(
            "Orphaned %s thread completed after abort timeout; running a second "
            "by-name reclaim to reap the possibly-created resource",
            description,
        )
        try:
            asyncio.ensure_future(on_abort())
        except RuntimeError as e:
            # No running loop at schedule time. Surface it loudly so the possible
            # leak is actionable rather than silent. (Does not cover the more
            # likely one-shot case where the loop is already closed before this
            # callback fires — see the docstring.)
            logger.error(
                "Could not schedule second reclaim for orphaned %s thread (%s); "
                "a cluster/job may be leaked and need manual teardown",
                description,
                e,
            )

    orphaned_fut.add_done_callback(_on_settled)


async def _run_sky_verb_off_loop(
    verb: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run a blocking sky SDK verb and its ``sky.get`` off the event loop.

    Mutating sky verbs (``sky.down``, ``sky.jobs.cancel``, ``sky.api_cancel``)
    return a request id that must be passed to ``sky.get`` to make the call
    synchronous. Both block, so running them inline freezes the event loop for
    the whole server round-trip and defeats any enclosing ``wait_for``; this
    offloads both to threads. Exceptions propagate — the caller owns error
    handling.

    :param verb: the blocking sky verb to invoke (e.g. ``sky.jobs.cancel``).
    :param args: positional arguments forwarded to ``verb``.
    :param kwargs: keyword arguments forwarded to ``verb``.
    :returns: the result of ``sky.get`` on the verb's request id.
    """
    request_id = await asyncio.to_thread(verb, *args, **kwargs)
    return await asyncio.to_thread(sky.get, request_id)


def _ensure_skypilot_api_running():
    """Start the SkyPilot API server if not already healthy.

    Probes via sky.api_info(); starts the server only if the probe indicates
    the server is unreachable or unhealthy. Other failure modes (auth, config,
    etc.) propagate so they're not masked by an unconditional ``api_start``.
    """
    _require_skypilot()
    try:
        info = sky.api_info()
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.info("SkyPilot API server not reachable (%s) — starting it now", e)
        sky.api_start()
        return
    if info.status.value != "healthy":
        logger.info(
            "SkyPilot API server status=%s — starting it now", info.status.value
        )
        sky.api_start()


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, max=128),
    reraise=True,
)
def _download_logs_with_retry(cluster_name: str, job_id: int):
    """Download SkyPilot job logs with retry for transient failures."""
    # sky.download_logs() returns Dict[str, str] mapping job_id to local log path
    # (it handles the API request/response internally, no sky.get() needed)
    result = sky.download_logs(cluster_name, job_ids=[str(job_id)])
    return result.get(str(job_id))


# Upper bound for how long monitor_skypilot_monitor waits for retry_workload to
# finish a teardown+relaunch before treating the step as failed. Generous: must
# comfortably exceed real relaunch time (provision-retry backoff + cloud
# provisioning). Purely defensive — retry_workload sets the complete event in a
# finally, so this should never actually trip.
RETRY_RELAUNCH_TIMEOUT_SECONDS = 1800

# SSH-provisioned bare HPC schedulers. They don't support SkyPilot autostop, and
# commonly don't track memory as a consumable resource (RealMemory unset in
# slurm.conf), so a --memory request fails resource matching. Cloud-specific
# handling groups them: skip the compute_config memory floor and force autostop
# off. Compared against the normalized first infra segment (lowercased).
_SSH_HPC_CLOUDS = ("slurm", "lsf")

# Per-step log-retrieval modes, selected via the ``log_retrieval.mode`` key in
# the skypilot_monitor config. See _parse_log_retrieval for semantics.
LOG_RETRIEVAL_ON_COMPLETION = "on_completion"
LOG_RETRIEVAL_PERIODIC = "periodic"
LOG_RETRIEVAL_STARTUP_WINDOW = "startup_window"
LOG_RETRIEVAL_STREAM = "stream"
_LOG_RETRIEVAL_MODES = frozenset(
    {
        LOG_RETRIEVAL_ON_COMPLETION,
        LOG_RETRIEVAL_PERIODIC,
        LOG_RETRIEVAL_STARTUP_WINDOW,
        LOG_RETRIEVAL_STREAM,
    }
)
_DEFAULT_STARTUP_WINDOW_SECONDS = 120.0


def _coerce_float(value, default: float) -> float:
    """Best-effort float coercion (templated configs may pass strings)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_log_retrieval(
    kwargs: dict, poll_interval: float
) -> Tuple[str, float, float]:
    """Resolve the log-retrieval policy from a monitor config block.

    Reads the optional ``log_retrieval`` dict from the monitor config kwargs and
    returns ``(mode, interval_seconds, startup_window_seconds)``:

    - ``on_completion`` (default): pull the full log once, at terminal status.
    - ``periodic``: pull incrementally every ``interval_seconds`` while RUNNING
      (defaults to ``poll_interval``), plus a final pull at terminal.
    - ``startup_window``: pull periodically only for the first
      ``startup_window_seconds`` after the job goes RUNNING, then stop (still
      pulls once at terminal).
    - ``stream``: real-time ``sky.tail_logs`` follow stream (heaviest; opt-in).

    An unknown mode warns and falls back to ``on_completion``.
    """
    block = kwargs.get("log_retrieval") or {}
    if not isinstance(block, dict):
        logger.warning(
            "log_retrieval config is not a mapping (%r); using %s",
            block,
            LOG_RETRIEVAL_ON_COMPLETION,
        )
        block = {}
    mode = str(block.get("mode", LOG_RETRIEVAL_ON_COMPLETION))
    if mode not in _LOG_RETRIEVAL_MODES:
        logger.warning(
            "Unknown log_retrieval mode %r; falling back to %s",
            mode,
            LOG_RETRIEVAL_ON_COMPLETION,
        )
        mode = LOG_RETRIEVAL_ON_COMPLETION
    interval_seconds = _coerce_float(block.get("interval_seconds"), poll_interval)
    startup_window_seconds = _coerce_float(
        block.get("startup_window_seconds"), _DEFAULT_STARTUP_WINDOW_SECONDS
    )
    return mode, interval_seconds, startup_window_seconds


def _effective_poll_timeout(
    poll_interval: float,
    log_mode: str,
    log_interval: float,
    pulls_active: bool,
) -> float:
    """How long the poll loop should sleep before its next wake-up.

    Status polling runs every ``poll_interval`` (often long, e.g. 900s), but
    periodic/startup_window log pulls must fire on their own (usually shorter)
    ``interval_seconds``. The single poll loop drives both, so while pulls are
    active the loop must wake at the *minimum* of the two cadences — otherwise a
    900s status poll would starve a 15s log-pull schedule (the startup-window
    binding scrape would only get one shot right after RUNNING). Once pulls stop
    (window elapsed, or non-pull mode), fall back to the status cadence.
    """
    if pulls_active and log_mode in (
        LOG_RETRIEVAL_PERIODIC,
        LOG_RETRIEVAL_STARTUP_WINDOW,
    ):
        return min(poll_interval, log_interval)
    return poll_interval


# Substrings that mark a transient resource-acquisition / provision failure.
# Conservative: drawn from observed SkyPilot/slurm failover messages. Anything
# else (auth, image-not-found, NotSupported, config, quota-denied) is treated as
# fatal and re-raised immediately so a genuine launch failure is never masked.
_TRANSIENT_PROVISION_SUBSTRINGS = (
    "failed to provision",  # "Failed to provision all possible launchable resources"
    "failed to acquire resources",  # slurm: "Failed to acquire resources in normal for ..."
    "resources unavailable",
    "in normal for",  # slurm partition acquisition failure tail
)

_NON_TRANSIENT_PROVISION_SUBSTRINGS = (
    "catalog does not contain",  # no matching instance type exists — config error
    "no launchable resource",  # similar permanent mismatch
)


def _is_transient_provision_error(exc: BaseException) -> bool:
    """Return True if exc is a retriable resource-acquisition/provision failure.

    The primary signal is the SkyPilot exception *type*; the substring scan is a
    conservative fallback for SDK builds that surface the failure as a plain
    Exception. Non-provision failures (auth, image-not-found, config, etc.)
    return False so they propagate without masking.

    Permanent configuration errors (e.g. "Catalog does not contain any
    instances") are excluded even when they raise ResourcesUnavailableError,
    since retrying will never succeed.

    Args:
        exc: The exception raised by the provisioning step.

    Returns:
        bool: True if the failure looks transient and worth retrying.
    """
    text = str(exc).lower()
    if any(s in text for s in _NON_TRANSIENT_PROVISION_SUBSTRINGS):
        return False
    if sky is not None:
        exc_types = tuple(
            t
            for t in (
                getattr(sky.exceptions, "ResourcesUnavailableError", None),
                getattr(sky.exceptions, "ResourcesMismatchError", None),
                getattr(sky.exceptions, "ProvisionPrechecksError", None),
            )
            if isinstance(t, type)
        )
        if exc_types and isinstance(exc, exc_types):
            return True
    return any(s in text for s in _TRANSIENT_PROVISION_SUBSTRINGS)


from gbserver.environment._skypilot_ssh import (
    execute_on_host_via_ssh as _execute_on_host_via_ssh,
)
from gbserver.environment._skypilot_ssh import (
    extract_host_ssh_info as _extract_host_ssh_info,
)


def _escapes_parent(rel_path: str) -> bool:
    """Return whether a *relative* path climbs out of its base via ``..``.

    ``normpath`` collapses a leading ``./`` and inner ``.``/``..`` segments; a
    relative path that escapes its base always normalizes to a ``..``-leading
    result (``..``, ``../x``, ``a/../../b`` -> ``../b``), whereas one that stays
    inside never does (``a/../b`` -> ``b``). Shared by the ``file_mounts`` source
    and destination guards so both reject the same escaping paths.

    :param rel_path: a relative path (callers exclude absolute/URI/``~`` inputs).
    :returns: ``True`` if it would resolve outside its base directory.
    """
    rel = os.path.normpath(rel_path)
    return rel == ".." or rel.startswith(".." + os.sep)


def _reject_home_prefixed(path: str, role: str) -> None:
    """Reject a ``~``/``~/``-prefixed ``file_mounts`` path.

    This launcher never expands ``~``: a ``~`` *source* would resolve to a literal
    ``~`` directory under the step dir, and a ``~`` *destination* would sidestep
    the single relative/absolute destination convention (it lands outside the
    per-run workdir and, on containerized LSF, outside the container). Both roles
    reject it so ``file_mounts`` has one consistent path model. Shared by the
    source and destination guards.

    :param path: the source or destination path from a ``file_mounts`` entry.
    :param role: ``"source"`` or ``"destination"`` — named in the error message.
    :raises ValueError: if ``path`` equals ``~`` or begins with ``~/``.
    """
    if path == "~" or path.startswith("~/"):
        raise ValueError(
            f"file_mounts {role} {path!r} uses '~', which is not expanded; "
            f"use a relative or absolute path instead"
        )


def _resolve_local_mount_source(source: str, asset_dir: Union[Path, str, None]) -> str:
    """Resolve a ``file_mounts`` local source against the step's asset dir.

    Remote URIs (``s3://``, ``gs://``, ``file://``, ``http…``) and absolute paths
    are returned unchanged. A relative local path is joined onto ``asset_dir`` —
    the per-run directory holding the rendered ``step.yaml`` and its sibling
    files — so a path written in ``step.yaml`` is interpreted relative to the
    ``step.yaml``'s own location (matching how bash/k8s treat step-relative
    assets).

    A ``~``/``~/``-prefixed source is rejected: this launcher resolves relative
    sources against the step dir and never expands ``~`` for sources, so it would
    otherwise become a literal ``<asset_dir>/~/…`` path rather than a home dir.
    A relative source that uses ``..`` to climb out of the step dir (e.g.
    ``../other``) is also rejected, so sources stay confined to the step's own
    assets. Use an absolute path or a step-relative one instead.

    :param source: the local/remote source string from a ``file_mounts`` entry.
    :param asset_dir: ``targetsteprun_asset_dir`` (a ``Path`` or ``file://``
        string), or ``None`` when unavailable (e.g. a retry with no stashed dir).
    :returns: the resolved source string (unchanged for URIs and absolute paths).
    :raises ValueError: if ``source`` is ``~``/``~/``-prefixed or escapes the
        step dir via ``..``.
    """
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme:  # remote URI (s3/gs/file/http/…) — leave as-is
        return source
    if os.path.isabs(source):
        return source  # absolute host path — author's explicit choice
    _reject_home_prefixed(source, "source")
    if _escapes_parent(source):
        raise ValueError(
            f"file_mounts source {source!r} uses '..' to escape the step "
            f"directory; use a path inside the step directory or an absolute "
            f"source"
        )
    if asset_dir is None:
        logger.warning(
            "Relative file_mount source %r but no asset dir available; "
            "leaving it unresolved",
            source,
        )
        return source
    # Tolerate a file:// URI form for asset_dir, matching the bash launcher.
    base = Path(urllib.parse.urlparse(str(asset_dir)).path)
    return str(base / source)


def _remap_relative_dest(dst: str, build_workdir: Optional[str]) -> str:
    """Map a relative ``file_mounts`` destination into the per-run workdir.

    Relative destinations (e.g. ``payload``, ``./payload``, ``sub/payload``) are
    rewritten to ``${build_workdir}/<dst>`` — an absolute path on the shared
    filesystem — so the payload is reachable at exactly ``./<dst>`` from the run
    script's CWD (``$GB_BUILD_WORKDIR``), giving implicit per-target isolation.

    On the LSF/enroot backend the shared ``/proj`` tree is bind-mounted identity
    into the step container, so a payload written to ``${build_workdir}`` on the
    (sudo-less) login node is visible to the job at the same path; the SkyPilot
    backend's symlink-wrap is exempted for these shared roots (see the fork's
    ``sky/provision/lsf`` runner hook), so no container staging or copy-back is
    needed.

    Absolute destinations pass through unchanged (the author's explicit fixed
    location). When ``build_workdir`` is unset (envs without ``shared_workdir``) a
    relative destination is likewise returned unchanged, preserving SkyPilot's own
    ``~/sky_workdir/`` rewrite.

    A ``~``/``~/``-prefixed destination is rejected, mirroring the source guard in
    :func:`_resolve_local_mount_source`: ``~`` is never expanded by this launcher,
    so ``file_mounts`` has a single destination convention — relative, or absolute
    for a fixed location. A relative destination that uses ``..`` to climb out of
    its target directory (e.g. ``../foo``) is likewise rejected — whether or not a
    remap applies — so it can leave neither the per-run workdir nor SkyPilot's
    default rewrite. These are authoring guards, not a security boundary; the step
    author controls the destination.

    :param dst: the destination key from the raw ``file_mounts`` mapping.
    :param build_workdir: absolute per-run workdir, or ``None`` to disable remap.
    :returns: the (possibly rewritten) destination path.
    :raises ValueError: if ``dst`` is ``~``/``~/``-prefixed, or a relative ``dst``
        escapes its target directory via ``..`` traversal.
    """
    _reject_home_prefixed(dst, "destination")
    if os.path.isabs(dst):
        return dst  # absolute: author's explicit fixed location, left as-is
    # Reject ``..`` escapes regardless of whether a build_workdir remap follows,
    # so the destination can leave neither the per-run workdir nor SkyPilot's
    # default rewrite.
    if _escapes_parent(dst):
        raise ValueError(
            f"file_mounts destination {dst!r} uses '..' to escape its target "
            f"directory; use a path without a leading '..' or an absolute "
            f"destination"
        )
    if not build_workdir:
        return dst  # no shared workdir: leave to SkyPilot's default handling
    return os.path.normpath(os.path.join(build_workdir, dst))


def _get_cli_prefix(build_workdir: Optional[str]) -> str:
    """Build the shell snippet prepended to each step's setup and run scripts.

    The snippet always leads with ``set -eu`` so any failure in the prefix aborts
    before the step body runs, rather than silently executing the body in a wrong
    or unexpected state. The prefix is prepended ahead of each body's own
    ``set -eu``, so without this the body's flags would not yet be in effect while
    the prefix runs.

    When a ``build_workdir`` was provisioned (the env configures
    ``shared_workdir``), the snippet also emits ``mkdir -p`` + ``cd
    "$GB_BUILD_WORKDIR"`` so both the ``setup`` and ``run`` scripts start in that
    per-run workdir. Making the launcher own the ``cd`` lets step authors write
    outputs with relative paths and stay agnostic about where the step runs: they
    never need to reference ``$GB_BUILD_WORKDIR`` themselves. With ``set -eu`` in
    front, a failing ``mkdir``/``cd`` (or an unset ``$GB_BUILD_WORKDIR``) aborts
    fast instead of leaving the body running in the wrong directory.

    When no ``build_workdir`` is set (envs without ``shared_workdir``), no ``cd``
    is emitted — only ``set -eu``. SkyPilot then runs the scripts in its own
    default working directory (``~/sky_workdir``), which is exactly where its
    relative ``file_mounts`` rewrite places payloads (see ``_remap_relative_dest``,
    which leaves relative destinations untouched in this case). Injecting a ``cd``
    elsewhere (e.g. ``$HOME``) would move the CWD away from the mounted payloads,
    so relative-in/relative-out steps would fail to find them; not cd'ing keeps
    the run CWD aligned with the mount location.

    :param build_workdir: the provisioned per-run workdir path, or ``None`` when
        no ``shared_workdir`` is configured (no ``cd`` is emitted).
    :returns: a shell snippet terminated by a trailing newline; always at least
        ``set -eu``, plus ``mkdir``/``cd`` when a ``build_workdir`` is set.
    """
    prefix = "set -eu\n"
    if build_workdir:
        prefix += 'mkdir -p "$GB_BUILD_WORKDIR"\ncd "$GB_BUILD_WORKDIR"\n'
    return prefix


def _build_skypilot_mounts(
    file_mounts_raw: dict,
    asset_dir: Union[Path, str, None],
    build_workdir: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Split a raw ``file_mounts`` mapping into file mounts and storage mounts.

    String values are local-to-remote copies (``Task.set_file_mounts``); dict
    values (``{source, mode}``) become ``sky.Storage`` storage mounts
    (``Task.set_storage_mounts``). Relative local sources are resolved via
    :func:`_resolve_local_mount_source`; bucket URIs keep the existing sub-path
    extraction (``MOUNT`` mode requires a bucket-only source). When
    ``build_workdir`` is given, relative *destinations* are remapped under it via
    :func:`_remap_relative_dest`.

    :param file_mounts_raw: the raw ``file_mounts`` mapping from the config.
    :param asset_dir: ``targetsteprun_asset_dir`` used to resolve relative sources.
    :param build_workdir: per-run workdir for relative-destination remap, or
        ``None`` to leave destinations unchanged.
    :returns: a ``(file_mounts, storage_mounts)`` tuple of dicts, either of which
        may be empty.
    """
    file_mounts: Dict[str, str] = {}
    storage_mounts: Dict[str, Any] = {}
    for raw_path, mount_val in file_mounts_raw.items():
        mount_path = _remap_relative_dest(raw_path, build_workdir)
        if isinstance(mount_val, dict):
            source = mount_val["source"]
            storage_kwargs: Dict[str, Any] = {
                "mode": sky.StorageMode[mount_val.get("mode", "MOUNT").upper()],
            }
            parsed = urllib.parse.urlparse(source)
            if parsed.scheme:  # bucket URI: extract the bucket-only source
                sub_path = parsed.path.lstrip("/")
                if sub_path:
                    storage_kwargs["source"] = f"{parsed.scheme}://{parsed.netloc}"
                    storage_kwargs["_bucket_sub_path"] = sub_path
                else:
                    storage_kwargs["source"] = source
            else:  # local path: resolve relative to the step.yaml dir
                storage_kwargs["source"] = _resolve_local_mount_source(
                    source, asset_dir
                )
            storage_mounts[mount_path] = sky.Storage(**storage_kwargs)
        else:
            file_mounts[mount_path] = _resolve_local_mount_source(mount_val, asset_dir)
    return file_mounts, storage_mounts


def aws_credentials_present() -> bool:
    """Return True when boto3 would resolve AWS credentials from the environment.

    Checks the credential environment variables boto3 reads: an explicit access
    key pair (``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY``) or a named
    profile (``AWS_PROFILE``). It does not validate the credentials — only that
    boto3 has something to try. Useful for gating operations (and tests) that
    require a real AWS backend so they can be skipped cleanly when no
    credentials are configured.

    :returns: True if AWS credential env vars are set, False otherwise.
    """
    has_key_pair = bool(os.environ.get("AWS_ACCESS_KEY_ID")) and bool(
        os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    return has_key_pair or bool(os.environ.get("AWS_PROFILE"))


class Skypilot(Environment):
    """SkyPilot environment — provisions pods/VMs for step execution (unmanaged)."""

    # Class-level semaphore so the cap applies across all Skypilot
    # instances within a process — for fan-out builds gbserver creates
    # one Environment per target, but they all share the SSH connection
    # pool to the cloud's login node and therefore the same MaxAuthTries
    # ceiling. Lazily constructed so no event loop is required at import.
    #
    # This MUST be a threading.Semaphore, not asyncio.Semaphore: in the
    # standalone (thread) build-runner each target runs in its own thread
    # under its own asyncio.run() event loop, and an asyncio primitive is
    # bound to the loop that first touches it — sharing one across target
    # loops raises "bound to a different event loop". A threading.Semaphore
    # is loop-agnostic and genuinely caps across the target threads.
    _launch_semaphore: Optional[threading.Semaphore] = None
    _launch_semaphore_lock = threading.Lock()

    # Cluster names we deliberately tore down (e.g. via
    # launch_skypilot_teardown downing a SERVICE). A SERVICE's monitor must
    # treat its cluster vanishing as SUCCESS rather than a crash. This is
    # CLASS-level (process-global) on purpose: gbserver creates a separate
    # Skypilot instance per target, so the teardown target and the monitored
    # SERVICE targets do not share instance state — but they do share this
    # set within the process, keyed by the globally-unique cluster name.
    _intentionally_torn_down_clusters: set = set()

    @classmethod
    def _get_launch_semaphore(cls) -> threading.Semaphore:
        if cls._launch_semaphore is None:
            with cls._launch_semaphore_lock:
                # Double-checked under the lock so concurrent target threads
                # don't each construct a separate semaphore (which would
                # defeat the process-global cap).
                if cls._launch_semaphore is None:
                    from gbserver.types.constants import (
                        GBSERVER_SKYPILOT_LAUNCH_CONCURRENCY,
                    )

                    cls._launch_semaphore = threading.Semaphore(
                        max(1, GBSERVER_SKYPILOT_LAUNCH_CONCURRENCY)
                    )
        return cls._launch_semaphore

    def __init__(
        self: Self,
        event_q: asyncio.Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        secrets: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self._cluster_names: Dict[str, str] = {}  # launch_id -> cluster_name
        self._job_ids: Dict[str, int] = {}  # launch_id -> sky job_id
        # launch_id -> relaunch attempt number. 0 (or absent) is the initial
        # launch; retry_workload bumps it so each relaunch provisions a fresh,
        # uniquely-named cluster instead of reusing the draining original.
        self._relaunch_attempts: Dict[str, int] = {}
        self._setup_workdirs: Dict[str, str] = {}  # setup_id -> per-run workdir
        # launch_id -> kwargs replayed by retry_workload
        self._launch_kwargs: Dict[str, Dict] = {}
        self._skypilot_retry_complete_events: Dict[str, asyncio.Event] = {}
        # launch_id -> set the instant retry_workload begins (before stop_event),
        # so monitor_skypilot_monitor can distinguish a retry-induced poll stop
        # from a terminal completion and await the (possibly slow) relaunch
        # instead of racing it.
        self._skypilot_retry_in_progress_events: Dict[str, asyncio.Event] = {}
        # Guard so inline config (cluster_ssh_configs / cloud_config /
        # aws_credentials) is materialized at most once per environment
        # instance; the merge itself is also idempotent, so retries are free.
        self._inline_configs_done: bool = False
        # launch_id -> highest 1-based log line number already parsed, so a
        # periodic/startup pull resumes after the lines it last emitted events
        # for instead of re-emitting from the top each time.
        self._log_lines_parsed: Dict[str, int] = {}
        super().__init__(
            event_q=event_q,
            environment_config=environment_config,
            secrets=secrets,
            **kwargs,
        )

    def _ensure_inline_configs_materialized(self: Self) -> None:
        """Materialize inline SkyPilot config from environment.yaml (once).

        Reads ``cluster_ssh_configs`` / ``cloud_config`` / ``aws_credentials``
        from the environment's free-form ``config`` block and delegates to
        ``skypilot_config.materialize``, which writes/merges the SSH config
        files, the per-request ``SKYPILOT_PROJECT_CONFIG`` override, and
        ``~/.aws/credentials``. No-op when none are present. Idempotent via an
        instance flag (and the merge itself), so retry relaunches are free.

        :raises SkypilotConfigCollisionError: If a section conflicts with config
            already materialized by another environment in this process/host.
        """
        if self._inline_configs_done:
            return
        cfg = self.config.config if self.config else {}
        ssh_raw = cfg.get("cluster_ssh_configs")
        cloud_config = cfg.get("cloud_config")
        aws_raw = cfg.get("aws_credentials")
        if ssh_raw or cloud_config or aws_raw:
            from gbserver.environment.skypilot_config import materialize
            from gbserver.types.environmentconfig import (
                AwsCredentialProfile,
                ClusterSshConfigs,
            )

            ssh = ClusterSshConfigs.model_validate(ssh_raw) if ssh_raw else None
            aws = (
                [AwsCredentialProfile.model_validate(p) for p in aws_raw]
                if aws_raw
                else None
            )
            name = self.config.name if self.config else "unknown"
            materialize(name, ssh, cloud_config, aws, self.secrets or {})
        self._inline_configs_done = True

    def _get_cloud(self: Self) -> str:
        """Get default cloud/infra from environment.yaml config."""
        if self.config is None:
            return "k8s"
        return self.config.config.get("default_cloud", "k8s")

    def _get_idle_minutes(self: Self) -> int:
        """Get idle_minutes_to_autostop from environment.yaml config."""
        if self.config is None:
            return 10
        return self.config.config.get("idle_minutes_to_autostop", 10)

    @staticmethod
    def _cluster_name_for(launch_id: str, attempt: int = 0) -> str:
        """Generate a unique cluster name from a launch_id.

        :param launch_id: The launch identifier the cluster belongs to.
        :param attempt: Relaunch attempt number. ``0`` (the initial launch)
            yields the bare ``gb-<launch_id>`` name for backward compatibility;
            ``> 0`` appends an ``-r<attempt>`` suffix so a retry provisions a
            distinct cluster/allocation instead of colliding with the original
            that may still be draining on the backend (slurm/lsf).
        :returns: The deterministic cluster name for this launch + attempt.
        """
        base = f"gb-{launch_id[:12]}"
        return base if attempt <= 0 else f"{base}-r{attempt}"

    async def setup_skypilot(
        self: Self,
        setup_id: str,
        runmetadata: EntityRunMetadata,
        **kwargs,
    ) -> Dict:
        """Compute the per-run workdir path and publish it to step launches.

        When the env config defines ``shared_workdir``, derive a path under
        ``${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/`` and
        return it as ``setup_config.skypilot.build_workdir`` so
        ``launch_skypilot`` can export ``GB_BUILD_WORKDIR`` and ``cd`` into
        it. The path is also stashed on ``self._setup_workdirs`` so
        ``teardown_skypilot`` can locate it (``runmetadata`` is not
        forwarded to teardown).

        :param setup_id: Setup identifier minted by ``Environment.setup``.
        :param runmetadata: Run metadata injected by ``Run._add_to_run_kwargs``.
        :returns: Setup config dict (empty when ``shared_workdir`` is unset).
        """
        # Materialize inline config as early as possible (before the
        # shared_workdir early-return below).
        self._ensure_inline_configs_materialized()
        shared_workdir = (
            self.config.config.get("shared_workdir") if self.config else None
        )
        if not shared_workdir:
            return {}
        workdir = os.path.join(
            shared_workdir,
            "builds",
            runmetadata.build_id or "",
            "runs",
            runmetadata.targetrun_id or "",
        )
        self._setup_workdirs[setup_id] = workdir
        logger.info(
            "setup_skypilot: per-run workdir for setup_id=%s -> %s",
            setup_id,
            workdir,
        )
        return {"skypilot": {"build_workdir": workdir}}

    async def teardown_skypilot(self: Self, setup_id: str, **kwargs) -> None:
        """Remove the per-run workdir provisioned by ``setup_skypilot``.

        Submits a one-shot ``sky launch`` whose run script ``rm -rf``s the
        per-run workdir. Failures are logged and swallowed — a stale
        workdir is not worth failing the build for, and the build has
        already finished by the time teardown runs.

        :param setup_id: Setup identifier originally returned to
            ``Environment.setup``; used to look up the stashed path.
        """
        workdir = self._setup_workdirs.pop(setup_id, None)
        if not workdir:
            return
        _require_skypilot()
        cluster_name = self._cluster_name_for(f"td-{setup_id}")
        logger.info(
            "teardown_skypilot: removing per-run workdir %s (setup_id=%s)",
            workdir,
            setup_id,
        )
        try:
            task = sky.Task(
                name=cluster_name,
                run=f"rm -rf {shlex.quote(workdir)}",
                resources=sky.Resources(infra=self._get_cloud()),
            )
            request_id = await asyncio.to_thread(
                sky.launch,
                task,
                cluster_name=cluster_name,
                idle_minutes_to_autostop=0,
                down=True,
            )
            await asyncio.to_thread(sky.stream_and_get, request_id)
        except Exception as e:  # don't fail the build for cleanup
            logger.warning("teardown_skypilot rm -rf %s failed: %s", workdir, e)

    @staticmethod
    def _parse_memory_gib(memory_str: str) -> Optional[float]:
        """Convert a ``total_memory_per_node`` string to a GiB number for
        ``sky.Resources(memory=...)``.

        SkyPilot treats ``memory`` as a GB number (or string). We map common
        Kubernetes/plain suffixes to a bare number, treating ``Gi`` as GB to
        match docker's :meth:`Docker._parse_memory` convention.

        :param memory_str: e.g. ``"1Gi"``, ``"512Mi"``, ``"4G"``, ``"4GB"``,
            ``"4"``. Empty string means "unset".
        :returns: the size in GiB (e.g. ``1.0``, ``0.5``, ``4.0``), or ``None``
            when ``memory_str`` is empty or cannot be parsed as a number.
        """
        if not memory_str:
            return None
        text = memory_str.strip()
        for suffix, factor in (
            ("Gi", 1.0),
            ("G", 1.0),
            ("GB", 1.0),
            ("Mi", 1.0 / 1024),
            ("M", 1.0 / 1024),
        ):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                try:
                    return float(text) * factor
                except ValueError:
                    return None
        try:
            return float(text)
        except ValueError:
            return None

    def _resources_from_compute_config(
        self: Self, compute_config: Dict, cloud: str = ""
    ) -> Dict:
        """Derive a ``sky.Resources`` floor from a step's ``compute_config``.

        Reads the raw dict (NOT the :class:`ComputeConfig` model, whose defaults
        of 8 GPUs / 512G memory would over-size a bare command). Only emits keys
        that are explicitly and validly set, so the caller can layer this as the
        lowest-precedence floor. Both ``cpus`` and ``memory`` are emitted as
        SkyPilot minimums (``"{n}+"``) so catalog matching selects the smallest
        instance with *at least* that much CPU/RAM — a bare number is an EXACT
        match no cloud catalog satisfies for odd sizes ("Catalog does not contain
        any instances satisfying the request: 1x AWS(mem=1.0)" / ``cpus=3``). The
        ``"+"`` form crashes SkyPilot's LSF cloud, so on slurm/lsf ``cpus`` stays
        a bare int (those schedulers match CPUs directly, not via a cloud
        catalog) and ``memory`` is skipped entirely (below).

        :param compute_config: the step's ``config.compute_config`` dict.
        :param cloud: the normalized target cloud (lowercased first infra
            segment, e.g. ``"slurm"``, ``"lsf"``, ``"k8s"``).
        :returns: a dict optionally containing ``cpus`` (a ``"{n}+"`` minimum on
            cloud catalogs, or a bare int on slurm/lsf, when ``num_cpus_per_node``
            > 0) and ``memory`` (a ``"{n}+"`` GiB-minimum string, when
            ``total_memory_per_node`` parses and the cloud is not slurm/lsf).
        """
        resources: Dict = {}
        num_cpus = compute_config.get("num_cpus_per_node", 0)
        if isinstance(num_cpus, int) and num_cpus > 0:
            # Same exact-match trap as memory (below): a bare number is an EXACT
            # request and no cloud catalog has an instance with, e.g., exactly 3
            # vCPUs, so SkyPilot dies with "Catalog does not contain any
            # instances satisfying the request". Emit a minimum ("{n}+") so it
            # picks the smallest instance with at least that many vCPUs. slurm/lsf
            # keep the bare int: the "+" form crashes the fork's LSF cloud, and
            # those schedulers match CPUs directly, so an exact request is fine.
            if cloud in _SSH_HPC_CLOUDS:
                resources["cpus"] = num_cpus
            else:
                resources["cpus"] = f"{num_cpus}+"
        # SLURM/LSF (bare HPC schedulers) commonly don't track memory as a
        # consumable resource (RealMemory unset in slurm.conf), so a --memory
        # request fails at resource matching ("Catalog does not contain any
        # instances satisfying ..."). Skip the compute_config memory floor for
        # them; an explicit launcher/build resources.memory still applies (and
        # works on clusters that do configure memory).
        if cloud not in _SSH_HPC_CLOUDS:
            memory = self._parse_memory_gib(
                compute_config.get("total_memory_per_node", "")
            )
            if memory is not None:
                # Emit as a minimum ("{n}+"), not an exact float: an exact
                # memory=1.0 matches no cloud instance type (SkyPilot dies with
                # "Catalog does not contain any instances satisfying ...
                # AWS(mem=1.0)"). Format without an exponent and strip the
                # trailing ".0"/zeros (1.0 -> "1+", 0.5 -> "0.5+"); ":g" would
                # switch large values to scientific notation (e.g. "1e+06+")
                # which SkyPilot cannot parse. Safe here: slurm/lsf excluded.
                mem_str = f"{memory:f}".rstrip("0").rstrip(".")
                resources["memory"] = f"{mem_str}+"
        return resources

    @staticmethod
    def _first_hf_token(bindings: Optional[Dict]) -> Optional[str]:
        """Return the first HF token found among inline hfpull bindings.

        :param bindings: the launch bindings mapping (may be None); each value
            may carry an ``_hfpull`` dict with an optional ``hf_token``.
        :returns: the first non-empty ``hf_token``, or None if none present.
        """
        for bval in (bindings or {}).values():
            if isinstance(bval, dict) and "_hfpull" in bval:
                token = bval["_hfpull"].get("hf_token")
                if token:
                    return token
        return None

    def get_launch_env_vars(
        self: Self,
        run_metadata: Optional[Dict[str, Any]] = None,
        launcher_config: Optional[Dict] = None,
        config: Optional[Dict] = None,
        launch_id: str = "",
        cluster_name: str = "",
        build_workdir: Optional[str] = None,
        bindings: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Build the full env dict for a skypilot step launch.

        Precedence (lowest->highest): secrets < launcher ``envs`` <
        ``config.launcher_config.envs`` < the built-in
        ``GB_SKYPILOT_*``/workdir/HF_TOKEN vars < the standard cross-environment
        set from ``super()`` (GBTEST_ test-control vars + e.g. GB_BUILD_ID),
        which is authoritative.

        :param run_metadata: launch run_metadata; forwarded to ``super()`` and
            the source of GB_TARGETRUN_ID.
        :param launcher_config: step.yaml launcher config (its ``envs``).
        :param config: full step config (``config.launcher_config.envs`` is
            picked up for auto-queued steps).
        :param launch_id: unique id for this launch (GB_SKYPILOT_LAUNCH_ID).
        :param cluster_name: the sky cluster name (GB_SKYPILOT_CLUSTER_NAME).
        :param build_workdir: per-run workdir (GB_BUILD_WORKDIR) if provisioned.
        :param bindings: launch bindings; scanned for an inline HF_TOKEN.
        :returns: the complete ``{name: value}`` env dict for the sky.Task.
        """
        launcher_config = launcher_config or {}
        config = config or {}
        run_metadata = run_metadata or {}
        env: Dict[str, str] = {}
        if self.secrets:
            env.update(self.secrets)
        env.update(launcher_config.get("envs", {}))
        env.update(config.get("launcher_config", {}).get("envs", {}))
        env["GB_SKYPILOT_LAUNCH_ID"] = launch_id
        env["GB_SKYPILOT_CLUSTER_NAME"] = cluster_name
        if run_metadata.get("targetrun_id"):
            env["GB_TARGETRUN_ID"] = run_metadata["targetrun_id"]
        shared_workdir = (
            self.config.config.get("shared_workdir") if self.config else None
        )
        if shared_workdir:
            env["GB_SHARED_WORKDIR"] = shared_workdir
        if build_workdir:
            env["GB_BUILD_WORKDIR"] = build_workdir
        hf_token = self._first_hf_token(bindings)
        if hf_token and "HF_TOKEN" not in env:
            env["HF_TOKEN"] = hf_token
        env.update(super().get_launch_env_vars(run_metadata=run_metadata))
        # Uniform with the other environments; a no-op here since skypilot's
        # launcher vars are already GB_-prefixed (GB_SKYPILOT_*), so there are no
        # LLMB_ names to mirror.
        return self._add_gb_aliases(env)

    async def launch_skypilot(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir=None,
        environment_config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ) -> None:
        """Launch a step on a SkyPilot cluster (unmanaged).

        Creates a sky.Task from step config, calls sky.launch() to provision
        pods/VMs, waits until the job starts, then signals launch readiness via release_monitors().

        Concurrency: the cluster bring-up is gated by a class-level
        semaphore (see ``GBSERVER_SKYPILOT_LAUNCH_CONCURRENCY``, default
        4). Each launch opens a fresh SSH session to the cloud's login
        node; LSF backends in particular trip sshd MaxAuthTries when
        many evals fan out at once. Capping the in-flight count keeps
        the SSH multiplexer from rejecting bring-ups with "Too many
        authentication failures". The cap wraps the whole bring-up
        (sky.launch → wait until job starts) but releases before
        ``monitor_skypilot_monitor`` runs, so post-launch polling for
        all targets continues in parallel.
        """
        # Acquire without blocking the event loop: threading.Semaphore.acquire
        # is a blocking call, so poll it non-blockingly and yield to the loop
        # between attempts. This lets the target's post-launch monitor polling
        # (and everything else on this loop) keep running while we wait for a
        # bring-up slot.
        sem = self._get_launch_semaphore()
        while not sem.acquire(blocking=False):
            await asyncio.sleep(0.5)
        try:
            await self._launch_skypilot_inner(
                launch_id=launch_id,
                targetsteprun_asset_dir=targetsteprun_asset_dir,
                environment_config=environment_config,
                **kwargs,
            )
        finally:
            sem.release()

    async def _launch_skypilot_inner(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir=None,
        environment_config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ) -> None:
        """Body of launch_skypilot. Split out so the semaphore wrapper in
        ``launch_skypilot`` doesn't have to re-indent the entire block.
        """
        try:
            # Materialize inline cluster/cloud/AWS config before the API server
            # starts and before sky.launch builds the per-request config override.
            self._ensure_inline_configs_materialized()
            _ensure_skypilot_api_running()

            # Stash kwargs so retry_workload can replay this launch. Include
            # targetsteprun_asset_dir so a relaunch re-resolves relative
            # file_mounts sources (it is a named param, so it is replayed via
            # launch_skypilot(launch_id, **original_kwargs)).
            self._launch_kwargs[launch_id] = {
                "targetsteprun_asset_dir": targetsteprun_asset_dir,
                "launcher_config": kwargs.get("launcher_config"),
                "config": kwargs.get("config"),
                "run_metadata": kwargs.get("run_metadata"),
                "setup_config": kwargs.get("setup_config"),
                "retry_enabled": kwargs.get("retry_enabled"),
                "retry_transparently": kwargs.get("retry_transparently"),
                "bindings": kwargs.get("bindings"),
            }

            launcher_config = kwargs.get("launcher_config", {}) or {}
            config = kwargs.get("config", {}) or {}

            attempt = self._relaunch_attempts.get(launch_id, 0)
            cluster_name = self._cluster_name_for(launch_id, attempt)
            cloud = (
                launcher_config.get("resources", {}).get("cloud") or self._get_cloud()
            )
            idle_minutes = launcher_config.get(
                "idle_minutes_to_autostop", self._get_idle_minutes()
            )

            # Higher-precedence resource layers that override the compute_config
            # floor. infra/cluster/zone come only from these (never the floor), so
            # the target cloud can be resolved before the floor is layered in.
            compute_config = config.get("compute_config", {}) or {}
            override_res = {
                **launcher_config.get("resources", {}),
                **config.get("launcher_config", {}).get("resources", {}),
            }

            # Build infra string: supports 'cloud/cluster/partition' format
            # (e.g., 'slurm/mycluster/gpu', 'lsf/bluevela/normal')
            infra = override_res.get("infra") or cloud
            zone = override_res.get("zone")
            if not override_res.get("infra") and override_res.get("cluster"):
                infra = f"{cloud}/{override_res['cluster']}"
                if zone:
                    infra = f"{infra}/{zone}"
                    zone = None
            elif not override_res.get("infra") and zone:
                # zone without cluster — fold into infra to avoid the
                # "cannot specify both infra and zone" error in sky.Resources
                infra = f"{infra}/{zone}" if infra else zone
                zone = None

            # Normalized target cloud: the first infra segment, lowercased — the
            # single source of truth for cloud-specific resource handling (the
            # slurm/lsf memory skip in the floor below, and autostop later). Using
            # the resolved infra matches what SkyPilot actually provisions for
            # `infra: "slurm/..."`, an explicit `resources.cloud`, or non-canonical
            # casing — not just the env's default_cloud.
            cloud_group = (str(infra).split("/", 1)[0] or "").lower()

            # Build sky.Resources — merge order is precedence (last wins). The
            # step's config.compute_config (num_cpus_per_node/total_memory_per_node)
            # is the lowest-precedence floor; override_res wins. GPUs/accelerators
            # are not sourced from compute_config — they flow via override_res.
            res_config = {
                **self._resources_from_compute_config(
                    compute_config, cloud=cloud_group
                ),
                # override_res is passed VERBATIM to sky.Resources. On cloud
                # catalogs (aws/gcp/azure/k8s) an explicit resources.cpus/memory
                # must therefore use the "N+" minimum form (e.g. cpus: "3+"); a
                # bare int is an EXACT request no catalog satisfies ("Catalog does
                # not contain any instances satisfying ..."). The compute_config
                # floor above converts for you, but this free-form passthrough of
                # SkyPilot's own resources spec does not. (slurm/lsf match CPUs
                # directly, so a bare int is fine there.)
                **override_res,
            }

            # Build cluster config overrides (docker run_options, etc.)
            # SkyPilot's top-level `config:` section maps to
            # _cluster_config_overrides on sky.Resources.
            cluster_config_overrides = {}
            docker_config = {
                **launcher_config.get("docker", {}),
                **config.get("launcher_config", {}).get("docker", {}),
            }
            if docker_config:
                cluster_config_overrides["docker"] = docker_config

            # Trailing `or None` maps an empty image_id to None: the merged
            # `command` step renders image_id to "" when no image is given, and
            # sky.Resources expects None (bare node) rather than an empty string.
            image_id = (
                config.get("launcher_config", {}).get("image_id")
                or launcher_config.get("image_id")
            ) or None

            logger.info(
                "SkyPilot resources: accelerators=%s, image_id=%s, "
                "cluster_config_overrides=%s",
                res_config.get("accelerators"),
                image_id,
                cluster_config_overrides or None,
            )

            resources = sky.Resources(
                infra=infra,
                accelerators=res_config.get("accelerators"),
                instance_type=res_config.get("instance_type"),
                cpus=res_config.get("cpus"),
                memory=res_config.get("memory"),
                disk_size=res_config.get("disk_size"),
                use_spot=res_config.get("use_spot"),
                zone=zone,
                image_id=image_id,
                _cluster_config_overrides=cluster_config_overrides or None,
            )

            # Per-run workdir provisioned by setup_skypilot. Exported as
            # GB_BUILD_WORKDIR (inside get_launch_env_vars) and also used below
            # as the initial CWD of the run script and the remap target for
            # relative file_mounts, so it is computed here as a local.
            run_metadata = kwargs.get("run_metadata", {})
            build_workdir = (
                kwargs.get("setup_config", {}).get("skypilot", {}).get("build_workdir")
            )
            # Build the full env for the sky.Task. GB_BUILD_ID (and any future
            # standard var) comes from Environment.get_launch_env_vars() and is
            # authoritative over launcher/config env.
            env_vars = self.get_launch_env_vars(
                run_metadata=run_metadata,
                launcher_config=launcher_config,
                config=config,
                launch_id=launch_id,
                cluster_name=cluster_name,
                build_workdir=build_workdir,
                bindings=kwargs.get("bindings"),
            )

            # Inject inline hfpull downloads into setup from per-step bindings.
            # (HF_TOKEN from these bindings is already applied in the env above.)
            setup_script = launcher_config.get("setup") or ""
            pending_hfpulls = {}
            for bid, bval in (kwargs.get("bindings") or {}).items():
                if isinstance(bval, dict) and "_hfpull" in bval:
                    pending_hfpulls[bid] = bval["_hfpull"]
            if pending_hfpulls:
                hfpull_lines = [
                    "# -- gbserver: inline hfpull for inputs --",
                    "pip install --no-cache-dir 'huggingface_hub[cli]' 2>/dev/null || true",
                ]
                for bid, pull_info in pending_hfpulls.items():
                    cmd = f'hf download "{pull_info["repo"]}" --local-dir "{pull_info["path"]}"'
                    if pull_info.get("revision"):
                        cmd += f' --revision "{pull_info["revision"]}"'
                    if pull_info.get("type"):
                        cmd += f' --repo-type {pull_info["type"]}'
                    hfpull_lines.append(cmd)
                hfpull_lines.append("# -- end inline hfpull --")
                hfpull_block = "\n".join(hfpull_lines) + "\n"
                setup_script = hfpull_block + setup_script
                logger.info(
                    "Injected %d inline hfpull download(s) into setup script",
                    len(pending_hfpulls),
                )

            # Compute file_mounts up front so relative destinations can be
            # remapped into $GB_BUILD_WORKDIR (an absolute path on the shared
            # filesystem) before the task is built. On LSF/enroot that shared
            # tree is bind-mounted identity into the step container, so the
            # payload written on the login node is visible to the job at the same
            # path; the fork's backend wrap-exemption keeps these shared-root
            # destinations un-wrapped. Relative local sources resolve against
            # targetsteprun_asset_dir (the dir holding the rendered step.yaml +
            # siblings). See _build_skypilot_mounts / _resolve_local_mount_source.
            file_mounts_raw = launcher_config.get("file_mounts") or config.get(
                "file_mounts"
            )
            file_mounts: Dict[str, str] = {}
            storage_mounts: Dict[str, Any] = {}
            if file_mounts_raw:
                file_mounts, storage_mounts = _build_skypilot_mounts(
                    file_mounts_raw,
                    targetsteprun_asset_dir,
                    build_workdir,
                )

            # The prefix always leads with `set -eu` (fail fast). When a per-run
            # workdir was provisioned, it also prepends a `cd` into it to both
            # setup and run so step scripts start in a known directory and can use
            # relative paths without referencing $GB_BUILD_WORKDIR. With no
            # shared_workdir, _get_cli_prefix emits only `set -eu` (no cd), so the
            # scripts stay in SkyPilot's default ~/sky_workdir, where relative
            # file_mounts land. Only prefix setup when there is a setup script, so
            # steps without one don't acquire a spurious setup phase.
            cli_prefix = _get_cli_prefix(build_workdir)
            run_script = cli_prefix + launcher_config.get("run", "")
            if setup_script:
                setup_script = cli_prefix + setup_script

            # Build sky.Task
            task = sky.Task(
                name=cluster_name,
                setup=setup_script or None,
                run=run_script,
                envs=env_vars if env_vars else None,
                resources=resources,
            )

            # Attach the file/storage mounts computed above (may originate in the
            # launcher config or the step config).
            if file_mounts:
                task.set_file_mounts(file_mounts)
            if storage_mounts:
                task.set_storage_mounts(storage_mounts)

            logger.info(
                "Launching SkyPilot cluster: name=%s target=%s step=%s cloud=%s resources=%s",
                cluster_name,
                run_metadata.get("target_name", "") if run_metadata else "",
                run_metadata.get("targetstep_uri", "") if run_metadata else "",
                cloud,
                res_config,
            )

            # SLURM and LSF do not support autostop; passing any non-None
            # value (including 0) fails provisioning. Per-step `sky down`
            # cleanup handles teardown anyway, so force None on these
            # backends regardless of the user's config. Reuses cloud_group
            # (normalized first infra segment) computed above.
            autostop = None if cloud_group in _SSH_HPC_CLOUDS else idle_minutes

            # Launch and wait for provisioning, retrying transient
            # resource-acquisition failures (e.g. a just-torn-down slurm/lsf
            # allocation not yet released on retry). See _provision_with_retry.
            job_id, _handle = await self._provision_with_retry(
                task, cluster_name, autostop
            )

            self._cluster_names[launch_id] = cluster_name
            if job_id is not None:
                self._job_ids[launch_id] = job_id

            logger.info(
                "SkyPilot cluster %s launched: job_id=%s launch_id=%s",
                cluster_name,
                job_id,
                launch_id,
            )

            # Ensure log directory exists for job log streaming
            os.makedirs(f"/tmp/sky-logs/{cluster_name}", exist_ok=True)

            # Execute post-launch tasks (e.g., start evaluator sidecars) if defined
            post_launch_task = launcher_config.get("post_launch_task")
            if post_launch_task:
                try:
                    logger.info(
                        "Executing post-launch task on cluster %s (launch_id=%s)",
                        cluster_name,
                        launch_id,
                    )
                    host_ip, ssh_key = await asyncio.to_thread(
                        _extract_host_ssh_info, cluster_name
                    )
                    await _execute_on_host_via_ssh(
                        host_ip=host_ip,
                        ssh_key=ssh_key,
                        commands=post_launch_task.get("run", ""),
                        env_vars=env_vars,
                    )
                    logger.info(
                        "Post-launch task completed on cluster %s (launch_id=%s)",
                        cluster_name,
                        launch_id,
                    )
                except Exception as e:
                    logger.error(
                        "Post-launch task failed on cluster %s (launch_id=%s): %s",
                        cluster_name,
                        launch_id,
                        e,
                    )
                    # Emit a MESSAGE_EVENT so the failure is visible in build state
                    if self.event_q and run_metadata:
                        from gbserver.types.buildevent import (
                            BuildEvent,
                            BuildEventMessagePayload,
                            BuildEventType,
                            EntityRunMetadata,
                        )

                        self.event_q.put_nowait(
                            BuildEvent(
                                run_metadata=EntityRunMetadata(**run_metadata),
                                type=BuildEventType.MESSAGE_EVENT,
                                payload=BuildEventMessagePayload(
                                    msg=f"Post-launch task failed on {cluster_name}: {e}"
                                ),
                            )
                        )

        except Exception as e:
            logger.error("Failed to launch SkyPilot cluster for %s: %s", launch_id, e)
            raise
        finally:
            self._release_monitors(launch_id)

    async def _provision_with_retry(
        self: Self,
        task: Any,
        cluster_name: str,
        autostop: Optional[int],
    ) -> Tuple[Optional[int], Any]:
        """Run ``sky.launch`` + ``sky.stream_and_get`` with bounded retry on
        transient resource-acquisition failures.

        On a retry (RetryHandler tears the cluster down then relaunches the same
        name), the backend (slurm/lsf) allocation may not be released yet, so the
        relaunch can fail with "Failed to acquire resources". Rather than fail the
        whole build, retry the provisioning a bounded number of times with capped
        exponential backoff — the backoff gives the backend time to release. A
        failed provision can leave a partial INIT/FAILED cluster record under the
        same name, so tear it down before each retry. Non-transient failures
        re-raise immediately; on exhaustion the original error is re-raised
        (``reraise=True``) so the genuine message surfaces.

        Args:
            task: The ``sky.Task`` to launch.
            cluster_name: Deterministic cluster name for this launch.
            autostop: idle_minutes_to_autostop (None on slurm/lsf).

        Returns:
            Tuple of (job_id, handle) from ``sky.stream_and_get``.

        Raises:
            Exception: The last provisioning error if all attempts are exhausted,
                or any non-transient error immediately.
        """
        from gbserver.types.constants import (
            GBSERVER_SKYPILOT_PROVISION_BACKOFF_MAX,
            GBSERVER_SKYPILOT_PROVISION_MAX_ATTEMPTS,
        )

        # Use environment config retry settings if available, else fall back to env vars
        retry_config = self.config.config.get("retry", {}) if self.config else {}
        max_attempts = int(
            retry_config.get("max_retries", GBSERVER_SKYPILOT_PROVISION_MAX_ATTEMPTS)
        )
        provision_backoff_max = int(
            retry_config.get(
                "provision_backoff_max",
                max(1800, GBSERVER_SKYPILOT_PROVISION_BACKOFF_MAX),
            )
        )

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_transient_provision_error),
            wait=wait_exponential(multiplier=30, max=provision_backoff_max),
            stop=stop_after_attempt(max(1, max_attempts)),
            reraise=True,
        ):
            with attempt:
                try:
                    # Both sky.launch (submit) and sky.stream_and_get (wait)
                    # block in an OS thread, so a CancelledError delivered to
                    # this task is deferred until the thread returns — the
                    # submit can block while the cluster is in INIT (see
                    # build/run.py), the stream wait for 2-5 min. Shield each
                    # future so the outer await observes the cancel
                    # *immediately* while the thread keeps running, letting us
                    # abort the SkyPilot request server-side and tear down any
                    # partial cluster rather than leaking it.
                    launch_fut = asyncio.ensure_future(
                        _sky_submit_to_thread(
                            sky.launch,
                            task,
                            cluster_name=cluster_name,
                            idle_minutes_to_autostop=autostop,
                        )
                    )
                    try:
                        request_id = await asyncio.shield(launch_fut)
                    except asyncio.CancelledError:
                        # Cancelled during the submit: no request_id yet, but
                        # the thread may still be creating the cluster under the
                        # deterministic name. _abort_provision drains the submit
                        # to recover the request_id (if any) and tears down by
                        # name so the cluster is not leaked.
                        await self._abort_provision(None, cluster_name, launch_fut)
                        raise
                    stream_fut = asyncio.ensure_future(
                        _sky_submit_to_thread(sky.stream_and_get, request_id)
                    )
                    try:
                        return await asyncio.shield(stream_fut)
                    except asyncio.CancelledError:
                        await self._abort_provision(
                            request_id, cluster_name, stream_fut
                        )
                        raise
                except Exception as e:
                    # Clear the partial INIT/FAILED cluster record before the
                    # next attempt so the relaunch doesn't reuse the stale
                    # allocation. Only for transient errors — others re-raise
                    # untouched and tenacity will not retry them.
                    if _is_transient_provision_error(e):
                        logger.warning(
                            "Transient provision failure for %s (attempt %d): %s "
                            "— tearing down partial cluster before retry",
                            cluster_name,
                            attempt.retry_state.attempt_number,
                            e,
                        )
                        await self._teardown(cluster_name)
                    raise
        # Unreachable: AsyncRetrying with reraise=True either returns from the
        # `return` above or raises; this satisfies the type checker.
        raise AssertionError("unreachable: _provision_with_retry exited loop")

    async def _abort_provision(
        self: Self,
        request_id: Any,
        cluster_name: str,
        pending_fut: Optional["asyncio.Future"],
    ) -> None:
        """Abort an in-flight SkyPilot provisioning request after cancellation.

        Thin wrapper over the shared ``_abort_shielded_request`` whose only
        launcher-specific part is the reclaim: tear down the (possibly partial)
        cluster by its deterministic name. ``self._cluster_names`` is not
        populated until provisioning returns, so teardown-by-name is the safety
        net that reclaims the cluster even when there is no request_id to abort.

        Args:
            request_id: The id from ``sky.launch``, or ``None`` if cancelled
                during the submit (recovered by draining ``pending_fut``).
            cluster_name: Deterministic cluster name to tear down.
            pending_fut: The shielded ``to_thread`` future to drain (the submit
                or the ``sky.stream_and_get`` wait).
        """

        async def _teardown_cluster() -> None:
            await self._teardown(cluster_name)

        await _abort_shielded_request(
            request_id,
            pending_fut,
            description=f"SkyPilot cluster {cluster_name}",
            on_abort=_teardown_cluster,
        )

    async def monitor_skypilot_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata=None,
        build_id: str = "",
        event_configs: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Monitor a SkyPilot job through the shared retry framework.

        Wraps ``_poll_skypilot_job`` in ``_with_retry_handler`` so terminal
        FAILED events are routed to ``RetryHandler``, which either calls
        ``retry_workload`` (cleanup + relaunch + sets the per-launch
        retry-complete event) or raises ``WorkloadFailedException`` to
        propagate failure.

        Each poll runs as its own task, raced (``asyncio.wait`` /
        ``FIRST_COMPLETED``) against the handler task: if the handler reaches a
        terminal no-retry verdict it raises and completes first, so the verdict
        surfaces promptly (the cancelled poll never hangs on its deferred
        ``stop_event`` wait); if the poll completes first it is either a terminal
        SUCCESS or a retry handoff (``stop_event`` set by ``retry_workload``),
        and the monitor awaits the relaunch and re-polls the fresh cluster.
        This lets a relaunched cluster that fails again be retried in turn, up to
        the handler's budget. When no handler exists (no strategies), the poll
        raises on terminal failure directly.
        """
        _require_skypilot()
        retry_complete_event = asyncio.Event()
        self._skypilot_retry_complete_events[launch_id] = retry_complete_event
        retry_in_progress_event = asyncio.Event()
        self._skypilot_retry_in_progress_events[launch_id] = retry_in_progress_event

        enabled, retry_transparently = self._get_step_retry_config(
            self._launch_kwargs.get(launch_id, {})
        )

        async with self._with_retry_handler(
            launch_id,
            event_q,
            build_id,
            enabled=enabled,
            entityrun_metadata=entityrun_metadata,
            retry_transparently=retry_transparently,
        ) as (monitor_queue, handler_task):
            try:
                while True:
                    retry_complete_event.clear()
                    retry_in_progress_event.clear()
                    poll_task = asyncio.create_task(
                        self._poll_skypilot_job(
                            launch_id=launch_id,
                            event_q=monitor_queue,
                            entityrun_metadata=entityrun_metadata,
                            event_configs=event_configs,
                            defer_terminal_failure=handler_task is not None,
                            **kwargs,
                        )
                    )
                    waiters = {poll_task}
                    if handler_task is not None:
                        waiters.add(handler_task)
                    done, _ = await asyncio.wait(
                        waiters, return_when=asyncio.FIRST_COMPLETED
                    )

                    if handler_task is not None and handler_task in done:
                        # Handler reached a terminal verdict (while the monitor
                        # body runs it completes only by raising). Cancel the
                        # deferred poll and return; __aexit__'s ``await task``
                        # surfaces the handler's WorkloadFailedException.
                        poll_task.cancel()
                        try:
                            await poll_task
                        except asyncio.CancelledError:
                            pass
                        return

                    # poll_task completed first: surface its result (a terminal
                    # raise when no handler is deferring) or fall through.
                    await poll_task
                    # Returned without raising: terminal SUCCESS, or stop_event
                    # was set by retry_workload to begin a retry. retry_in_progress
                    # — set before stop_event in retry_workload — disambiguates.
                    if not retry_in_progress_event.is_set():
                        return  # terminal success path; done.
                    # A retry is underway. Wait for the (possibly slow) relaunch
                    # to finish before polling again. retry_complete is set in
                    # retry_workload's finally, on both success and failure.
                    try:
                        await asyncio.wait_for(
                            retry_complete_event.wait(),
                            timeout=RETRY_RELAUNCH_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError as e:
                        raise WorkloadFailedException(
                            f"Retry relaunch never signalled completion within "
                            f"{RETRY_RELAUNCH_TIMEOUT_SECONDS}s (launch_id={launch_id})"
                        ) from e
                    if self._cluster_names.get(launch_id):
                        # Relaunch succeeded -> poll the fresh cluster/job.
                        continue
                    # Relaunch failed (no fresh cluster). Raise so the step fails
                    # regardless of how the RetryHandler classified the trigger
                    # event (never return cleanly on a failed step).
                    raise WorkloadFailedException(
                        f"Retry relaunch failed; no cluster for launch_id={launch_id}"
                    )
            finally:
                self._skypilot_retry_complete_events.pop(launch_id, None)
                self._skypilot_retry_in_progress_events.pop(launch_id, None)

    async def _poll_skypilot_job(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata=None,
        event_configs: Optional[List] = None,
        defer_terminal_failure: bool = False,
        **kwargs,
    ) -> None:
        """Poll ``sky.job_status`` for one launch attempt, emit events.

        Emits a ``WORKLOAD_STATUS_EVENT(FAILED)`` on a non-success terminal
        state so the RetryHandler can decide between retry and final-failure.

        Terminal non-success handling depends on ``defer_terminal_failure``:

        - ``True`` (used when a RetryHandler is active): after emitting the
          FAILED event, wait on ``stop_event`` and return. ``retry_workload``
          sets ``stop_event`` to begin a retry; on a no-retry verdict the
          handler raises and ``monitor_skypilot_monitor`` cancels this poll.
          The handler — not this coroutine — owns failure propagation.
        - ``False`` (no RetryHandler to defer to): raise
          ``WorkloadFailedException`` directly so the step fails.

        Returns on terminal SUCCESS, on ``stop_event`` (retry), or (when
        deferring) after the terminal FAILED handoff.
        """
        event_log_parser_configs = []
        if event_configs is not None:
            event_log_parser_configs = [
                EventLogLineParserConfig.model_validate(config)
                for config in event_configs
            ]

        cluster_name = self._cluster_names.get(launch_id)
        job_id = self._job_ids.get(launch_id)
        if not cluster_name:
            logger.error("No cluster_name for launch_id %s", launch_id)
            return
        stop_event = self._get_launch_stopped_event(launch_id)
        # Canonical key across step.yaml configs is ``poll_interval_seconds``;
        # accept the legacy ``poll_interval`` for back-compat. Templated configs
        # may render this as a string (e.g. "120"), so coerce to a number.
        _raw_poll = kwargs.get(
            "poll_interval_seconds",
            kwargs.get("poll_interval", _DEFAULT_POLL_INTERVAL_SECONDS),
        )
        try:
            poll_interval = float(_raw_poll)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid poll_interval_seconds %r; falling back to %d",
                _raw_poll,
                _DEFAULT_POLL_INTERVAL_SECONDS,
            )
            poll_interval = _DEFAULT_POLL_INTERVAL_SECONDS
        # Per-step log-retrieval policy (mode + cadence). Defaults to
        # on_completion: pull the full log once at terminal status.
        log_mode, log_interval, startup_window = _parse_log_retrieval(
            kwargs, poll_interval
        )
        last_status = None
        consecutive_poll_failures = 0
        max_poll_failures = 3

        # Live log streaming state (only used in ``stream`` mode)
        log_stream_task: Optional[asyncio.Task] = None
        logfile_monitor: Optional["LogFileMonitor"] = None
        log_stream_stop = asyncio.Event()
        lines_already_processed = 0
        # Pull-mode bookkeeping (periodic / startup_window).
        run_start: Optional[float] = None  # monotonic time job entered RUNNING
        last_pull_at: Optional[float] = None  # monotonic time of last pull
        self._log_lines_parsed.setdefault(launch_id, 0)

        while not stop_event.is_set():
            status = None
            poll_failed = False
            try:
                request_id = await asyncio.to_thread(
                    lambda: sky.job_status(
                        cluster_name,
                        job_ids=[job_id] if job_id is not None else None,
                    )
                )
                statuses = await asyncio.to_thread(sky.get, request_id)
                status = statuses.get(job_id) if statuses else None
                consecutive_poll_failures = 0
            except Exception as e:
                logger.error(
                    "Error polling SkyPilot job %s on %s: %s",
                    job_id,
                    cluster_name,
                    e,
                )
                poll_failed = True
                consecutive_poll_failures += 1
                if (
                    "does not exist" in str(e)
                    or consecutive_poll_failures >= max_poll_failures
                ):
                    logger.warning(
                        "Cluster %s is gone (preempted or terminated) after %d consecutive poll failures. "
                        "Treating as FAILED for launch_id %s.",
                        cluster_name,
                        consecutive_poll_failures,
                        launch_id,
                    )
                    status = sky.JobStatus.FAILED
                    poll_failed = False

            # launch_skypilot_teardown downs this SERVICE's cluster on purpose,
            # so a poll seeing it "gone" (FAILED above) is success, not a crash.
            # The teardown runs in a DIFFERENT Skypilot instance (one per target),
            # so we match on the process-global set of torn-down cluster names --
            # cluster_name here is gb-<launch_id[:12]>, the same name the teardown
            # recorded. Exit cleanly before any FAILED event or raise so the step
            # is marked SUCCESS. Checked after the poll (not only at the loop top)
            # to close the race where teardown fires while this poll is in flight.
            if cluster_name in Skypilot._intentionally_torn_down_clusters:
                logger.info(
                    "Cluster %s (launch_id %s) was intentionally torn down; "
                    "ending monitor as success.",
                    cluster_name,
                    launch_id,
                )
                return

            # Skip change-detection on poll failures so a transient error
            # doesn't emit a spurious RUNNING -> None -> RUNNING flap event.
            if not poll_failed and status != last_status:
                logger.info(
                    "SkyPilot job %s on %s status: %s -> %s (launch_id=%s)",
                    job_id,
                    cluster_name,
                    last_status,
                    status,
                    launch_id,
                )
                if event_q and entityrun_metadata:
                    from gbserver.types.buildevent import (
                        BuildEvent,
                        BuildEventMessagePayload,
                        BuildEventType,
                    )

                    event = BuildEvent(
                        run_metadata=entityrun_metadata,
                        type=BuildEventType.MESSAGE_EVENT,
                        payload=BuildEventMessagePayload(
                            msg=f"SkyPilot job {job_id} on {cluster_name}: {status}"
                        ),
                    )
                    await event_q.put(event)
                last_status = status

            # --- Log retrieval dispatch (runs every poll while the job lives) ---
            # Only meaningful once we have event parsers, a sink, and a job id.
            log_retrieval_active = (
                event_log_parser_configs
                and event_q
                and entityrun_metadata
                and job_id is not None
            )
            is_running = status is not None and str(status) == "JobStatus.RUNNING"
            # Set while a pull-mode step is still within its pulling window, so
            # the loop sleep below shortens to the log-pull cadence.
            pulls_active = False

            if log_retrieval_active and is_running and run_start is None:
                run_start = time.monotonic()

            if log_retrieval_active and is_running and log_mode == LOG_RETRIEVAL_STREAM:
                # Real-time follow stream: start once on RUNNING, then supervise.
                if log_stream_task is None:
                    log_stream_task, logfile_monitor = self._start_log_stream_task(
                        cluster_name=cluster_name,
                        job_id=job_id,
                        launch_id=launch_id,
                        event_q=event_q,
                        entityrun_metadata=entityrun_metadata,
                        event_log_parser_configs=event_log_parser_configs,
                        stop_event=log_stream_stop,
                        abort_event=stop_event,
                        start_line=0,
                    )
            elif (
                log_retrieval_active
                and is_running
                and log_mode
                in (
                    LOG_RETRIEVAL_PERIODIC,
                    LOG_RETRIEVAL_STARTUP_WINDOW,
                )
            ):
                # Incremental pull: re-download the log and parse only lines past
                # the last one we emitted events for. startup_window stops pulling
                # once the configured window after RUNNING has elapsed.
                now = time.monotonic()
                in_window = log_mode == LOG_RETRIEVAL_PERIODIC or (
                    run_start is not None and now - run_start <= startup_window
                )
                pulls_active = in_window
                due = last_pull_at is None or (now - last_pull_at) >= log_interval
                if in_window and due:
                    last_pull_at = now
                    resume = self._log_lines_parsed.get(launch_id, 0)
                    new_last = await self._download_and_parse_logs(
                        cluster_name=cluster_name,
                        job_id=job_id,
                        launch_id=launch_id,
                        event_q=event_q,
                        entityrun_metadata=entityrun_metadata,
                        event_log_parser_configs=event_log_parser_configs,
                        start_line_num=resume,
                    )
                    if new_last:
                        self._log_lines_parsed[launch_id] = max(resume, new_last)

            # Supervise the live stream task (stream mode only): restart on crash,
            # record covered line count on clean finish.
            if log_stream_task is not None and log_stream_task.done():
                exc = (
                    log_stream_task.exception()
                    if not log_stream_task.cancelled()
                    else None
                )
                processed = logfile_monitor.line_num if logfile_monitor else 0
                if exc is not None:
                    logger.warning(
                        "Log stream task failed after %d lines for %s job %s: %s. "
                        "Attempting restart.",
                        processed,
                        cluster_name,
                        job_id,
                        exc,
                    )
                    log_stream_stop = asyncio.Event()
                    log_stream_task, logfile_monitor = self._start_log_stream_task(
                        cluster_name=cluster_name,
                        job_id=job_id,
                        launch_id=launch_id,
                        event_q=event_q,
                        entityrun_metadata=entityrun_metadata,
                        event_log_parser_configs=event_log_parser_configs,
                        stop_event=log_stream_stop,
                        abort_event=stop_event,
                        start_line=processed,
                    )
                else:
                    lines_already_processed = processed
                    log_stream_task = None

            if status is not None and status.is_terminal():
                logger.info(
                    "SkyPilot job %s reached terminal status: %s",
                    job_id,
                    status,
                )
                # Stop the live log stream and determine how many lines it covered
                if log_stream_task is not None and not log_stream_task.done():
                    log_stream_stop.set()
                    try:
                        await asyncio.wait_for(log_stream_task, timeout=15.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logger.warning(
                            "Log stream task did not finish in time for %s job %s, cancelling",
                            cluster_name,
                            job_id,
                        )
                        log_stream_task.cancel()
                        try:
                            await log_stream_task
                        except (asyncio.CancelledError, Exception):
                            pass
                if logfile_monitor is not None:
                    # Use lines_consumed from the stream source (not line_num
                    # from the monitor) to avoid re-emitting events for lines
                    # that were read from the log but not yet processed by the
                    # monitor when the stream was cancelled.
                    lines_already_processed = getattr(
                        logfile_monitor.stream_source,
                        "lines_consumed",
                        logfile_monitor.line_num,
                    )

                # Final pull at terminal status. For pull modes resume past the
                # lines already parsed during the run; for stream mode pull only
                # if the live stream never ran (lines_already_processed == 0).
                if log_mode == LOG_RETRIEVAL_STREAM:
                    terminal_resume = lines_already_processed
                    should_pull = lines_already_processed == 0
                else:
                    terminal_resume = self._log_lines_parsed.get(launch_id, 0)
                    should_pull = True
                if (
                    should_pull
                    and event_log_parser_configs
                    and event_q
                    and entityrun_metadata
                    and job_id is not None
                ):
                    new_last = await self._download_and_parse_logs(
                        cluster_name=cluster_name,
                        job_id=job_id,
                        launch_id=launch_id,
                        event_q=event_q,
                        entityrun_metadata=entityrun_metadata,
                        event_log_parser_configs=event_log_parser_configs,
                        start_line_num=terminal_resume,
                    )
                    if new_last:
                        self._log_lines_parsed[launch_id] = max(
                            terminal_resume, new_last
                        )
                if str(status) != "JobStatus.SUCCEEDED":
                    if event_q and entityrun_metadata:
                        from gbserver.types.buildevent import (
                            BuildEvent,
                            BuildEventType,
                            BuildEventWorkloadStatusPayload,
                        )
                        from gbserver.types.status import Status

                        fail_event = BuildEvent(
                            run_metadata=entityrun_metadata,
                            type=BuildEventType.WORKLOAD_STATUS_EVENT,
                            payload=BuildEventWorkloadStatusPayload(
                                status=Status.FAILED,
                            ),
                        )
                        await event_q.put(fail_event)
                    terminal_msg = (
                        f"SkyPilot job {job_id} on {cluster_name} "
                        f"terminated with status {status} "
                        f"(launch_id={launch_id})"
                    )
                    if defer_terminal_failure:
                        # A RetryHandler is active: hand the FAILED event off to
                        # it and wait. It either initiates a retry (sets
                        # stop_event via retry_workload) or raises a terminal
                        # verdict, on which monitor_skypilot_monitor cancels this
                        # poll. Do NOT raise here — that would tear the handler
                        # down before it can decide (the no-retry gap bug).
                        await stop_event.wait()
                        return
                    # No RetryHandler to defer to: raise so the failure
                    # propagates up through monitor_skypilot_monitor ->
                    # Run.run, which sets Status.FAILED on the step.
                    raise WorkloadFailedException(terminal_msg)
                return

            try:
                sleep_timeout = _effective_poll_timeout(
                    poll_interval, log_mode, log_interval, pulls_active
                )
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_timeout)
                # stop_event was set (retry or external cancellation) — clean up log stream
                if log_stream_task is not None and not log_stream_task.done():
                    log_stream_stop.set()
                    log_stream_task.cancel()
                    try:
                        await log_stream_task
                    except (asyncio.CancelledError, Exception):
                        pass
                return
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue polling

    def _start_log_stream_task(
        self: Self,
        cluster_name: str,
        job_id: int,
        launch_id: str,
        event_q: asyncio.Queue,
        entityrun_metadata,
        event_log_parser_configs: list,
        stop_event: asyncio.Event,
        abort_event: asyncio.Event,
        start_line: int = 0,
    ) -> Tuple[asyncio.Task, "LogFileMonitor"]:
        """Create and launch a log streaming task for a SkyPilot job."""
        from gbserver.monitoring.logfile_monitor import LogFileMonitor
        from gbserver.monitoring.streams.skypilot_log_stream import (
            SkyPilotLogStreamSource,
        )

        # Open a local log file for streaming writes
        tmp_log_dir = f"/tmp/sky-logs/{cluster_name}"
        os.makedirs(tmp_log_dir, exist_ok=True)
        log_file_path = f"{tmp_log_dir}/job-{job_id}.log"
        log_file = open(log_file_path, "a", encoding="utf-8")
        logger.info("Streaming job logs to %s", log_file_path)

        stream_source = SkyPilotLogStreamSource(
            cluster_name=cluster_name,
            job_id=job_id,
            start_line=start_line,
            abort_event=abort_event,
            log_file=log_file,
        )
        monitor = LogFileMonitor(
            step_id=launch_id,
            stream_source=stream_source,
            event_configs=event_log_parser_configs,
            launch_id=launch_id,
            entityrun_metadata=entityrun_metadata,
            event_queue=event_q,
            stop_event=stop_event,
        )
        task = asyncio.create_task(monitor.monitor())
        logger.info(
            "Started live log stream for %s job %s (start_line=%d)",
            cluster_name,
            job_id,
            start_line,
        )
        return task, monitor

    async def _download_and_parse_logs(
        self: Self,
        cluster_name: str,
        job_id: int,
        launch_id: str,
        event_q: asyncio.Queue,
        entityrun_metadata,
        event_log_parser_configs: list,
        start_line_num: int = 0,
    ) -> int:
        """Download job logs and parse for artifact events.

        Args:
            start_line_num: Skip lines at or below this number (1-based).
                Used to avoid re-emitting events already processed by a prior
                pull or by live log streaming.

        Returns:
            The highest 1-based line number seen in the log (0 if nothing was
            read). Callers use this as the next ``start_line_num`` to resume an
            incremental pull without re-emitting events.
        """
        if start_line_num > 0:
            logger.info(
                "Downloading logs for %s job %s, skipping first %d lines "
                "(already processed by live stream)",
                cluster_name,
                job_id,
                start_line_num,
            )
        max_line = start_line_num
        try:
            log_dir = _download_logs_with_retry(cluster_name, job_id)
            if not log_dir:
                logger.warning(
                    "No log directory returned for cluster %s job %s",
                    cluster_name,
                    job_id,
                )
                return max_line

            log_dir = os.path.expanduser(log_dir)
            # Save a copy to /tmp for easy debugging access
            tmp_log_dir = f"/tmp/sky-logs/{cluster_name}/job-{job_id}"
            os.makedirs(tmp_log_dir, exist_ok=True)
            for f in glob.glob(f"{log_dir}/*"):
                try:
                    import shutil

                    shutil.copy2(f, tmp_log_dir)
                except OSError:
                    pass
            logger.info(
                "Saved job logs to %s (cluster %s job %s)",
                tmp_log_dir,
                cluster_name,
                job_id,
            )

            log_files = sorted(glob.glob(f"{log_dir}/*.log"))
            if not log_files:
                logger.info(
                    "No log files found in %s for cluster %s job %s",
                    log_dir,
                    cluster_name,
                    job_id,
                )
                return max_line

            for log_file in log_files:
                try:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
                            if line_num > max_line:
                                max_line = line_num
                            if line_num <= start_line_num:
                                continue
                            line = line.rstrip("\n")
                            if line:
                                await self.get_events_from_log_line(
                                    log_line=line,
                                    event_configs=event_log_parser_configs,
                                    event_q=event_q,
                                    entityrun_metadata=entityrun_metadata,
                                    line_num=line_num,
                                )
                except OSError as e:
                    logger.warning("Failed to read log file %s: %s", log_file, e)
                    continue

        except Exception as e:
            logger.error(
                "Failed to download/parse logs for cluster %s job %s (launch_id=%s): %s",
                cluster_name,
                job_id,
                launch_id,
                e,
            )
        return max_line

    async def _teardown(self: Self, cluster_name: str) -> None:
        """Tear down a SkyPilot cluster by name, off the event loop.

        Wraps ``sky.down(purge=True)`` + ``sky.get`` in ``asyncio.to_thread`` so
        the blocking SDK calls don't stall the event loop, and tolerates a
        cluster that is already gone. Does not touch the per-launch bookkeeping
        dicts — callers own those.

        Args:
            cluster_name: The SkyPilot cluster name to remove.
        """
        try:
            request_id = await asyncio.to_thread(sky.down, cluster_name, purge=True)
            await asyncio.to_thread(sky.get, request_id)
            logger.info("Torn down SkyPilot cluster %s", cluster_name)
        except Exception as e:
            cluster_gone = (
                getattr(sky.exceptions, "ClusterDoesNotExist", ())
                if sky is not None
                else ()
            )
            if isinstance(cluster_gone, type) and isinstance(e, cluster_gone):
                logger.info("SkyPilot cluster %s already gone", cluster_name)
                return
            logger.error("Failed to tear down SkyPilot cluster %s: %s", cluster_name, e)

    async def cleanup_skypilot(
        self: Self,
        launch_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Tear down a SkyPilot cluster."""
        if launch_id is None:
            logger.warning("cleanup_skypilot called with no launch_id")
            return

        self._monitoring_cleanup(launch_id=launch_id)

        cluster_name = self._cluster_names.get(launch_id)
        if not cluster_name:
            logger.warning("No cluster to cleanup for launch_id %s", launch_id)
            return

        try:
            _require_skypilot()
            logger.info(
                "Tearing down SkyPilot cluster %s (launch_id=%s)",
                cluster_name,
                launch_id,
            )
            await self._teardown(cluster_name)
        except Exception as e:
            logger.error("Failed to tear down SkyPilot cluster %s: %s", cluster_name, e)
        finally:
            self._cluster_names.pop(launch_id, None)
            self._job_ids.pop(launch_id, None)
            self._launch_kwargs.pop(launch_id, None)
            self._relaunch_attempts.pop(launch_id, None)
            self._log_lines_parsed.pop(launch_id, None)

    async def launch_skypilot_teardown(
        self: Self,
        launch_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """In-process launcher that tears down named SkyPilot clusters.

        Does NOT provision a cluster. Reads ``config.teardown_config.cluster_names``
        (surfaced from upstream cluster_name bindings) and downs each one. The
        ``Skypilot`` instance is shared across a build's targets, so the
        ``rm-server`` / ``code-server`` clusters are in ``self._cluster_names``
        here -- reuse ``cleanup_skypilot`` so monitoring/state is cleaned up too.
        Falls back to a direct ``sky.down`` for any name without a tracked
        launch_id. SERVICE clusters on LSF never autostop and never get a
        terminal-status cleanup, so this is how they get reclaimed.
        """
        config = kwargs.get("config") or {}
        names = (config.get("teardown_config") or {}).get("cluster_names") or []
        names = [n.strip() for n in names if isinstance(n, str) and n.strip()]
        if not names:
            logger.warning(
                "launch_skypilot_teardown: no cluster_names to tear down "
                "(launch_id=%s)",
                launch_id,
            )
            return

        # Reverse launch_id -> cluster_name so we can reuse cleanup_skypilot.
        name_to_launch = {v: k for k, v in self._cluster_names.items()}

        for name in names:
            # Record BEFORE downing so the SERVICE's monitor -- which runs in a
            # different Skypilot instance and may be mid-poll -- treats the
            # cluster going away as success, not a WorkloadFailedException. Keyed
            # by cluster name (gb-<launch_id[:12]>), the name the monitor sees.
            Skypilot._intentionally_torn_down_clusters.add(name)
            try:
                target_launch_id = name_to_launch.get(name)
                if target_launch_id is not None:
                    logger.info(
                        "launch_skypilot_teardown: cleanup cluster %s "
                        "(launch_id=%s)",
                        name,
                        target_launch_id,
                    )
                    await self.cleanup_skypilot(launch_id=target_launch_id)
                else:
                    logger.info(
                        "launch_skypilot_teardown: no tracked launch_id for "
                        "cluster %s, calling sky.down directly",
                        name,
                    )
                    _require_skypilot()
                    request_id = await asyncio.to_thread(sky.down, name, purge=True)
                    await asyncio.to_thread(sky.get, request_id)
                    logger.info("launch_skypilot_teardown: torn down cluster %s", name)
            except Exception as e:  # don't let one failure skip the rest
                logger.error(
                    "launch_skypilot_teardown: failed to tear down %s: %s", name, e
                )

    async def retry_workload(
        self: Self,
        launch_id: str,
        nodes_to_avoid: Optional[List[str]] = None,
        retry_count: int = 0,
        **kwargs,
    ) -> None:
        """Retry a failed Skypilot workload via tear-down + relaunch.

        Called by ``RetryHandler`` when a strategy decides the failure is
        retriable. Sets ``_skypilot_retry_in_progress_events[launch_id]``,
        stops the polling loop, takes the cluster down, and re-invokes
        ``launch_skypilot`` with the kwargs stashed during the first launch.
        The relaunch provisions a *fresh, uniquely-named* cluster
        (``gb-<launch_id>-r<retry_count>``) rather than reusing the original
        name: ``sky down`` returning does not guarantee the backend (slurm/lsf)
        allocation has drained, so reusing the name races the still-draining
        original and intermittently fails provisioning. A distinct name sidesteps
        that contention. Sets ``_skypilot_retry_complete_events[launch_id]`` in a
        ``finally`` — on BOTH relaunch success and failure — to release
        ``monitor_skypilot_monitor``, which then polls the fresh cluster
        (success) or fails the step (failure, no fresh cluster).

        :param launch_id: The launch identifier to retry.
        :param nodes_to_avoid: Currently logged-and-ignored — Skypilot
            has no portable per-launch node-exclusion knob.
        :param retry_count: 1-based relaunch attempt from ``RetryHandler``; used
            to derive the fresh cluster name so each attempt is distinct.
        :raises Exception: Re-raises any failure from the relaunch.
        """
        original_kwargs = self._launch_kwargs.get(launch_id, {})
        cluster_name = self._cluster_names.get(launch_id, launch_id)
        if nodes_to_avoid:
            logger.info(
                "retry_workload: nodes_to_avoid=%s ignored for launch_id=%s "
                "(no portable Skypilot node-exclusion knob)",
                nodes_to_avoid,
                launch_id,
            )

        msg = (
            f"⚠️ Skypilot error on cluster {cluster_name} "
            f"(launch_id={launch_id}), retrying..."
        )
        self._send_message(msg=msg, **original_kwargs)

        # Mark the retry as in-progress BEFORE stopping the poll loop. Ordering
        # is load-bearing: monitor_skypilot_monitor observes stop_event only
        # after this set (no await between the two), so it can distinguish a
        # retry-induced poll stop from a terminal completion.
        retry_in_progress = self._skypilot_retry_in_progress_events.get(launch_id)
        if retry_in_progress is not None:
            retry_in_progress.set()

        # Stop the polling loop cleanly before sky down.
        self._get_launch_stopped_event(launch_id).set()

        try:
            try:
                await self.cleanup_skypilot(launch_id=launch_id)
            except Exception as e:
                logger.warning(
                    "retry_workload cleanup_skypilot failed for %s: %s", launch_id, e
                )

            # Reset the stop event so the next polling iteration runs.
            self._get_launch_stopped_event(launch_id).clear()
            # Re-arm the launch-ready gate so launch_skypilot's release_monitors
            # call has a fresh event to set.
            self._get_launch_ready_event(launch_id)

            # Record the attempt so _launch_skypilot_inner provisions a fresh,
            # uniquely-named cluster. Set AFTER cleanup_skypilot (which pops this
            # entry) so the new value is the one the relaunch reads.
            self._relaunch_attempts[launch_id] = retry_count

            await self.launch_skypilot(launch_id, **original_kwargs)
        except Exception as launch_error:
            logger.error(
                "retry_workload could not relaunch launch_id=%s: %s",
                launch_id,
                launch_error,
            )
            raise
        finally:
            # Signal monitor_skypilot_monitor on BOTH relaunch success and
            # failure. On failure _cluster_names[launch_id] is absent (set only
            # after provisioning succeeds in _launch_skypilot_inner), so the
            # monitor wakes, sees no fresh cluster, and fails the step. Setting
            # this in finally also prevents the monitor from hanging on its wait.
            retry_event = self._skypilot_retry_complete_events.get(launch_id)
            if retry_event is not None:
                retry_event.set()

    def _get_default_retry_strategies(self: Self) -> List["RetryStrategy"]:
        """Return Skypilot's default retry strategies.

        Skypilot ships ``AnyFailureRetryStrategy`` as the sole default —
        any failure event (a ``WORKLOAD_STATUS_EVENT`` with
        ``status=FAILED`` or a ``MESSAGE_EVENT`` whose body reports
        ``state=Failed``) triggers a retry, up to ``max_retries``.
        Cause-specific strategies (NCCL, FileNotFound, …) are still
        opt-in via ``retry.strategies`` in environment.yaml; the broad
        default fits Skypilot's typical failure modes (cloud capacity
        flakes, transient distributed-training crashes, preempted spot
        VMs) where finer signals are rarely available without custom
        log parsers.

        Reads ``retry.delay_seconds`` from environment config for backoff
        between retry attempts (default: 0).
        """
        # Local import to avoid circular dependencies at module load.
        from gbserver.resilience.strategies.any_failure import AnyFailureRetryStrategy

        delay = 0.0
        if self.config is not None:
            delay = float(self.config.config.get("retry", {}).get("delay_seconds", 0))
        return [AnyFailureRetryStrategy(retry_delay_seconds=delay)]

    def _get_retry_test_scenario(self: Self) -> Optional[str]:
        """Scenario name used by ``_inject_event_to_trigger_retry_when_testing``.

        Returning a non-None value lets integration tests with
        ``simulate_step_failure: true`` (env var
        ``GBTEST_SIMULATE_FAILURE_SCENARIO=true``) inject a synthetic
        failure event, exercising the full retry path without an
        actual workload crash. Any scenario works for the default
        ``AnyFailureRetryStrategy`` since every canned payload in
        ``simulate.py`` is a ``MESSAGE_EVENT`` with ``state="Failed"``.
        """
        return "nccl_error"

    async def pullasset_hfstore(
        self: Self,
        uri: Optional[URI] = None,
        binding: Optional[Any] = None,
        storeload_config=None,
        assetstore=None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> tuple:
        """Pull an HF model/dataset/space onto the Skypilot cluster via the hfpull step.

        Resolves the local cache path, builds the canonical hfpull_config dict,
        and queues the builtin hfpull step (its Skypilot launcher uses ``hf
        download``).  Returns a binding dict whose ``path`` points at the cache
        location so downstream steps can consume the downloaded snapshot.

        When ``inline: true`` is set in the storeload config, the download is
        deferred to the main step's setup phase (no separate cluster launched).
        This is required for environments without shared filesystems (e.g. AWS).
        """
        from gbcommon.uri.hf import HfURI
        from gbserver.asset.hfstore import Hfstore
        from gbserver.environment.local_assets import get_hf_cache_dir

        assert isinstance(
            assetstore, Hfstore
        ), f"invalid assetstore: {type(assetstore).__name__} (expected 'Hfstore')"

        self._warn_non_default_mode(storeload_config, uri)

        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)  # type: ignore[arg-type]
        shared_workdir = (
            self.config.config.get("shared_workdir") if self.config else None
        )
        cache_dir = Path(
            get_hf_cache_dir(storeload_config, default_workdir=shared_workdir)
        )
        binding_path = (
            cache_dir / hfuri.get_owner() / hfuri.get_repo() / hfuri.get_revision()
        )

        hf_token = assetstore.resolve_token(hfuri) or ""
        binding_config = {"binding": {"path": str(binding_path)}}

        # Inline mode: stash download metadata for injection into the main
        # step's setup script rather than launching a separate cluster.
        inline = (
            storeload_config is not None
            and isinstance(getattr(storeload_config, "config", None), dict)
            and storeload_config.config.get("inline", False)
        )
        if inline:
            # Embed hfpull metadata in the binding_config so it flows per-step
            # through kwargs["bindings"] to _launch_skypilot_inner (no shared state).
            binding_config["_hfpull"] = {
                "path": str(binding_path),
                "repo": f"{hfuri.get_owner()}/{hfuri.get_repo()}",
                "revision": hfuri.get_revision(),
                "type": hfuri.get_hf_type() or "model",
                "uri": str(hfuri),
                "hf_token": hf_token,
            }
            logger.info(
                "pullasset_hfstore: inline mode — deferring download of %s to main step setup (dest=%s)",
                str(hfuri),
                binding_path,
            )
            return binding_config, None

        # Default: launch a separate hfpull step on its own cluster
        hfpull_config = Hfstore.build_hfpull_step_config(
            hfuri=hfuri,
            binding_path=str(binding_path),
        )

        hfpull_stepuri = "space://steps/hfpull"
        if (
            storeload_config is not None
            and storeload_config.config is not None
            and "step_uri" in storeload_config.config
        ):
            hfpull_stepuri = storeload_config.config["step_uri"]

        logger.info(
            "pullasset_hfstore: queuing hfpull step_uri=%s uri=%s dest=%s",
            hfpull_stepuri,
            str(hfuri),
            binding_path,
        )

        pull_step_config = BuildTargetStepConfig(
            step_uri=hfpull_stepuri,
            config={
                "hfpull_config": hfpull_config,
                "launcher_config": {"envs": {"HF_TOKEN": hf_token}},
            },
        )
        return binding_config, pull_step_config

    async def pushasset_hfstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config=None,
        uri: Optional[Union[str, URI]] = None,
        assetstore=None,
        output_config=None,
        **kwargs,
    ) -> BuildTargetStepConfig:
        """Push an artifact from the cluster to HuggingFace Hub via the hfpush step.

        Mirrors the K8s ``pushasset_hfstore`` resolution order for resource group
        and private fields, then queues the builtin hfpush step (its Skypilot
        launcher creates the repo via curl and uploads with ``hf upload``).
        """
        from gbcommon.uri.hf import HfURI
        from gbserver.asset.hfstore import Hfstore

        self._warn_non_default_mode(storepush_config, uri)
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")
        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)  # type: ignore[arg-type]
        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, actual: {type(binding)} {binding}"
        assert (
            "path" in binding
        ), f"expected 'path' to be in the binding, actual: {binding}"
        binding_path = binding["path"]

        assert isinstance(
            assetstore, Hfstore
        ), f"invalid assetstore: {type(assetstore).__name__} (expected 'Hfstore')"
        space_name = output_config.space_name if output_config else None
        # Enterprise/non-enterprise split + config precedence (environment-level
        # storepush_config, overridden by build.yaml store_push) + table-first
        # resolution all live in the shared helper. A non-Enterprise org
        # resolves to None with no HF API call.
        resource_group_id, hf_private, _hf_cfg = resolve_hfpush_resource_group_id(
            hfuri=hfuri,
            assetstore=assetstore,
            space_name=space_name,
            storepush_config=storepush_config,
            output_config=output_config,
        )

        hfpush_config = Hfstore.build_hfpush_step_config(
            hfuri=hfuri,
            binding_path=binding_path,
            binding_id=binding_id or "",
            hf_private=hf_private,
            hf_resource_group_id=resource_group_id,
        )
        # Apply remaining hf fields from the merged push config (strips
        # use_resource_group, re-asserts the resolved resource_group_id). Shared
        # with k8s so the overlay invariants cannot drift between the two.
        apply_hf_step_overlay(hfpush_config, _hf_cfg, resource_group_id)

        hf_token = assetstore.resolve_token(hfuri) or ""

        # Use a space:// URI so the resolver picks the env-keyed split
        # (`builtins/steps/<env-class>/hfpush/`) for the active env class.
        hfpush_stepuri = "space://steps/hfpush"
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            hfpush_stepuri = storepush_config.config["step_uri"]

        logger.info(
            "pushasset_hfstore: queuing hfpush step_uri=%s uri=%s source=%s",
            hfpush_stepuri,
            str(hfuri),
            binding_path,
        )

        return BuildTargetStepConfig(
            step_uri=hfpush_stepuri,
            config={
                "hfpush_config": hfpush_config,
                "launcher_config": {"envs": {"HF_TOKEN": hf_token}},
            },
        )

    async def pushasset_cosstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config=None,
        uri: Optional[Union[str, URI]] = None,
        assetstore=None,
        **kwargs,
    ) -> BuildTargetStepConfig:
        """Push artifact to S3/COS by queuing the builtin s3push step."""
        from gbcommon.uri.cos import CosURI
        from gbserver.asset.asset import Asset

        self._warn_non_default_mode(storepush_config, uri)
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received for pushasset: {binding}")

        cosuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(cosuri, CosURI), f"expected CosURI, got {type(cosuri)}"

        assert (
            isinstance(binding, dict) and "path" in binding
        ), f"expected binding dict with 'path', got {binding}"
        local_path = binding["path"]

        metadata = cosuri.get_metadata()
        bucket_path = metadata["bucket_path"]
        s3_uri = f"s3://{bucket_path}"

        cos_md = Asset(cosuri).get_metadata() if assetstore else {}
        cos_config = cos_md.get("config", cos_md) if cos_md else {}
        endpoint_url = cos_config.get("cos_endpoint", "") if cos_config else ""

        # Resolve AWS credentials from assetstore secrets, environment, or kwargs
        secrets = kwargs.get("secrets", {}) or {}
        aws_key_id = (
            secrets.get("AWS_ACCESS_KEY_ID")
            or secrets.get("COS_ACCESS_KEY_ID")
            or os.environ.get("AWS_ACCESS_KEY_ID", "")
        )
        aws_secret = (
            secrets.get("AWS_SECRET_ACCESS_KEY")
            or secrets.get("COS_SECRET_ACCESS_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        )

        s3push_config: Dict[str, Any] = {
            "s3push_config": {
                "local_path": local_path,
                "s3_uri": s3_uri,
                "endpoint_url": endpoint_url,
            },
            "launcher_config": {
                "envs": {
                    "AWS_ACCESS_KEY_ID": aws_key_id,
                    "AWS_SECRET_ACCESS_KEY": aws_secret,
                },
            },
        }

        s3push_stepuri = "file://" + str(
            Path(__file__).parent.parent / "builtins" / "steps" / "s3push"
        )
        if (
            storepush_config is not None
            and hasattr(storepush_config, "config")
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            s3push_stepuri = storepush_config.config["step_uri"]

        logger.info(
            "pushasset_cosstore: queuing s3push step_uri=%s local=%s s3=%s endpoint=%s",
            s3push_stepuri,
            local_path,
            s3_uri,
            endpoint_url,
        )

        return BuildTargetStepConfig(
            step_uri=s3push_stepuri,
            config=s3push_config,
        )
