"""SkyPilot environment backend (unmanaged mode).

Manages build step execution on SkyPilot-provisioned pods/VMs using
sky.launch(). Each step gets its own cluster; pods auto-stop after
idle timeout. The sky SDK is lazy-imported so gbserver does not
require it unless a Skypilot environment is actually configured.
"""

import asyncio
import glob
import os
import shlex
import threading
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Self, Tuple, Union

from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from gbcommon.types.testing import get_exported_gbtest_env_vars
from gbcommon.uri.uri import URI
from gbserver.environment.environment import Environment, EventLogLineParserConfig
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
            _ensure_skypilot_api_running()

            # Stash kwargs so retry_workload can replay this launch.
            self._launch_kwargs[launch_id] = {
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

            # Build sky.Resources — merge build-level overrides on top of
            # step defaults (config.launcher_config.resources wins over
            # step's environment_configs.*.launchers.*.config.resources)
            res_config = {
                **launcher_config.get("resources", {}),
                **config.get("launcher_config", {}).get("resources", {}),
            }

            # Build infra string: supports 'cloud/cluster/partition' format
            # (e.g., 'slurm/mycluster/gpu', 'lsf/bluevela/normal')
            infra = res_config.get("infra") or cloud
            zone = res_config.get("zone")
            if not res_config.get("infra") and res_config.get("cluster"):
                infra = f"{cloud}/{res_config['cluster']}"
                if zone:
                    infra = f"{infra}/{zone}"
                    zone = None
            elif not res_config.get("infra") and zone:
                # zone without cluster — fold into infra to avoid the
                # "cannot specify both infra and zone" error in sky.Resources
                infra = f"{infra}/{zone}" if infra else zone
                zone = None

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

            image_id = config.get("launcher_config", {}).get(
                "image_id"
            ) or launcher_config.get("image_id")

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

            # Build environment variables
            env_vars: Dict[str, str] = {}
            if self.secrets:
                env_vars.update(self.secrets)
            env_vars.update(launcher_config.get("envs", {}))
            # Also pick up envs from config.launcher_config (for auto-queued steps)
            env_vars.update(config.get("launcher_config", {}).get("envs", {}))
            # Forward GBTEST_ test-control vars (e.g. GBTEST_MOCKED_HF_OPS) to the
            # remote run so hfpull/hfpush steps honor mocking on the cluster.
            env_vars.update(get_exported_gbtest_env_vars())
            env_vars["GB_SKYPILOT_LAUNCH_ID"] = launch_id
            env_vars["GB_SKYPILOT_CLUSTER_NAME"] = cluster_name
            # Expose run metadata so steps in the same target can share state
            run_metadata = kwargs.get("run_metadata", {})
            if run_metadata.get("targetrun_id"):
                env_vars["GB_TARGETRUN_ID"] = run_metadata["targetrun_id"]
            if run_metadata.get("build_id"):
                env_vars["GB_BUILD_ID"] = run_metadata["build_id"]
            # Expose the env-level shared workdir so steps can stage cross-step
            # state under a path that is mounted on every worker.
            shared_workdir = (
                self.config.config.get("shared_workdir") if self.config else None
            )
            if shared_workdir:
                env_vars["GB_SHARED_WORKDIR"] = shared_workdir

            # Per-run workdir provisioned by setup_skypilot. When present,
            # export it as GB_BUILD_WORKDIR and make it the initial CWD of
            # the run script so step authors can write outputs with
            # relative paths and get implicit per-run isolation.
            build_workdir = (
                kwargs.get("setup_config", {}).get("skypilot", {}).get("build_workdir")
            )
            if build_workdir:
                env_vars["GB_BUILD_WORKDIR"] = build_workdir

            # Inject inline hfpull downloads into setup from per-step bindings
            setup_script = launcher_config.get("setup") or ""
            pending_hfpulls = {}
            for bid, bval in (kwargs.get("bindings") or {}).items():
                if isinstance(bval, dict) and "_hfpull" in bval:
                    pending_hfpulls[bid] = bval["_hfpull"]
            if pending_hfpulls:
                # Inject HF_TOKEN into env vars if any pull provides one
                # (hf download picks it up automatically from the environment)
                for pull_info in pending_hfpulls.values():
                    if pull_info.get("hf_token") and "HF_TOKEN" not in env_vars:
                        env_vars["HF_TOKEN"] = pull_info["hf_token"]
                        break
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

            run_script = launcher_config.get("run", "")
            if build_workdir:
                run_script = (
                    'mkdir -p "$GB_BUILD_WORKDIR"\n'
                    'cd "$GB_BUILD_WORKDIR"\n'
                    f"{run_script}"
                )

            # Build sky.Task
            task = sky.Task(
                name=cluster_name,
                setup=setup_script or None,
                run=run_script,
                envs=env_vars if env_vars else None,
                resources=resources,
            )

            # Handle file_mounts (may be in launcher config or step config)
            # Dict values → sky.Storage (set_storage_mounts), strings → set_file_mounts
            file_mounts_raw = launcher_config.get("file_mounts") or config.get(
                "file_mounts"
            )
            if file_mounts_raw:
                file_mounts = {}
                storage_mounts = {}
                for mount_path, mount_val in file_mounts_raw.items():
                    if isinstance(mount_val, dict):
                        mode_str = mount_val.get("mode", "MOUNT").upper()
                        source = mount_val["source"]
                        storage_kwargs: Dict[str, Any] = {
                            "mode": sky.StorageMode[mode_str],
                        }
                        # MOUNT mode requires bucket-only source; extract
                        # sub-path for URIs like s3://bucket/prefix
                        parsed = urllib.parse.urlparse(source)
                        sub_path = parsed.path.lstrip("/")
                        if sub_path:
                            storage_kwargs["source"] = (
                                f"{parsed.scheme}://{parsed.netloc}"
                            )
                            storage_kwargs["_bucket_sub_path"] = sub_path
                        else:
                            storage_kwargs["source"] = source
                        storage_mounts[mount_path] = sky.Storage(**storage_kwargs)
                    else:
                        file_mounts[mount_path] = mount_val
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
            # backends regardless of the user's config.
            cloud_for_infra = (str(infra).split("/", 1)[0] or "").lower()
            no_autostop_clouds = ("slurm", "lsf")
            autostop = None if cloud_for_infra in no_autostop_clouds else idle_minutes

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
                    request_id = await asyncio.to_thread(
                        sky.launch,
                        task,
                        cluster_name=cluster_name,
                        idle_minutes_to_autostop=autostop,
                    )
                    return await asyncio.to_thread(sky.stream_and_get, request_id)
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
            "poll_interval_seconds", kwargs.get("poll_interval", 900)
        )
        try:
            poll_interval = float(_raw_poll)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid poll_interval_seconds %r; falling back to 900s", _raw_poll
            )
            poll_interval = 900.0
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

        if storeload_config is not None and storeload_config.mode not in (
            None,
            "hf_pull",
        ):
            raise ValueError(f"unsupported storeload mode: {storeload_config.mode}")

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

    async def pullasset_envstore(
        self: Self,
        uri: Optional[Union[str, URI]] = None,
        binding: Optional[Any] = None,
        storeload_config=None,
        **kwargs,
    ) -> Tuple[Dict, Optional[Any]]:
        """Pull asset for env:// store — artifact is already on shared FS.

        No-op pull: the path is directly accessible on the shared filesystem.
        Returns the binding with the path extracted from the URI.
        """
        path = str(uri).replace("env://", "") if uri else ""
        logger.info(
            "pullasset_envstore: artifact at path=%s (shared FS, no transfer needed)",
            path,
        )
        binding_config = {"binding": {"path": path}}
        return binding_config, None

    async def pushasset_envstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config=None,
        uri: Optional[Union[str, URI]] = None,
        assetstore=None,
        secrets: Optional[Dict[str, str]] = None,
        run_metadata: Optional[Any] = None,
        output_config: Optional[Any] = None,
    ) -> URI:
        """Push asset for env:// store — artifact is already on shared FS.

        No-op push: the artifact path from the container is directly
        accessible on the shared filesystem, so no transfer is needed.
        """
        if not uri:
            raise ValueError(
                f"pushasset_envstore: empty uri for binding={binding_id!r}; "
                "an env:// store push requires a concrete artifact path."
            )
        logger.info(
            "pushasset_envstore: registering artifact %s at uri=%s binding=%s",
            binding_id,
            uri,
            binding,
        )
        return URI.get_uri(str(uri))

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

        hf_resource_group_id = None
        hf_resource_group_name = None
        hf_private = True
        if output_config is not None and output_config.store_push is not None:
            hf_cfg = output_config.store_push.config.get("hf", {})
            hf_resource_group_id = hf_cfg.get("resource_group_id", hf_resource_group_id)
            hf_resource_group_name = hf_cfg.get(
                "resource_group_name", hf_resource_group_name
            )
            hf_private = hf_cfg.get("private", hf_private)

        assert isinstance(
            assetstore, Hfstore
        ), f"invalid assetstore: {type(assetstore).__name__} (expected 'Hfstore')"
        space_name = output_config.space_name if output_config else None
        if hf_resource_group_id:
            resource_group_id: Optional[str] = hf_resource_group_id
        else:
            resource_group_id = hfuri.resolve_resource_group_id(
                token=assetstore.resolve_token(hfuri),
                resource_group_name=hf_resource_group_name,
                space_name=space_name,
            )

        hfpush_config = Hfstore.build_hfpush_step_config(
            hfuri=hfuri,
            binding_path=binding_path,
            binding_id=binding_id or "",
            hf_private=hf_private,
            hf_resource_group_id=resource_group_id,
        )
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "hf" in storepush_config.config
        ):
            hfpush_config["hf"].update(storepush_config.config["hf"])
        if (
            output_config is not None
            and output_config.store_push is not None
            and "hf" in output_config.store_push.config
        ):
            hfpush_config["hf"].update(output_config.store_push.config["hf"])

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
