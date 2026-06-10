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
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Self, Union

from tenacity import retry, stop_after_attempt, wait_exponential

from gbcommon.uri.uri import URI
from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.types.buildconfig import BuildTargetStepConfig
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.utils.logger import get_logger

if TYPE_CHECKING:
    from gbserver.resilience.retry_handler import RetryStrategy

logger = get_logger(__name__)

from gbserver.utils.optional_imports import HAS_SKYPILOT

if HAS_SKYPILOT:
    import sky
else:
    sky = None  # type: ignore[assignment]


def _require_skypilot():
    """Raise a clear error if the sky SDK is not installed."""
    if not HAS_SKYPILOT:
        raise ImportError(
            "The 'skypilot' package is required for the Skypilot environment. "
            "Install it with: pip install 'gbserver[skypilot]'"
        )


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


class Skypilot(Environment):
    """SkyPilot environment — provisions pods/VMs for step execution (unmanaged)."""

    def __init__(
        self: Self,
        event_q: asyncio.Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        secrets: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self._cluster_names: Dict[str, str] = {}  # launch_id -> cluster_name
        self._job_ids: Dict[str, int] = {}  # launch_id -> sky job_id
        self._setup_workdirs: Dict[str, str] = {}  # setup_id -> per-run workdir
        # launch_id -> kwargs replayed by retry_workload
        self._launch_kwargs: Dict[str, Dict] = {}
        self._skypilot_retry_complete_events: Dict[str, asyncio.Event] = {}
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
    def _cluster_name_for(launch_id: str) -> str:
        """Generate a unique cluster name from a launch_id."""
        return f"gb-{launch_id[:12]}"

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
            request_id = sky.launch(
                task,
                cluster_name=cluster_name,
                idle_minutes_to_autostop=0,
                down=True,
            )
            sky.stream_and_get(request_id)
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
        """
        try:
            _require_skypilot()

            # Stash kwargs so retry_workload can replay this launch.
            self._launch_kwargs[launch_id] = {
                "launcher_config": kwargs.get("launcher_config"),
                "config": kwargs.get("config"),
                "run_metadata": kwargs.get("run_metadata"),
                "setup_config": kwargs.get("setup_config"),
                "retry_enabled": kwargs.get("retry_enabled"),
                "retry_transparently": kwargs.get("retry_transparently"),
            }

            launcher_config = kwargs.get("launcher_config", {}) or {}
            config = kwargs.get("config", {}) or {}

            cluster_name = self._cluster_name_for(launch_id)
            cloud = (
                launcher_config.get("resources", {}).get("cloud") or self._get_cloud()
            )
            idle_minutes = launcher_config.get(
                "idle_minutes_to_autostop", self._get_idle_minutes()
            )

            # Build sky.Resources
            res_config = launcher_config.get("resources", {})

            # Build infra string: supports 'slurm/cluster/partition' format
            infra = res_config.get("infra") or cloud
            zone = res_config.get("zone")
            if not res_config.get("infra") and res_config.get("cluster"):
                infra = f"{cloud}/{res_config['cluster']}"
                if zone:
                    infra = f"{infra}/{zone}"
                    zone = None

            resources = sky.Resources(
                infra=infra,
                accelerators=res_config.get("accelerators"),
                cpus=res_config.get("cpus"),
                memory=res_config.get("memory"),
                disk_size=res_config.get("disk_size"),
                zone=zone,
                image_id=launcher_config.get("image_id"),
            )

            # Build environment variables
            env_vars: Dict[str, str] = {}
            if self.secrets:
                env_vars.update(self.secrets)
            env_vars.update(launcher_config.get("envs", {}))
            # Also pick up envs from config.launcher_config (for auto-queued steps)
            env_vars.update(config.get("launcher_config", {}).get("envs", {}))
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
                setup=launcher_config.get("setup") or None,
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
                "Launching SkyPilot cluster: name=%s cloud=%s resources=%s",
                cluster_name,
                cloud,
                res_config,
            )

            # SLURM does not support autostop; passing any non-None value
            # (including 0) fails provisioning with "Slurm does not support
            # autostop." Per-step `sky down` cleanup handles teardown anyway,
            # so force None on SLURM regardless of the user's config.
            cloud_for_infra = (str(infra).split("/", 1)[0] or "").lower()
            autostop = None if cloud_for_infra == "slurm" else idle_minutes

            # Launch and wait for provisioning
            request_id = sky.launch(
                task,
                cluster_name=cluster_name,
                idle_minutes_to_autostop=autostop,
            )
            job_id, handle = sky.stream_and_get(request_id)

            self._cluster_names[launch_id] = cluster_name
            if job_id is not None:
                self._job_ids[launch_id] = job_id

            logger.info(
                "SkyPilot cluster %s launched: job_id=%s launch_id=%s",
                cluster_name,
                job_id,
                launch_id,
            )

        except Exception as e:
            logger.error("Failed to launch SkyPilot cluster for %s: %s", launch_id, e)
            raise
        finally:
            self._release_monitors(launch_id)

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
        """
        _require_skypilot()
        retry_complete_event = asyncio.Event()
        self._skypilot_retry_complete_events[launch_id] = retry_complete_event

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
        ) as monitor_queue:
            try:
                while True:
                    retry_complete_event.clear()
                    await self._poll_skypilot_job(
                        launch_id=launch_id,
                        event_q=monitor_queue,
                        entityrun_metadata=entityrun_metadata,
                        event_configs=event_configs,
                        **kwargs,
                    )
                    if retry_complete_event.is_set():
                        # retry_workload re-launched and set this event;
                        # loop to poll the fresh cluster_name/job_id.
                        continue
                    return
            finally:
                self._skypilot_retry_complete_events.pop(launch_id, None)

    async def _poll_skypilot_job(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata=None,
        event_configs: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Poll ``sky.job_status`` for one launch attempt, emit events.

        Returns when the job reaches a terminal state or ``stop_event``
        is set. Emits a ``WORKLOAD_STATUS_EVENT(FAILED)`` on a non-success
        terminal state but does NOT raise — the upstream
        ``_with_retry_handler`` interprets the FAILED event and decides
        between retry and final-failure propagation.
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
        poll_interval = kwargs.get("poll_interval", 15)
        last_status = None

        while not stop_event.is_set():
            status = None
            poll_failed = False
            try:
                request_id = sky.job_status(
                    cluster_name,
                    job_ids=[job_id] if job_id is not None else None,
                )
                statuses = sky.get(request_id)
                status = statuses.get(job_id) if statuses else None
            except Exception as e:
                logger.error(
                    "Error polling SkyPilot job %s on %s: %s",
                    job_id,
                    cluster_name,
                    e,
                )
                poll_failed = True

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

            if status is not None and status.is_terminal():
                logger.info(
                    "SkyPilot job %s reached terminal status: %s",
                    job_id,
                    status,
                )
                if (
                    event_log_parser_configs
                    and event_q
                    and entityrun_metadata
                    and job_id is not None
                ):
                    await self._download_and_parse_logs(
                        cluster_name=cluster_name,
                        job_id=job_id,
                        launch_id=launch_id,
                        event_q=event_q,
                        entityrun_metadata=entityrun_metadata,
                        event_log_parser_configs=event_log_parser_configs,
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
                return

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                return  # stop_event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue polling

    async def _download_and_parse_logs(
        self: Self,
        cluster_name: str,
        job_id: int,
        launch_id: str,
        event_q: asyncio.Queue,
        entityrun_metadata,
        event_log_parser_configs: list,
    ) -> None:
        """Download job logs and parse for artifact events."""
        try:
            log_dir = _download_logs_with_retry(cluster_name, job_id)
            if not log_dir:
                logger.warning(
                    "No log directory returned for cluster %s job %s",
                    cluster_name,
                    job_id,
                )
                return

            log_dir = os.path.expanduser(log_dir)
            log_files = sorted(glob.glob(f"{log_dir}/*.log"))
            if not log_files:
                logger.info(
                    "No log files found in %s for cluster %s job %s",
                    log_dir,
                    cluster_name,
                    job_id,
                )
                return

            for log_file in log_files:
                try:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
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
            request_id = sky.down(cluster_name, purge=True)
            res = sky.get(request_id)
            logger.info("Torn down SkyPilot cluster %s, res=%s", cluster_name, res)
        except Exception as e:
            logger.error("Failed to tear down SkyPilot cluster %s: %s", cluster_name, e)
        finally:
            self._cluster_names.pop(launch_id, None)
            self._job_ids.pop(launch_id, None)
            self._launch_kwargs.pop(launch_id, None)

    async def retry_workload(
        self: Self,
        launch_id: str,
        nodes_to_avoid: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Retry a failed Skypilot workload via tear-down + relaunch.

        Called by ``RetryHandler`` when a strategy decides the failure is
        retriable. Stops the polling loop, takes the cluster down, and
        re-invokes ``launch_skypilot`` with the kwargs stashed during
        the first launch. Sets ``_skypilot_retry_complete_events[launch_id]``
        so ``monitor_skypilot_monitor``'s outer loop runs another
        iteration against the fresh cluster.

        :param launch_id: The launch identifier to retry.
        :param nodes_to_avoid: Currently logged-and-ignored — Skypilot
            has no portable per-launch node-exclusion knob.
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

        # Stop the polling loop cleanly before sky down.
        self._get_launch_stopped_event(launch_id).set()

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

        try:
            await self.launch_skypilot(launch_id, **original_kwargs)
        except Exception as launch_error:
            logger.error(
                "retry_workload could not relaunch launch_id=%s: %s",
                launch_id,
                launch_error,
            )
            raise

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
        """
        # Local import to avoid circular dependencies at module load.
        from gbserver.resilience.strategies.any_failure import AnyFailureRetryStrategy

        return [AnyFailureRetryStrategy()]

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

        hfpull_config = Hfstore.build_hfpull_step_config(
            hfuri=hfuri,
            binding_path=str(binding_path),
        )
        hf_token = assetstore.resolve_token(hfuri) or ""

        # Use a space:// URI so the resolver picks the env-keyed split
        # (`builtins/steps/<env-class>/hfpull/`) for the active env class.
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
        binding_config = {"binding": {"path": str(binding_path)}}
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
