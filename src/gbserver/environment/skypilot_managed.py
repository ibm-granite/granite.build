"""SkyPilot managed jobs environment backend.

Manages build step execution using SkyPilot's managed jobs controller
(sky.jobs.launch()). The controller runs in-cluster and handles
monitoring, recovery from pod evictions, and automatic restarts.
The gbserver process does not need to stay running for jobs to complete.
"""

import asyncio
import glob
import os
from typing import Any, Dict, List, Optional, Self

from tenacity import retry, stop_after_attempt, wait_exponential

from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.utils.logger import get_logger

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
            "The 'skypilot' package is required for the Skypilot_managed environment. "
            "Install it with: pip install 'gbserver[skypilot]'"
        )


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, max=128),
    reraise=True,
)
def _download_logs_with_retry(cluster_name: str, job_name: str):
    """Download SkyPilot managed job logs with retry for transient failures."""
    # sky.download_logs() returns Dict[str, str] mapping job_id to local log path
    # (it handles the API request/response internally, no sky.get() needed)
    result = sky.download_logs(cluster_name, job_ids=[job_name])
    return result.get(job_name)


from gbserver.environment._skypilot_ssh import (
    execute_on_host_via_ssh as _execute_on_host_via_ssh,
)
from gbserver.environment._skypilot_ssh import (
    extract_host_ssh_info as _extract_host_ssh_info,
)

# Shared file_mounts builder — keeps the unmanaged and managed launchers in sync
# (relative-source resolution against the step.yaml dir, bucket sub-path handling).
from gbserver.environment.skypilot import (
    _abort_shielded_request,
    _build_skypilot_mounts,
    _run_sky_verb_off_loop,
    _sky_submit_to_thread,
)


class Skypilot_managed(Environment):
    """SkyPilot environment — managed jobs with in-cluster controller."""

    def __init__(
        self: Self,
        event_q: asyncio.Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        secrets: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self._job_names: Dict[str, str] = {}  # launch_id -> managed job name
        super().__init__(
            event_q=event_q,
            environment_config=environment_config,
            secrets=secrets,
            **kwargs,
        )

    def _get_cloud(self: Self) -> str:
        if self.config is None:
            return "k8s"
        return self.config.config.get("default_cloud", "k8s")

    def _get_idle_minutes(self: Self) -> int:
        if self.config is None:
            return 10
        return self.config.config.get("idle_minutes_to_autostop", 10)

    @staticmethod
    def _job_name_for(launch_id: str) -> str:
        """Generate a unique managed job name from a launch_id."""
        return f"gb-{launch_id[:12]}"

    def get_launch_env_vars(
        self: Self,
        run_metadata: Optional[Dict[str, Any]] = None,
        launcher_config: Optional[Dict] = None,
        launch_id: str = "",
        job_name: str = "",
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Build the full env dict for a skypilot managed-job launch.

        Precedence (lowest->highest): secrets < launcher ``envs`` < the
        built-in ``GB_SKYPILOT_*`` vars < the standard cross-environment set
        from ``super()`` (GBTEST_ test-control vars + e.g. GB_BUILD_ID), which
        is authoritative.

        :param run_metadata: launch run_metadata; forwarded to ``super()`` and
            the source of GB_TARGETRUN_ID.
        :param launcher_config: step.yaml launcher config (its ``envs``).
        :param launch_id: unique id for this launch (GB_SKYPILOT_LAUNCH_ID).
        :param job_name: the managed job name (GB_SKYPILOT_JOB_NAME).
        :returns: the complete ``{name: value}`` env dict for the sky.Task.
        """
        launcher_config = launcher_config or {}
        run_metadata = run_metadata or {}
        env: Dict[str, str] = {}
        if self.secrets:
            env.update(self.secrets)
        env.update(launcher_config.get("envs", {}))
        env["GB_SKYPILOT_LAUNCH_ID"] = launch_id
        env["GB_SKYPILOT_JOB_NAME"] = job_name
        if run_metadata.get("targetrun_id"):
            env["GB_TARGETRUN_ID"] = run_metadata["targetrun_id"]
        env.update(super().get_launch_env_vars(run_metadata=run_metadata))
        # Uniform with the other environments; a no-op here since skypilot's
        # launcher vars are already GB_-prefixed (GB_SKYPILOT_*), so there are no
        # LLMB_ names to mirror.
        return self._add_gb_aliases(env)

    async def launch_skypilot_managed(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir=None,
        environment_config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ) -> None:
        """Launch a step as a SkyPilot managed job.

        Submits the task to SkyPilot's managed jobs controller, which
        handles monitoring, recovery, and restarts independently.
        """
        try:
            _require_skypilot()

            launcher_config = kwargs.get("launcher_config", {}) or {}
            config = kwargs.get("config", {}) or {}
            # Kept as a local: reused by the post-launch-failure event below.
            run_metadata = kwargs.get("run_metadata", {})

            job_name = self._job_name_for(launch_id)
            cloud = (
                launcher_config.get("resources", {}).get("cloud") or self._get_cloud()
            )

            # Build sky.Resources
            res_config = launcher_config.get("resources", {})

            # Build cluster config overrides (docker run_options, etc.)
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

            resources = sky.Resources(
                infra=cloud,
                accelerators=res_config.get("accelerators"),
                cpus=res_config.get("cpus"),
                memory=res_config.get("memory"),
                disk_size=res_config.get("disk_size"),
                image_id=image_id,
                _cluster_config_overrides=cluster_config_overrides or None,
            )

            # Build the full env for the sky.Task. GB_BUILD_ID (and any future
            # standard var) comes from Environment.get_launch_env_vars() and is
            # authoritative over launcher/config env.
            env_vars = self.get_launch_env_vars(
                run_metadata=run_metadata,
                launcher_config=launcher_config,
                launch_id=launch_id,
                job_name=job_name,
            )

            # Build sky.Task
            task = sky.Task(
                name=job_name,
                setup=launcher_config.get("setup") or None,
                run=launcher_config.get("run", ""),
                envs=env_vars if env_vars else None,
                resources=resources,
            )

            # Handle file_mounts (may be in launcher config or step config).
            # Relative local sources resolve against targetsteprun_asset_dir (the
            # dir holding the rendered step.yaml + siblings); shared with the
            # unmanaged launcher via _build_skypilot_mounts.
            file_mounts_raw = launcher_config.get("file_mounts") or config.get(
                "file_mounts"
            )
            if file_mounts_raw:
                # build_workdir is intentionally omitted: the managed launcher
                # does not implement the shared_workdir / GB_BUILD_WORKDIR scheme
                # (no _compute_run_workdir, no per-run workdir export/cd), so
                # there is no per-run dir to remap relative destinations into.
                # Relative destinations therefore keep SkyPilot's default
                # handling here, unlike the unmanaged launcher.
                file_mounts, storage_mounts = _build_skypilot_mounts(
                    file_mounts_raw, targetsteprun_asset_dir
                )
                if file_mounts:
                    task.set_file_mounts(file_mounts)
                if storage_mounts:
                    task.set_storage_mounts(storage_mounts)

            logger.info(
                "Launching SkyPilot managed job: name=%s cloud=%s resources=%s",
                job_name,
                cloud,
                res_config,
            )

            # Both sky.jobs.launch and sky.stream_and_get are blocking calls.
            # Running them directly on the event loop freezes it (defeating any
            # enclosing wait_for) and makes cancellation impossible, so offload
            # them to threads. Each blocks in an OS thread (the submit while the
            # jobs controller provisions, stream_and_get while the job runs), so
            # a CancelledError delivered to this task would be deferred until the
            # thread returns. Shield each future so the outer await observes the
            # cancel *immediately* while the thread keeps running, letting us
            # abort the request server-side and cancel the job by name (mirrors
            # the unmanaged launcher's _provision_with_retry).
            launch_fut = asyncio.ensure_future(
                _sky_submit_to_thread(sky.jobs.launch, task, name=job_name)
            )
            try:
                request_id = await asyncio.shield(launch_fut)
            except asyncio.CancelledError:
                # Cancelled during the submit: no request_id yet, but the
                # controller may already be accepting the job. _abort_managed_launch
                # recovers the request_id by draining the submit and cancels the
                # job by its deterministic name so it is not left running.
                await self._abort_managed_launch(None, job_name, launch_fut)
                raise
            stream_fut = asyncio.ensure_future(
                _sky_submit_to_thread(sky.stream_and_get, request_id)
            )
            try:
                await asyncio.shield(stream_fut)
            except asyncio.CancelledError:
                await self._abort_managed_launch(request_id, job_name, stream_fut)
                raise

            self._job_names[launch_id] = job_name
            logger.info(
                "SkyPilot managed job %s submitted (launch_id=%s)",
                job_name,
                launch_id,
            )

            # Execute post-launch tasks (e.g., start evaluator sidecars) if defined
            post_launch_task = launcher_config.get("post_launch_task")
            if post_launch_task:
                try:
                    logger.info(
                        "Executing post-launch task on managed job %s (launch_id=%s)",
                        job_name,
                        launch_id,
                    )
                    host_ip, ssh_key = await asyncio.to_thread(
                        _extract_host_ssh_info, job_name
                    )
                    await _execute_on_host_via_ssh(
                        host_ip=host_ip,
                        ssh_key=ssh_key,
                        commands=post_launch_task.get("run", ""),
                        env_vars=env_vars,
                    )
                    logger.info(
                        "Post-launch task completed on managed job %s (launch_id=%s)",
                        job_name,
                        launch_id,
                    )
                except Exception as e:
                    logger.error(
                        "Post-launch task failed on managed job %s (launch_id=%s): %s",
                        job_name,
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
                                    msg=f"Post-launch task failed on {job_name}: {e}"
                                ),
                            )
                        )

        except Exception as e:
            logger.error(
                "Failed to launch SkyPilot managed job for %s: %s", launch_id, e
            )
            raise
        finally:
            self._release_monitors(launch_id)

    async def _abort_managed_launch(
        self: Self,
        request_id: Any,
        job_name: str,
        pending_fut: Optional["asyncio.Future"],
    ) -> None:
        """Abort an in-flight managed-job launch after cancellation.

        Thin wrapper over the shared ``_abort_shielded_request`` whose only
        launcher-specific part is the reclaim: cancel the managed job by its
        deterministic name so a job the controller already accepted does not
        keep running. ``self._job_names`` is not populated until the launch
        returns, so ``cleanup_skypilot_managed`` cannot find the job on the
        cancel path — cancelling by name here is the safety net, and also covers
        the case where the submit was cancelled before it returned a request_id.

        Args:
            request_id: The id from ``sky.jobs.launch``, or ``None`` if cancelled
                during the submit (recovered by draining ``pending_fut``).
            job_name: Deterministic managed-job name to cancel.
            pending_fut: The shielded ``to_thread`` future to drain (the submit
                or the ``sky.stream_and_get`` wait).
        """

        async def _cancel_job() -> None:
            # Cancel the managed job by name in case the controller already
            # accepted it before the launch request was aborted.
            try:
                await _run_sky_verb_off_loop(sky.jobs.cancel, name=job_name)
            except Exception as e:
                logger.warning("jobs.cancel for %s failed: %s", job_name, e)

        await _abort_shielded_request(
            request_id,
            pending_fut,
            description=f"SkyPilot managed job {job_name}",
            on_abort=_cancel_job,
        )

    async def monitor_skypilot_managed_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata=None,
        build_id: str = "",
        event_configs: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Monitor a SkyPilot managed job by polling sky.jobs.queue().

        The managed jobs controller handles actual monitoring; this method
        polls for status updates, translates them to BuildEvents, and
        parses logs for artifact events after terminal status.
        """
        _require_skypilot()

        event_log_parser_configs = []
        if event_configs is not None:
            event_log_parser_configs = [
                EventLogLineParserConfig.model_validate(config)
                for config in event_configs
            ]

        job_name = self._job_names.get(launch_id)
        if not job_name:
            logger.error("No job_name for launch_id %s", launch_id)
            return

        stop_event = self._get_launch_stopped_event(launch_id)
        poll_interval = kwargs.get("poll_interval", 30)
        last_status = None
        cluster_name = None

        while not stop_event.is_set():
            try:
                request_id = sky.jobs.queue(refresh=False)
                jobs = sky.get(request_id)

                status = None
                if jobs:
                    for job in jobs:
                        if job.get("name") == job_name:
                            status = job.get("status")
                            cluster_name = job.get("cluster_name")
                            break

                if status != last_status:
                    logger.info(
                        "SkyPilot managed job %s status: %s -> %s (launch_id=%s)",
                        job_name,
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
                                msg=f"SkyPilot managed job {job_name}: {status}"
                            ),
                        )
                        await event_q.put(event)
                    last_status = status

                if (
                    status is not None
                    and hasattr(status, "is_terminal")
                    and status.is_terminal()
                ):
                    logger.info(
                        "SkyPilot managed job %s reached terminal status: %s",
                        job_name,
                        status,
                    )
                    if event_log_parser_configs and event_q and entityrun_metadata:
                        if cluster_name:
                            await self._download_and_parse_logs(
                                cluster_name=cluster_name,
                                job_name=job_name,
                                launch_id=launch_id,
                                event_q=event_q,
                                entityrun_metadata=entityrun_metadata,
                                event_log_parser_configs=event_log_parser_configs,
                            )
                        else:
                            logger.warning(
                                "event_configs provided but no cluster_name available "
                                "for managed job %s (launch_id=%s); skipping log parsing",
                                job_name,
                                launch_id,
                            )
                    if str(status) != "ManagedJobStatus.SUCCEEDED":
                        raise RuntimeError(
                            f"SkyPilot managed job {job_name} ended with "
                            f"status {status}"
                        )
                    return

            except RuntimeError:
                raise
            except Exception as e:
                logger.error("Error polling SkyPilot managed job %s: %s", job_name, e)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                return
            except asyncio.TimeoutError:
                pass

    async def _download_and_parse_logs(
        self: Self,
        cluster_name: str,
        job_name: str,
        launch_id: str,
        event_q: asyncio.Queue,
        entityrun_metadata,
        event_log_parser_configs: list,
    ) -> None:
        """Download managed job logs and parse for artifact events."""
        try:
            log_dir = _download_logs_with_retry(cluster_name, job_name)
            if not log_dir:
                logger.warning(
                    "No log directory returned for cluster %s job %s",
                    cluster_name,
                    job_name,
                )
                return

            log_dir = os.path.expanduser(log_dir)
            log_files = sorted(glob.glob(f"{log_dir}/*.log"))
            if not log_files:
                logger.info(
                    "No log files found in %s for cluster %s job %s",
                    log_dir,
                    cluster_name,
                    job_name,
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
                job_name,
                launch_id,
                e,
            )

    async def cleanup_skypilot_managed(
        self: Self,
        launch_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Cancel a SkyPilot managed job."""
        if launch_id is None:
            logger.warning("cleanup_skypilot_managed called with no launch_id")
            return

        self._monitoring_cleanup(launch_id=launch_id)

        job_name = self._job_names.get(launch_id)
        if not job_name:
            logger.warning("No managed job to cleanup for launch_id %s", launch_id)
            return

        try:
            _require_skypilot()
            logger.info(
                "Cancelling SkyPilot managed job %s (launch_id=%s)",
                job_name,
                launch_id,
            )
            await _run_sky_verb_off_loop(sky.jobs.cancel, name=job_name)
            logger.info("Cancelled SkyPilot managed job %s", job_name)
        except Exception as e:
            logger.error("Failed to cancel SkyPilot managed job %s: %s", job_name, e)
        finally:
            self._job_names.pop(launch_id, None)
