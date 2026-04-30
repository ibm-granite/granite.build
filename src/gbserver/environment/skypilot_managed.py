"""SkyPilot managed jobs environment backend.

Manages build step execution using SkyPilot's managed jobs controller
(sky.jobs.launch()). The controller runs in-cluster and handles
monitoring, recovery from pod evictions, and automatic restarts.
The gbserver process does not need to stay running for jobs to complete.
"""

import asyncio
import glob
import os
import urllib.parse
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

            job_name = self._job_name_for(launch_id)
            cloud = launcher_config.get("resources", {}).get("cloud") or self._get_cloud()

            # Build sky.Resources
            res_config = launcher_config.get("resources", {})
            resources = sky.Resources(
                infra=cloud,
                accelerators=res_config.get("accelerators"),
                cpus=res_config.get("cpus"),
                memory=res_config.get("memory"),
                disk_size=res_config.get("disk_size"),
                image_id=launcher_config.get("image_id"),
            )

            # Build environment variables
            env_vars: Dict[str, str] = {}
            if self.secrets:
                env_vars.update(self.secrets)
            env_vars.update(launcher_config.get("envs", {}))
            env_vars["GB_SKYPILOT_LAUNCH_ID"] = launch_id
            env_vars["GB_SKYPILOT_JOB_NAME"] = job_name
            # Expose run metadata so steps in the same target can share state
            run_metadata = kwargs.get("run_metadata", {})
            if run_metadata.get("targetrun_id"):
                env_vars["GB_TARGETRUN_ID"] = run_metadata["targetrun_id"]
            if run_metadata.get("build_id"):
                env_vars["GB_BUILD_ID"] = run_metadata["build_id"]

            # Build sky.Task
            task = sky.Task(
                name=job_name,
                setup=launcher_config.get("setup") or None,
                run=launcher_config.get("run", ""),
                envs=env_vars if env_vars else None,
                resources=resources,
            )

            # Handle file_mounts (may be in launcher config or step config)
            # Dict values → sky.Storage (set_storage_mounts), strings → set_file_mounts
            file_mounts_raw = launcher_config.get("file_mounts") or config.get("file_mounts")
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
                            storage_kwargs["source"] = f"{parsed.scheme}://{parsed.netloc}"
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
                "Launching SkyPilot managed job: name=%s cloud=%s resources=%s",
                job_name,
                cloud,
                res_config,
            )

            request_id = sky.jobs.launch(task, name=job_name)
            sky.stream_and_get(request_id)

            self._job_names[launch_id] = job_name
            logger.info(
                "SkyPilot managed job %s submitted (launch_id=%s)",
                job_name,
                launch_id,
            )

        except Exception as e:
            logger.error("Failed to launch SkyPilot managed job for %s: %s", launch_id, e)
            raise
        finally:
            self._release_monitors(launch_id)

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
                EventLogLineParserConfig.model_validate(config) for config in event_configs
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

                if status is not None and hasattr(status, "is_terminal") and status.is_terminal():
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
                            f"SkyPilot managed job {job_name} ended with " f"status {status}"
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
            request_id = sky.jobs.cancel(name=job_name)
            sky.get(request_id)
            logger.info("Cancelled SkyPilot managed job %s", job_name)
        except Exception as e:
            logger.error("Failed to cancel SkyPilot managed job %s: %s", job_name, e)
        finally:
            self._job_names.pop(launch_id, None)
