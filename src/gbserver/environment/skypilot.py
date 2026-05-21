"""SkyPilot environment backend (unmanaged mode).

Manages build step execution on SkyPilot-provisioned pods/VMs using
sky.launch(). Each step gets its own cluster; pods auto-stop after
idle timeout. The sky SDK is lazy-imported so gbserver does not
require it unless a Skypilot environment is actually configured.
"""

import asyncio
import glob
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union

from tenacity import retry, stop_after_attempt, wait_exponential

from gbcommon.uri.uri import URI
from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.types.buildconfig import BuildTargetStepConfig
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

            # Build sky.Task
            task = sky.Task(
                name=cluster_name,
                setup=launcher_config.get("setup") or None,
                run=launcher_config.get("run", ""),
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

            # Launch and wait for provisioning
            request_id = sky.launch(
                task,
                cluster_name=cluster_name,
                idle_minutes_to_autostop=idle_minutes or None,
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
        """Monitor a SkyPilot job by polling sky.job_status().

        Polls the job status at intervals, translates SkyPilot JobStatus
        to BuildEvent objects, and puts them on event_q. Exits when the
        job reaches a terminal state or when stop_event is set.
        """
        _require_skypilot()

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
            try:
                request_id = sky.job_status(
                    cluster_name,
                    job_ids=[job_id] if job_id is not None else None,
                )
                statuses = sky.get(request_id)
                status = statuses.get(job_id) if statuses else None

                if status != last_status:
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
                    if (
                        event_q
                        and entityrun_metadata
                        and str(status) != "JobStatus.SUCCEEDED"
                    ):
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

            except Exception as e:
                logger.error(
                    "Error polling SkyPilot job %s on %s: %s",
                    job_id,
                    cluster_name,
                    e,
                )

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
            sky.get(request_id)
            logger.info("Torn down SkyPilot cluster %s", cluster_name)
        except Exception as e:
            logger.error("Failed to tear down SkyPilot cluster %s: %s", cluster_name, e)
        finally:
            self._cluster_names.pop(launch_id, None)
            self._job_ids.pop(launch_id, None)

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
        from gbserver.types.constants import CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI

        assert isinstance(
            assetstore, Hfstore
        ), f"invalid assetstore: {type(assetstore).__name__} (expected 'Hfstore')"

        if storeload_config is not None and storeload_config.mode not in (
            None,
            "hf_pull",
        ):
            raise ValueError(
                f"unsupported storeload mode: {storeload_config.mode}"
            )

        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)  # type: ignore[arg-type]
        cache_dir = Path(get_hf_cache_dir(storeload_config))
        binding_path = (
            cache_dir / hfuri.get_owner() / hfuri.get_repo() / hfuri.get_revision()
        )

        hfpull_config = Hfstore.build_hfpull_step_config(
            hfuri=hfuri,
            binding_path=str(binding_path),
        )
        hf_token = assetstore._resolve_token(hfuri) or ""

        hfpull_stepuri = CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI
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
        from gbserver.types.constants import CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI

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

        space_name = output_config.space_name if output_config else None

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
        if hf_resource_group_id:
            resource_group_id: Optional[str] = hf_resource_group_id
        else:
            resource_group_id = hfuri.resolve_resource_group_id(
                token=assetstore._resolve_token(hfuri),
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

        hf_token = assetstore._resolve_token(hfuri) or ""

        hfpush_stepuri = CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI
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
