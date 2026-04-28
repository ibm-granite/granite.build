"""RunPod GPU compute environment backend.

Manages build step execution on RunPod Pods (persistent GPU VMs).
The runpod SDK is lazy-imported so gbserver does not require it
unless a RunPod environment is actually configured.
"""

import asyncio
from typing import Any, Dict, Optional, Self

from gbserver.environment.environment import Environment
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.utils.logger import get_logger

# ---------------------------------------------------------------------------
# GPU type normalization and mapping
# ---------------------------------------------------------------------------


class UnknownGPUType(Exception):
    """Raised when a gpu_type cannot be resolved to a backend-native identifier."""


RUNPOD_GPU_MAP: dict[str, str] = {
    "A100-80GB": "NVIDIA A100 80GB PCIe",
    "A100-40GB": "NVIDIA A100-SXM4-40GB",
    "H100-80GB": "NVIDIA H100 80GB HBM3",
    "H100-SXM": "NVIDIA H100 SXM",
    "L40S": "NVIDIA L40S",
    "RTX-4090": "NVIDIA GeForce RTX 4090",
    "RTX-A6000": "NVIDIA RTX A6000",
    "A40": "NVIDIA A40",
}

_RUNPOD_NATIVE_IDS: set[str] = set(RUNPOD_GPU_MAP.values())


def resolve_runpod_gpu_type(gpu_type: str) -> str:
    """Resolve a normalized gpu_type to a RunPod-native gpu_type_id."""
    if gpu_type in _RUNPOD_NATIVE_IDS:
        return gpu_type
    gpu_type_upper = gpu_type.upper()
    for normalized, native in RUNPOD_GPU_MAP.items():
        if normalized.upper() == gpu_type_upper:
            return native
    raise UnknownGPUType(
        f"Unknown gpu_type '{gpu_type}'. "
        f"Valid normalized names: {sorted(RUNPOD_GPU_MAP.keys())}. "
        f"Or use a RunPod-native ID directly."
    )


logger = get_logger(__name__)


def _import_runpod():
    """Lazy import of the runpod SDK."""
    try:
        import runpod

        return runpod
    except ImportError as e:
        raise ImportError(
            "The 'runpod' package is required for the RunPod environment. "
            "Install it with: pip install runpod"
        ) from e


class Runpod(Environment):
    """A RunPod GPU compute environment."""

    def __init__(
        self: Self,
        event_q: asyncio.Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        secrets: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self.launched_pods: Dict[str, str] = {}  # launch_id -> pod_id
        super().__init__(
            event_q=event_q,
            environment_config=environment_config,
            secrets=secrets,
            **kwargs,
        )

    def _get_api_key(self: Self) -> str:
        """Resolve the RunPod API key from secrets."""
        if self.config is None:
            raise ValueError("environment_config is None")
        auth_config = self.config.config.get("authentication", {})
        api_key_name = auth_config.get("api_key", "RUNPOD_API_KEY")
        if self.secrets and api_key_name in self.secrets:
            return self.secrets[api_key_name]
        import os

        api_key = os.environ.get(api_key_name, "")
        if not api_key:
            raise ValueError(
                f"RunPod API key not found. Set the '{api_key_name}' secret "
                f"or environment variable."
            )
        logger.debug(
            "Using RunPod API key from environment variable '%s'", api_key_name
        )
        return api_key

    def _get_defaults(self: Self) -> Dict[str, Any]:
        """Get default pod configuration from environment.yaml."""
        if self.config is None:
            return {}
        return self.config.config.get("defaults", {})

    def _resolve_gpu_type(self: Self, compute_config: Optional[Dict] = None) -> str:
        """Resolve GPU type from compute_config or environment defaults."""
        defaults = self._get_defaults()
        gpu_type = None
        if compute_config:
            gpu_type = compute_config.get("gpu_type")
        if not gpu_type:
            gpu_type = defaults.get("gpu_type")
        if not gpu_type:
            raise ValueError(
                "No gpu_type specified in compute_config or environment defaults"
            )
        return resolve_runpod_gpu_type(gpu_type)

    async def launch_runpod(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir=None,
        environment_config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ) -> None:
        """Launch a step on a RunPod Pod.

        Creates a RunPod Pod with the specified Docker image and GPU type,
        waits for it to reach RUNNING status, then signals launch readiness via release_monitors().
        """
        try:
            runpod = _import_runpod()
            # NOTE: runpod.api_key is module-level state. This is safe as long as
            # all RunPod instances in a process share the same API key.
            runpod.api_key = self._get_api_key()

            launcher_config = kwargs.get("launcher_config", {}) or {}
            config = kwargs.get("config", {}) or {}
            compute_config = config.get("compute_config", {})
            step = kwargs.get("step", {})
            run_metadata = kwargs.get("run_metadata", {})

            # Resolve image
            image = launcher_config.get("image", "")
            if not image:
                raise ValueError("No 'image' specified in launcher_config")

            # Resolve GPU type
            gpu_type_id = self._resolve_gpu_type(compute_config)
            gpu_count = compute_config.get("num_gpus_per_node", 1)

            # Build pod name
            step_name = step.get("name", run_metadata.get("target_name", "gb"))
            pod_name = f"gb-{step_name}-{launch_id[:8]}"

            # Get defaults from environment.yaml
            defaults = self._get_defaults()
            cloud_type = defaults.get("cloud_type", "SECURE")
            container_disk_gb = defaults.get("container_disk_gb", 50)
            volume_gb = defaults.get("volume_gb", 100)
            volume_mount_path = defaults.get("volume_mount_path", "/workspace")

            # Build environment variables
            env_vars: Dict[str, str] = {}
            if environment_config:
                env_vars.update(environment_config.config.get("env", {}) or {})
            env_vars.update(launcher_config.get("env", {}))
            env_vars["LLMB_RUNPOD_LAUNCH_ID"] = launch_id
            env_vars["LLMB_RUNPOD_POD_NAME"] = pod_name

            # Container command
            command = launcher_config.get("command", "")

            logger.info(
                "Creating RunPod pod: name=%s image=%s gpu=%s gpu_count=%s",
                pod_name,
                image,
                gpu_type_id,
                gpu_count,
            )

            pod = runpod.create_pod(
                name=pod_name,
                image_name=image,
                gpu_type_id=gpu_type_id,
                gpu_count=gpu_count,
                cloud_type=cloud_type,
                container_disk_in_gb=container_disk_gb,
                volume_in_gb=volume_gb,
                volume_mount_path=volume_mount_path,
                docker_args=command,
                env=env_vars,
            )

            pod_id = pod.id if hasattr(pod, "id") else pod["id"]
            self.launched_pods[launch_id] = pod_id
            logger.info("Created RunPod pod: id=%s launch_id=%s", pod_id, launch_id)

            # Poll until pod is running
            await self._wait_for_pod_running(runpod, pod_id, launch_id)

        except Exception as e:
            logger.error("Failed to launch RunPod pod for %s: %s", launch_id, e)
            raise
        finally:
            self._release_monitors(launch_id)

    async def _wait_for_pod_running(
        self: Self, runpod, pod_id: str, launch_id: str, timeout: int = 600
    ) -> None:
        """Poll until the pod reaches RUNNING status or timeout."""
        import time

        start = time.monotonic()
        poll_interval = 5
        while time.monotonic() - start < timeout:
            pod_info = runpod.get_pod(pod_id)
            status = pod_info.get("desiredStatus", "UNKNOWN")
            runtime = pod_info.get("runtime")
            if runtime and runtime.get("uptimeInSeconds", 0) > 0:
                logger.info("RunPod pod %s is RUNNING", pod_id)
                return
            if status in ("EXITED", "TERMINATED", "ERROR"):
                raise RuntimeError(
                    f"RunPod pod {pod_id} reached terminal status: {status}"
                )
            logger.debug(
                "Waiting for RunPod pod %s (status=%s), polling in %ds",
                pod_id,
                status,
                poll_interval,
            )
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 30)
        raise TimeoutError(
            f"RunPod pod {pod_id} did not reach RUNNING within {timeout}s"
        )

    async def monitor_pod_status_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata=None,
        build_id: str = "",
        **kwargs,
    ) -> None:
        """Monitor a RunPod Pod's status by polling.

        Polls the pod status at intervals and emits status change events.
        Exits when the pod reaches a terminal state (EXITED, TERMINATED, ERROR)
        or when the stop_event is set.
        """
        runpod = _import_runpod()
        runpod.api_key = self._get_api_key()

        pod_id = self.launched_pods.get(launch_id)
        if not pod_id:
            logger.error("No pod_id for launch_id %s", launch_id)
            return

        stop_event = self._get_launch_stopped_event(launch_id)
        poll_interval = kwargs.get("poll_interval", 10)
        last_status = None
        terminal_states = {"EXITED", "TERMINATED", "ERROR"}

        while not stop_event.is_set():
            try:
                pod_info = runpod.get_pod(pod_id)
                status = pod_info.get("desiredStatus", "UNKNOWN")

                if status != last_status:
                    logger.info(
                        "RunPod pod %s status: %s -> %s (launch_id=%s)",
                        pod_id,
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
                                msg=f"RunPod pod {pod_id} status: {status}"
                            ),
                        )
                        await event_q.put(event)
                    last_status = status

                if status in terminal_states:
                    logger.info(
                        "RunPod pod %s reached terminal status: %s", pod_id, status
                    )
                    return

            except Exception as e:
                logger.error("Error polling RunPod pod %s: %s", pod_id, e)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
                return  # stop_event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue polling

    async def cleanup_runpod(
        self: Self,
        launch_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Terminate and remove a RunPod Pod."""
        if launch_id is None:
            logger.warning("cleanup_runpod called with no launch_id")
            return

        self._monitoring_cleanup(launch_id=launch_id)

        pod_id = self.launched_pods.get(launch_id)
        if not pod_id:
            logger.warning("No pod to cleanup for launch_id %s", launch_id)
            return

        try:
            runpod = _import_runpod()
            runpod.api_key = self._get_api_key()
            logger.info("Terminating RunPod pod %s (launch_id=%s)", pod_id, launch_id)
            runpod.terminate_pod(pod_id)
            logger.info("Terminated RunPod pod %s", pod_id)
        except Exception as e:
            logger.error("Failed to terminate RunPod pod %s: %s", pod_id, e)
        finally:
            self.launched_pods.pop(launch_id, None)
