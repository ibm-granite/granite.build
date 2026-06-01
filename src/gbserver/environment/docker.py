"""Docker container execution environment backend.

Manages build step execution in local Docker containers.
The docker SDK is lazy-imported so gbserver does not require it
unless a Docker environment is actually configured.

Compatible with Podman via the Docker-compatible API socket.
Set DOCKER_HOST to point to Podman's socket:
  export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock
"""

import asyncio
import functools
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple

from gbcommon.uri.uri import URI
from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.environment.local_assets import (
    get_hf_cache_dir,
    pull_asset_hfstore,
    push_asset_hfstore,
)
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
)
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.types.errors import LogMonitoringFailedException
from gbserver.utils.filesystem import sync_or_copy
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def _import_docker():
    """Lazy import of the docker SDK."""
    try:
        import docker

        return docker
    except ImportError as e:
        raise ImportError(
            "The 'docker' package is required for the Docker environment. "
            "Install it with: pip install 'gbserver[docker]'"
        ) from e


def _find_podman_socket() -> Optional[str]:
    """Ask the local Podman installation for its active machine socket path.

    Runs ``podman machine inspect`` and extracts the socket path so callers
    do not need to hard-code session-specific ``/var/folders/…`` paths.

    Returns:
        A ``unix://`` URL string if Podman is installed and a machine is
        running, or ``None`` if the command fails or the socket does not exist.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "podman",
                "machine",
                "inspect",
                "--format",
                "{{.ConnectionInfo.PodmanSocket.Path}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            if path and Path(path).exists():
                return f"unix://{path}"
    except Exception:
        pass
    return None


def _connect_docker_client(docker_module):
    """Connect to Docker/Podman, falling back to Podman socket discovery if needed.

    Tries ``docker.from_env()`` first (honours ``DOCKER_HOST``), then queries
    ``podman machine inspect`` for the active socket path, then tries
    ``/var/run/docker.sock`` as a last resort.

    Args:
        docker_module: The imported ``docker`` module.

    Returns:
        A connected ``DockerClient`` instance.

    Raises:
        RuntimeError: If no Docker/Podman daemon can be reached.
    """
    try:
        return docker_module.from_env()
    except docker_module.errors.DockerException:
        pass

    for socket_url in filter(
        None, [_find_podman_socket(), "unix:///var/run/docker.sock"]
    ):
        try:
            client = docker_module.DockerClient(base_url=socket_url)
            client.ping()
            return client
        except Exception:
            continue

    raise RuntimeError(
        "Could not connect to Docker/Podman daemon. "
        "Ensure Docker Desktop or Podman is running. "
        "Set DOCKER_HOST to point to the correct socket if needed "
        "(e.g. export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock)."
    )


class Docker(Environment):
    """A Docker container execution environment."""

    def __init__(
        self: Self,
        event_q: asyncio.Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        secrets: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self._launched_containers: Dict[str, str] = {}  # launch_id -> container_id
        self._launched_workspaces: Dict[str, str] = (
            {}
        )  # launch_id -> host workspace dir
        self._extra_volumes: Dict[str, Dict] = {}
        self._docker_module = None  # cached docker module
        self._docker_client = None  # cached Docker client
        super().__init__(
            event_q=event_q,
            environment_config=environment_config,
            secrets=secrets,
            **kwargs,
        )

    def _get_docker(self: Self):
        """Get or create cached docker module and client."""
        if self._docker_module is None:
            self._docker_module = _import_docker()
        if self._docker_client is None:
            self._docker_client = _connect_docker_client(self._docker_module)
        return self._docker_module, self._docker_client

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_defaults(self: Self) -> Dict:
        """Get default configuration from environment.yaml."""
        if self.config is None:
            return {}
        return self.config.config.get("defaults", {})

    @staticmethod
    def _parse_memory(memory_str: str) -> Optional[str]:
        """Convert memory strings like '32Gi' to docker-compatible format."""
        if not memory_str:
            return None
        memory_str = memory_str.strip()
        if memory_str.endswith("Gi"):
            return memory_str[:-2] + "g"
        if memory_str.endswith("Mi"):
            return memory_str[:-2] + "m"
        return memory_str

    def _resolve_image(self: Self, launcher_config: Dict, docker_config: Dict) -> str:
        """Resolve image from launcher_config, docker config, or environment defaults."""
        image = launcher_config.get("image", "")
        if not image:
            image = docker_config.get("image", "")
        if not image:
            image = self._get_defaults().get("image", "")
        if not image:
            raise ValueError(
                "No Docker image specified in launcher_config, step docker config, "
                "or environment defaults"
            )
        return image

    def _resolve_host_path(self: Self, container_path: str) -> str:
        """Translate a container-side path to the corresponding host path.

        Inverts ``_extra_volumes`` (host → container) to find the longest
        matching container mount-point prefix.  Returns the input unchanged
        when no registered volume covers the path.

        Args:
            container_path: Absolute path as seen inside the container.

        Returns:
            The host-side equivalent of *container_path*, or *container_path*
            itself if no registered volume covers it.
        """
        best_match = ""
        result = container_path
        for host, vol in self._extra_volumes.items():
            mount = vol["bind"]
            if container_path == mount or container_path.startswith(mount + "/"):
                if len(mount) > len(best_match):
                    best_match = mount
                    suffix = container_path[len(mount) :]
                    result = host + suffix
        return result

    def _pull_image(
        self: Self,
        client,
        docker_module,
        image: str,
        pull_policy: str,
        registry_auth: Optional[Dict[str, str]] = None,
    ) -> None:
        """Pull image according to pull policy."""
        auth_config = None
        if registry_auth:
            auth_config = {
                "username": registry_auth.get("username", ""),
                "password": registry_auth.get("password", ""),
            }

        if pull_policy == "always":
            logger.info("Pulling image (policy=always): %s", image)
            client.images.pull(image, auth_config=auth_config)
        elif pull_policy == "if-not-present":
            try:
                client.images.get(image)
                logger.info("Image already present: %s", image)
            except docker_module.errors.ImageNotFound:
                logger.info("Pulling image (policy=if-not-present): %s", image)
                # TODO: Need to also handle exception here.
                client.images.pull(image, auth_config=auth_config)
        elif pull_policy == "never":
            client.images.get(image)  # raises ImageNotFound if missing
        else:
            logger.warning(
                "Unknown pull_policy '%s', treating as if-not-present", pull_policy
            )
            try:
                client.images.get(image)
            except docker_module.errors.ImageNotFound:
                client.images.pull(image, auth_config=auth_config)

    # ------------------------------------------------------------------
    # pullasset_hfstore
    # ------------------------------------------------------------------

    async def pullasset_hfstore(
        self: Self,
        uri,
        binding: Optional[Any] = None,
        storeload_config=None,
        assetstore=None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict, None]:
        """Download an HF model snapshot and bind as a container volume mount.

        HuggingFace ``snapshot_download`` stores files in a ``blobs/`` directory
        and creates symlinks from ``snapshots/<rev>/`` into ``blobs/``.  When the
        local path is inside an HF cache (has a sibling ``blobs/`` directory two
        levels up), we mount the entire ``models--<owner>--<repo>`` directory so
        the symlinks resolve inside the container.  Otherwise we mount the
        snapshot directory directly (non-HF-cache paths, tests, etc.).
        """
        assert assetstore is not None, "assetstore is required for hfstore loading"
        local_path = pull_asset_hfstore(uri, assetstore, storeload_config)
        relpath = assetstore.get_relpath(uri)
        container_path = f"/gb-hf-models/{relpath}"

        local_path_obj = Path(local_path)
        # Detect HF cache layout: local_path is .../models--org--repo/snapshots/<rev>/
        # and the sibling blobs/ directory exists two levels up.
        model_root = local_path_obj.parent.parent
        if (model_root / "blobs").is_dir():
            # HF cache with symlinks — mount the whole model root so symlinks
            # (snapshots/<rev>/file -> ../../blobs/<hash>) resolve correctly.
            snapshot_subpath = local_path_obj.relative_to(model_root)
            container_model_root = f"/gb-hf-models/{relpath}-cache"
            container_snapshot = f"{container_model_root}/{snapshot_subpath}"
            self._extra_volumes[str(model_root)] = {
                "bind": container_model_root,
                "mode": "ro",
            }
            container_path = container_snapshot
        else:
            # Non-HF-cache path — mount the directory directly.
            self._extra_volumes[str(local_path)] = {
                "bind": container_path,
                "mode": "ro",
            }

        from gbserver.environment.environment import BINDING_KEY

        binding_config = {BINDING_KEY: {"path": container_path}}
        return binding_config, None

    @staticmethod
    def _get_hf_cache_dir(storeload_config) -> str:
        """Resolve HF model cache directory from config or default."""
        return get_hf_cache_dir(storeload_config)

    # ------------------------------------------------------------------
    # pushasset_hfstore
    # ------------------------------------------------------------------

    async def pushasset_hfstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        uri: Optional[Any] = None,
        assetstore=None,
        run_metadata=None,
        **_kwargs,
    ) -> Any:
        """Upload a local file or directory to a HuggingFace repo.

        Translates any container-side path in ``binding`` to the corresponding
        host path using the volume map built by ``pullasset_hfstore`` and
        ``launch_docker``, then delegates to
        :func:`~gbserver.environment.local_assets.push_asset_hfstore`.

        Args:
            binding: Dict with a ``"path"`` key.  The path may be a
                container-side path (e.g. ``/gb-workspace/outputs/model``)
                which is resolved to the host path before pushing.
            binding_id: Output binding name, included in the commit message.
            uri: Target HfURI string or object.
            assetstore: Hfstore instance whose secrets supply the HF token.
            run_metadata: EntityRunMetadata with build_id and target_name.

        Returns:
            The resolved HfURI after a successful push.

        Raises:
            ValueError: If ``uri`` is absent or ``binding`` has no ``"path"``.
            RuntimeError: If the push itself fails.
        """
        if not isinstance(binding, dict) or "path" not in binding:
            raise ValueError(f"binding must be a dict with 'path', got: {binding}")
        host_path = self._resolve_host_path(binding["path"])
        return push_asset_hfstore(
            src=host_path,
            binding_id=binding_id,
            uri=uri,
            assetstore=assetstore,
            run_metadata=run_metadata,
        )

    # ------------------------------------------------------------------
    # launch_docker
    # ------------------------------------------------------------------

    async def launch_docker(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir=None,
        environment_config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ) -> None:
        """Launch a step in a Docker container."""
        try:
            docker, client = self._get_docker()

            # Extract kwargs
            launcher_config = kwargs.get("launcher_config", {}) or {}
            config = kwargs.get("config", {}) or {}
            compute_config = config.get("compute_config", {}) or {}
            docker_config = config.get("docker", {}) or {}
            step = kwargs.get("step", {}) or {}
            run_metadata = kwargs.get("run_metadata", {}) or {}

            # Resolve image
            image = self._resolve_image(launcher_config, docker_config)

            # Pull image according to policy
            pull_policy = docker_config.get("pull_policy", "if-not-present")
            registry_auth = docker_config.get("registry_auth")
            self._pull_image(client, docker, image, pull_policy, registry_auth)

            # Build container name
            step_name = step.get("name", run_metadata.get("target_name", "gb"))
            container_name = f"gb-{step_name}-{launch_id[:8]}"

            # Build environment variables
            env_vars: Dict[str, str] = {}
            # 1. Defaults from environment.yaml
            defaults = self._get_defaults()
            defaults_env = defaults.get("env", {})
            if isinstance(defaults_env, dict):
                for k, v in defaults_env.items():
                    if isinstance(v, dict) and "value" in v:
                        env_vars[k] = v["value"]
                    else:
                        env_vars[k] = str(v)
            # 2. docker_config env (from step config)
            docker_env = docker_config.get("env", {})
            if isinstance(docker_env, dict):
                for k, v in docker_env.items():
                    if isinstance(v, dict) and "value" in v:
                        env_vars[k] = v["value"]
                    else:
                        env_vars[k] = str(v)
            # 3. launcher_config env
            launcher_env = launcher_config.get("env", {})
            if isinstance(launcher_env, dict):
                env_vars.update(launcher_env)
            # 4. Built-in env vars
            env_vars["LLMB_DOCKER_LAUNCH_ID"] = launch_id
            env_vars["LLMB_DOCKER_CONTAINER_NAME"] = container_name

            # Build bind mount volumes
            volumes = {}
            if targetsteprun_asset_dir:
                self._extra_volumes[str(targetsteprun_asset_dir)] = {
                    "bind": "/gb-workspace",
                    "mode": "rw",
                }
            volumes = dict(self._extra_volumes)

            # Build resource kwargs
            resource_kwargs: Dict = {}
            total_memory = compute_config.get("total_memory_per_node", "")
            if total_memory:
                mem_limit = self._parse_memory(total_memory)
                if mem_limit:
                    resource_kwargs["mem_limit"] = mem_limit

            num_cpus = compute_config.get("num_cpus_per_node", 0)
            if num_cpus:
                resource_kwargs["nano_cpus"] = int(num_cpus * 1e9)

            num_gpus = compute_config.get("num_gpus_per_node", 0)
            if num_gpus:
                resource_kwargs["device_requests"] = [
                    docker.types.DeviceRequest(count=num_gpus, capabilities=[["gpu"]])
                ]

            # Container command
            command = launcher_config.get("command")

            logger.info(
                "Creating Docker container: name=%s image=%s launch_id=%s",
                container_name,
                image,
                launch_id,
            )

            container = client.containers.run(
                image=image,
                command=command,
                name=container_name,
                detach=True,
                volumes=volumes,
                environment=env_vars,
                **resource_kwargs,
            )

            self._launched_containers[launch_id] = container.id
            if targetsteprun_asset_dir:
                self._launched_workspaces[launch_id] = str(targetsteprun_asset_dir)
            logger.info(
                "Created Docker container: id=%s launch_id=%s",
                container.id,
                launch_id,
            )

        except Exception as e:
            logger.error("Failed to launch Docker container for %s: %s", launch_id, e)
            raise
        finally:
            self._release_monitors(launch_id)

    # ------------------------------------------------------------------
    # monitor_docker_log
    # ------------------------------------------------------------------

    async def monitor_docker_log(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata=None,
        build_id: str = "",
        **kwargs,
    ) -> None:
        """Monitor a Docker container's logs and status.

        Uses a dedicated streaming thread with a queue.Queue bridge to avoid
        the 'generator already executing' race condition that occurs when
        asyncio.wait_for timeouts overlap with run_in_executor(next(gen)).
        """
        container_id = self._launched_containers.get(launch_id)
        if not container_id:
            logger.warning("No container_id for launch_id %s, returning", launch_id)
            self._get_launch_stopped_event(launch_id).set()
            return

        exit_code = -1
        oom_killed = False

        try:
            _docker, client = self._get_docker()
            container = client.containers.get(container_id)

            stop_event = self._get_launch_stopped_event(launch_id)
            loop = asyncio.get_running_loop()

            # Parse event_configs from kwargs if provided
            event_configs: Optional[List[EventLogLineParserConfig]] = None
            raw_event_configs = kwargs.get("event_configs")
            if raw_event_configs:
                event_configs = [
                    EventLogLineParserConfig.model_validate(cfg)
                    for cfg in raw_event_configs
                ]

            # Thread-safe queue bridge: a dedicated thread streams Docker logs
            # into a queue.Queue so no generator is shared across threads.
            log_q: queue.Queue = queue.Queue()

            def _stream_logs_to_queue():
                try:
                    for chunk in container.logs(stream=True, follow=True):
                        log_q.put(chunk)
                except Exception as e:
                    logger.error(
                        "Error streaming Docker logs for %s: %s", container_id, e
                    )
                finally:
                    log_q.put(None)  # EOF sentinel

            log_thread = threading.Thread(
                target=_stream_logs_to_queue,
                daemon=True,
                name=f"docker-log-{launch_id[:8]}",
            )
            log_thread.start()

            while not stop_event.is_set():
                try:
                    line_bytes = await loop.run_in_executor(
                        None, functools.partial(log_q.get, timeout=2)
                    )
                except queue.Empty:
                    continue

                if line_bytes is None:
                    break

                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue

                if event_configs and event_q and entityrun_metadata:
                    await self.get_events_from_log_line(
                        log_line=line,
                        event_configs=event_configs,
                        event_q=event_q,
                        entityrun_metadata=entityrun_metadata,
                    )
                elif event_q and entityrun_metadata:
                    event = BuildEvent(
                        run_metadata=entityrun_metadata,
                        type=BuildEventType.MESSAGE_EVENT,
                        payload=BuildEventMessagePayload(msg=line),
                    )
                    await event_q.put(event)

            # Wait for streaming thread to finish
            log_thread.join(timeout=5)

            # Get exit code
            try:
                result = await loop.run_in_executor(
                    None, functools.partial(container.wait, timeout=30)
                )
                exit_code = result.get("StatusCode", -1)
            except Exception as e:
                logger.warning(
                    "Failed to get exit code for container %s: %s", container_id, e
                )

            # Check OOM
            try:
                await loop.run_in_executor(None, container.reload)
                oom_killed = container.attrs.get("State", {}).get("OOMKilled", False)
            except Exception as e:
                logger.warning(
                    "Failed to check OOM status for container %s: %s",
                    container_id,
                    e,
                )

            # Emit final status message
            status_msg = (
                f"Docker container {container_id[:12]} exited with code {exit_code}"
            )
            if oom_killed:
                status_msg += " (OOMKilled)"
            if event_q and entityrun_metadata:
                event = BuildEvent(
                    run_metadata=entityrun_metadata,
                    type=BuildEventType.MESSAGE_EVENT,
                    payload=BuildEventMessagePayload(msg=status_msg),
                )
                await event_q.put(event)

            logger.info(
                "Docker container %s finished: exit_code=%d oom=%s",
                container_id,
                exit_code,
                oom_killed,
            )

        except LogMonitoringFailedException:
            raise
        except Exception as e:
            logger.error("Error monitoring Docker container %s: %s", container_id, e)
            raise
        finally:
            # Always signal completion so other monitors and cleanup can proceed
            self._get_launch_stopped_event(launch_id).set()

        # Raise after finally so stop_event is set even on failure
        if exit_code != 0:
            raise LogMonitoringFailedException(
                f"Docker container {container_id[:12]} exited with code {exit_code}"
                + (" (OOMKilled)" if oom_killed else ""),
                build_id=build_id,
            )

    # ------------------------------------------------------------------
    # cleanup_docker
    # ------------------------------------------------------------------

    async def cleanup_docker(
        self: Self,
        launch_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Stop and remove a Docker container."""
        if launch_id is None:
            logger.warning("cleanup_docker called with no launch_id")
            return

        self._monitoring_cleanup(launch_id=launch_id)

        container_id = self._launched_containers.get(launch_id)
        if not container_id:
            logger.warning("No container to cleanup for launch_id %s", launch_id)
            return

        try:
            _docker, client = self._get_docker()
            container = client.containers.get(container_id)
            logger.info(
                "Stopping Docker container %s (launch_id=%s)", container_id, launch_id
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, functools.partial(container.stop, timeout=30)
            )
            await loop.run_in_executor(
                None, functools.partial(container.remove, force=True)
            )
            logger.info("Removed Docker container %s", container_id)
        except Exception as e:
            logger.error("Failed to cleanup Docker container %s: %s", container_id, e)
        finally:
            self._launched_containers.pop(launch_id, None)
            self._launched_workspaces.pop(launch_id, None)

    # ------------------------------------------------------------------
    # pushasset_filestore
    # ------------------------------------------------------------------

    async def pushasset_filestore(
        self: Self,
        binding: Any,
        uri: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Push an artifact from a Docker container workspace to a file URI.

        The artifact was written inside the container at /gb-workspace/...
        which is bind-mounted to the host workspace directory. This method
        translates the container path to the host path and copies to the
        output URI location.
        """
        if uri is None:
            return None

        # Extract source path from binding (dict or string)
        if isinstance(binding, dict):
            source_path = binding.get("path", "")
        else:
            source_path = str(binding)

        # Translate container path (/gb-workspace/...) to host path
        if source_path.startswith("/gb-workspace"):
            workspace_dir = None
            for _lid, wdir in self._launched_workspaces.items():
                workspace_dir = wdir
            if workspace_dir is None:
                logger.warning(
                    "No workspace mapping found for Docker pushasset, path=%s",
                    source_path,
                )
                return uri
            host_path = source_path.replace("/gb-workspace", workspace_dir, 1)
        else:
            host_path = source_path

        uriobj = uri
        if isinstance(uri, str):
            uriobj = URI.get_uri(uri)
        assert uriobj.uri is not None, "the URI is None"
        sync_or_copy(host_path, uriobj.uri.path)
        return uri
