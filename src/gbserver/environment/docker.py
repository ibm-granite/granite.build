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
import os
import queue
import threading
from asyncio import Queue, Task, TaskGroup
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple

import requests
import urllib3
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from gbcommon.types.constants import get_gb_home_dir
from gbcommon.uri.uri import URI
from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.environment.local_assets import (
    get_hf_cache_dir,
    pull_asset_hfstore,
    push_asset_hfstore,
)
from gbserver.types.buildconfig import BuildTargetOutputConfig
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.constants import (
    DOCKER_PULL_MAX_ATTEMPTS,
    DOCKER_PULL_RETRY_BASE_DELAY,
    DOCKER_PULL_RETRY_MAX_DELAY,
)
from gbserver.types.environment.docker import StepDockerConfig
from gbserver.types.environment.environment import EnvironmentVariableConfig
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.types.errors import LogMonitoringFailedException
from gbserver.utils.filesystem import sync_or_copy
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Container path the targetsteprun workspace is bind-mounted at.
_WORKSPACE_MOUNT = "/gb-workspace"
# Container path prefix env:// inputs are bind-mounted under.
_ENV_INPUTS_MOUNT = "/gb-env-inputs"
# Container path prefix HuggingFace inputs are bind-mounted under.
_HF_MODELS_MOUNT = "/gb-hf-models"
# Container paths no host has: a binding under one only makes sense inside a
# container, so it is never cached for a consumer or registered as env://.
_CONTAINER_MOUNTS = (_WORKSPACE_MOUNT, _ENV_INPUTS_MOUNT, _HF_MODELS_MOUNT)
# Host dirs an env:// input mount may neither be, contain, nor sit inside. The
# daemon mounts as root, so this includes root-only dirs the gbserver user can
# stat but not read, e.g. other containers' configs (and their env vars).
_DENIED_MOUNT_DIRS = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/run",
    "/var/run",
    "/root",
    "/var/lib/docker",
    "/var/lib/containers",
    "/var/log",
)


def _is_under(path: str, mount: str) -> bool:
    """Whether container *path* is *mount* or lies below it."""
    return path == mount or path.startswith(mount + "/")


def _is_container_path(path: Any) -> bool:
    """Whether *path* lies under one of the container-only mounts."""
    return isinstance(path, str) and any(_is_under(path, m) for m in _CONTAINER_MOUNTS)


# Pull errors no retry fixes: bad reference, missing image or platform, bad
# credentials, full disk.
_PERMANENT_PULL_ERRORS = (
    "manifest unknown",
    "repository does not exist",
    "invalid reference format",
    "no matching manifest",
    "unauthorized",
    "denied",
    "no space left on device",
)

# Connection failures, including a stream dropped mid-pull (raised by urllib3).
_TRANSIENT_PULL_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.TimeoutError,
)


class ImagePullError(RuntimeError):
    """An image pull reported an error or did not produce the image."""


def _is_retryable_pull_error(exc: BaseException, docker_module) -> bool:
    """Return True if *exc* from an image pull may succeed on retry."""
    if isinstance(exc, _TRANSIENT_PULL_EXCEPTIONS):
        return True
    if isinstance(exc, docker_module.errors.APIError):
        status = exc.status_code
        if status is not None and status < 500 and status != 429:
            return False
    elif not isinstance(exc, ImagePullError):
        return False
    message = str(exc).lower()
    return not any(p in message for p in _PERMANENT_PULL_ERRORS)


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
        # targetsteprun_id -> host workspace dir. Never popped by cleanup_docker,
        # so an output push that races the container teardown still finds its
        # producer's workspace. An instance serves one build (and its in-place
        # retries), which bounds it.
        self._workspaces_by_targetsteprun: Dict[str, str] = {}
        self._extra_volumes: Dict[str, Dict] = {}
        # host dir -> volume for env:// inputs. Unlike _extra_volumes these are
        # not mounted into every container: a launch gets only the ones its own
        # bindings reference (see _env_input_volumes_for).
        self._env_input_volumes: Dict[str, Dict] = {}
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

        Inverts ``_extra_volumes`` and ``_env_input_volumes`` (host →
        container) to find the longest matching container mount-point prefix.
        Returns the input unchanged when no registered volume covers the path.

        Args:
            container_path: Absolute path as seen inside the container.

        Returns:
            The host-side equivalent of *container_path*, or *container_path*
            itself if no registered volume covers it.
        """
        best_match = ""
        result = container_path
        for host, vol in {**self._extra_volumes, **self._env_input_volumes}.items():
            mount = vol["bind"]
            if _is_under(container_path, mount):
                if len(mount) > len(best_match):
                    best_match = mount
                    suffix = container_path[len(mount) :]
                    result = host + suffix
        return result

    def _host_path(self: Self, container_path: str, run_metadata: Optional[Any]) -> str:
        """Translate a container path from a pushed output to its host path.

        A path under ``/gb-workspace`` maps into the host workspace of the
        launch that produced the output, looked up by
        ``run_metadata.targetsteprun_id``: each launch mounts its own workspace
        there, so no shared volume map can answer. Any other path is resolved
        through the shared read-only mounts by :meth:`_resolve_host_path`.

        :raises LookupError: the path is under ``/gb-workspace`` but no
            workspace is recorded for the producing targetsteprun (guessing
            another launch's workspace would push the wrong files).
        """
        if not _is_under(container_path, _WORKSPACE_MOUNT):
            return self._resolve_host_path(container_path)
        if isinstance(run_metadata, dict):
            tsr_id = run_metadata.get("targetsteprun_id")
        else:
            tsr_id = getattr(run_metadata, "targetsteprun_id", None)
        workspace = self._workspaces_by_targetsteprun.get(tsr_id or "")
        if workspace is None:
            raise LookupError(
                f"no Docker workspace recorded for targetsteprun_id={tsr_id!r}; "
                f"cannot resolve container path {container_path}"
            )
        return workspace + container_path[len(_WORKSPACE_MOUNT) :]

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
            self._pull_with_retry(client, docker_module, image, auth_config)
        elif pull_policy == "if-not-present":
            try:
                client.images.get(image)
                logger.info("Image already present: %s", image)
            except docker_module.errors.ImageNotFound:
                logger.info("Pulling image (policy=if-not-present): %s", image)
                self._pull_with_retry(client, docker_module, image, auth_config)
        elif pull_policy == "never":
            client.images.get(image)  # raises ImageNotFound if missing
        else:
            logger.warning(
                "Unknown pull_policy '%s', treating as if-not-present", pull_policy
            )
            try:
                client.images.get(image)
            except docker_module.errors.ImageNotFound:
                self._pull_with_retry(client, docker_module, image, auth_config)

    def _pull_with_retry(
        self: Self,
        client,
        docker_module,
        image: str,
        auth_config: Optional[Dict[str, str]],
    ) -> None:
        """Pull *image*, retrying transient registry errors."""

        def _log_retry(retry_state: RetryCallState) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            sleep = getattr(retry_state.next_action, "sleep", 0.0)
            logger.warning(
                "Retrying pull of %s in %.1fs (attempt %d/%d) after: %s",
                image,
                sleep,
                retry_state.attempt_number,
                DOCKER_PULL_MAX_ATTEMPTS,
                exc,
            )

        for attempt in Retrying(
            stop=stop_after_attempt(DOCKER_PULL_MAX_ATTEMPTS),
            wait=wait_random_exponential(
                multiplier=DOCKER_PULL_RETRY_BASE_DELAY, max=DOCKER_PULL_RETRY_MAX_DELAY
            ),
            retry=retry_if_exception(
                lambda e: _is_retryable_pull_error(e, docker_module)
            ),
            before_sleep=_log_retry,
            reraise=True,
        ):
            with attempt:
                self._pull_once(client, docker_module, image, auth_config)

    def _pull_once(
        self: Self,
        client,
        docker_module,
        image: str,
        auth_config: Optional[Dict[str, str]],
    ) -> None:
        """Pull *image* once; raise ImagePullError if the pull did not land.

        Unlike ``client.images.pull``, which drops the stream and reports any
        registry error as ImageNotFound, this raises the daemon's own message.
        """
        for chunk in client.api.pull(
            image, stream=True, decode=True, auth_config=auth_config
        ):
            error = chunk.get("error") or (chunk.get("errorDetail") or {}).get(
                "message"
            )
            if error:
                raise ImagePullError(f"Failed to pull image {image}: {error}")
        # A stream can end early without an error chunk.
        try:
            client.images.get(image)
        except docker_module.errors.ImageNotFound as e:
            raise ImagePullError(
                f"Pull of {image} ended but the image is missing"
            ) from e

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
        self._warn_non_default_mode(storeload_config, uri)
        assert assetstore is not None, "assetstore is required for hfstore loading"
        local_path = pull_asset_hfstore(uri, assetstore, storeload_config)
        relpath = assetstore.get_relpath(uri)
        container_path = f"{_HF_MODELS_MOUNT}/{relpath}"

        local_path_obj = Path(local_path)
        # Detect HF cache layout: local_path is .../models--org--repo/snapshots/<rev>/
        # and the sibling blobs/ directory exists two levels up.
        model_root = local_path_obj.parent.parent
        if (model_root / "blobs").is_dir():
            # HF cache with symlinks — mount the whole model root so symlinks
            # (snapshots/<rev>/file -> ../../blobs/<hash>) resolve correctly.
            snapshot_subpath = local_path_obj.relative_to(model_root)
            container_model_root = f"{_HF_MODELS_MOUNT}/{relpath}-cache"
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

    # ------------------------------------------------------------------
    # pullasset_envstore
    # ------------------------------------------------------------------

    async def pullasset_envstore(
        self: Self,
        uri: Optional[Any] = None,
        binding: Optional[Any] = None,
        storeload_config=None,
        **kwargs,
    ) -> Tuple[Dict, None]:
        """Bind-mount an ``env://`` host path into the container, read-only.

        The base pull is a no-op that returns the host path, which a container
        cannot see. Register a mount of the path (a file's parent directory, or
        the directory itself) under ``/gb-env-inputs/<n>`` and return the
        container-side path as the binding. The mount source is always the
        input's host path, including one an earlier Docker target wrote in its
        own workspace (its env:// output is registered at that host path).

        The mount is not added to every container: :meth:`launch_docker` mounts
        it only into launches whose bindings reference it. A directory already
        mounted for another input, or as a shared read-only HF mount of the same
        resolved directory, is reused, not mounted twice.

        A file's whole parent directory is mounted so that steps can read its
        siblings. To keep that scope bounded the mount source is checked by
        :meth:`_check_env_input_mount_src`.

        :raises FileNotFoundError: the host path does not exist.
        :raises ValueError: the mount source is refused.
        """
        from gbserver.environment.environment import BINDING_KEY

        binding_config, _ = await super().pullasset_envstore(
            uri=uri, binding=binding, storeload_config=storeload_config, **kwargs
        )
        host = Path(binding_config[BINDING_KEY]["path"])
        if not host.exists():
            raise FileNotFoundError(f"env:// input {uri} not found at {host}")
        is_dir = host.is_dir()
        # Resolve the directory once and use that one value for the check, the
        # mount key and the mount source, so the container gets exactly the
        # directory that was checked, and two spellings of one directory share
        # a mount. The file name is kept as given: a file symlink resolves
        # inside the container's own filesystem, so it cannot leave the mount.
        try:
            mount_src = (host if is_dir else host.parent).resolve(strict=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"env:// input {uri} not found at {host}") from e
        self._check_env_input_mount_src(mount_src, uri)
        key = str(mount_src)
        volume = self._shared_volume_for(mount_src) or self._env_input_volumes.get(key)
        if volume is None:
            n = len(self._env_input_volumes)
            volume = {"bind": f"{_ENV_INPUTS_MOUNT}/{n}", "mode": "ro"}
            self._env_input_volumes[key] = volume
        container = volume["bind"] if is_dir else f"{volume['bind']}/{host.name}"
        logger.info(
            "pullasset_envstore: mounted %s at %s (container path %s)",
            mount_src,
            volume["bind"],
            container,
        )
        return {BINDING_KEY: {"path": container}}, None

    def _shared_volume_for(self: Self, mount_src: Path) -> Optional[Dict]:
        """The shared read-only mount (e.g. HF) of resolved *mount_src*, if any.

        ``_extra_volumes`` keys are the host paths as the HF pull produced them,
        so compare resolved paths, not strings.
        """
        for host, vol in self._extra_volumes.items():
            if Path(host).resolve() == mount_src:
                return vol
        return None

    def _env_output_roots(self: Self) -> List[Path]:
        """Resolved host dirs whose subdirectories gbserver wrote outputs to.

        - ``<GB home>/workdir``, the Bash environment's default output dir;
          only a directory strictly below it is allowed, so one input never
          exposes every Bash build's outputs.
        - The workspace of each Docker launch of this build (recorded by
          :meth:`launch_docker`), so an env:// output of an earlier Docker
          target can feed a later one. Not the whole workspace root: it also
          holds the build watcher's PR clones, whose git config carries a
          GitHub token.

        A root that contains the home or GB home directory is dropped, since
        allowing it would reopen ``~/.ssh`` or the secrets kept in GB home.
        """
        home = Path.home().resolve()
        gb_home = Path(get_gb_home_dir()).resolve()
        roots = [(gb_home / "workdir").resolve()]
        roots += [Path(w).resolve() for w in self._workspaces_by_targetsteprun.values()]
        return [
            r
            for r in roots
            if not (home.is_relative_to(r) or gb_home.is_relative_to(r))
        ]

    def _check_env_input_mount_src(self: Self, mount_src: Path, uri: Any) -> None:
        """Refuse an env:// input mount source that would expose host secrets.

        *mount_src* must already be resolved: it is checked as given, and the
        caller mounts that same path. The container may hold space secrets, and
        the daemon mounts as root, so the source must:

        - be readable by the gbserver user (the container never gets a
          directory the server itself cannot read, e.g. root-only dirs);
        - not contain the gbserver user's home directory (``/``, ``/home``,
          ``~``);
        - not be, contain, or sit inside a system directory in
          ``_DENIED_MOUNT_DIRS`` (e.g. ``/etc`` for ``env:///etc/hosts``,
          ``/var`` because it contains ``/var/lib/docker``);
        - not be, contain, or sit inside the GB home directory, nor sit inside a
          hidden directory of the home directory (``~/.ssh``, ``~/.aws``,
          ``~/.gbcli``, ...), unless it lies in an output root from
          :meth:`_env_output_roots`. That lets a Docker target read an env://
          output a Bash target wrote under ``~/.granite.build/workdir/<build>``
          while the rest of the GB home (secrets, the SQLite DB) stays refused.

        :raises ValueError: the mount source is refused; the message names the
            rule.
        """
        src = mount_src
        home = Path.home().resolve()
        gb_home = Path(get_gb_home_dir()).resolve()
        system = [Path(d).resolve() for d in _DENIED_MOUNT_DIRS]
        in_output_root = any(
            src.is_relative_to(r) and src != r for r in self._env_output_roots()
        )
        reason = None
        if not os.access(src, os.R_OK | os.X_OK):
            reason = "it is not readable by the gbserver user"
        elif home.is_relative_to(src):
            reason = "it contains the home directory"
        elif any(src.is_relative_to(d) or d.is_relative_to(src) for d in system):
            reason = "it is, contains or lies in a system directory"
        elif not in_output_root and (
            src.is_relative_to(gb_home) or gb_home.is_relative_to(src)
        ):
            reason = "it is, contains or lies in the GB home directory"
        elif (
            not in_output_root
            and src.is_relative_to(home)
            and src != home
            and src.relative_to(home).parts[0].startswith(".")
        ):
            reason = "it lies in a hidden home directory"
        if reason:
            raise ValueError(
                f"env:// input {uri} would mount {src} into the container, but "
                f"{reason}; put the input in its own directory"
            )

    def _env_input_volumes_for(self: Self, bindings: Optional[Dict]) -> Dict[str, Dict]:
        """The env:// input mounts a launch's own *bindings* reference.

        :param bindings: the launch's ``bindings`` (binding id -> pulled
            binding config, as returned by :meth:`pullasset_envstore`).
        :returns: the host dir -> volume entries to add to the container.
        """
        from gbserver.environment.environment import BINDING_KEY

        paths = [
            b[BINDING_KEY].get("path")
            for b in (bindings or {}).values()
            if isinstance(b, dict) and isinstance(b.get(BINDING_KEY), dict)
        ]
        return {
            host: vol
            for host, vol in self._env_input_volumes.items()
            if any(isinstance(p, str) and _is_under(p, vol["bind"]) for p in paths)
        }

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
        storepush_config=None,
        output_config=None,
        **_kwargs,
    ) -> Any:
        """Upload a local file or directory to a HuggingFace repo.

        Translates any container-side path in ``binding`` to the corresponding
        host path (see :meth:`_host_path`: a ``/gb-workspace`` path maps into the
        producing launch's workspace, others through the read-only input
        mounts), then delegates to
        :func:`~gbserver.environment.local_assets.push_asset_hfstore`.

        Args:
            binding: Dict with a ``"path"`` key.  The path may be a
                container-side path (e.g. ``/gb-workspace/outputs/model``)
                which is resolved to the host path before pushing.
            binding_id: Output binding name, included in the commit message.
            uri: Target HfURI string or object.
            assetstore: Hfstore instance whose secrets supply the HF token.
            run_metadata: EntityRunMetadata with build_id and target_name; its
                ``targetsteprun_id`` selects the workspace of a
                ``/gb-workspace`` path.
            storepush_config: Environment-level push config. Its ``hf`` block is
                forwarded so the resource group settings apply here as they do on
                the step-based environments.
            output_config: Per-output build.yaml config, forwarded so
                ``store_push.config.hf`` (including ``use_resource_group``) is
                honored.

        Returns:
            The resolved HfURI after a successful push.

        Raises:
            ValueError: If ``uri`` is absent or ``binding`` has no ``"path"``.
            LookupError: A ``/gb-workspace`` path whose producing launch is
                unknown.
            RuntimeError: If the push itself fails.
        """
        self._warn_non_default_mode(storepush_config, uri)
        if not isinstance(binding, dict) or "path" not in binding:
            raise ValueError(f"binding must be a dict with 'path', got: {binding}")
        host_path = self._host_path(binding["path"], run_metadata)
        return push_asset_hfstore(
            src=host_path,
            binding_id=binding_id,
            uri=uri,
            assetstore=assetstore,
            run_metadata=run_metadata,
            storepush_config=storepush_config,
            output_config=output_config,
        )

    # ------------------------------------------------------------------
    # launch_docker
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_env_section(env: Dict[str, str], section: Any) -> None:
        """Merge a config env section into ``env``, unwrapping value dicts.

        A section maps env name -> value, where a value may itself be a
        ``{"value": ...}`` dict (the environment.yaml/step-config form).

        :param env: the env dict to update in place.
        :param section: the env mapping; ignored if not a dict.
        """
        if not isinstance(section, dict):
            return
        for k, v in section.items():
            if isinstance(v, dict) and "value" in v:
                env[k] = v["value"]
            else:
                env[k] = str(v)

    def _launch_env_layers(self: Self, **kwargs: Any) -> List[Dict[str, str]]:
        """Docker env layers for the shared launch-env composer.

        Overrides :meth:`Environment._launch_env_layers` (declared space secrets
        come from :meth:`_declared_secret_mappings`). Ordered lowest->highest
        above the declared-secret layer and below the standard set: environment.yaml defaults ``env`` <
        ``config.docker.env`` < launcher ``env`` < the built-in
        ``LLMB_DOCKER_*`` vars.

        The two config env sections are pre-merged with :meth:`_merge_env_section`
        (which unwraps the ``{"value": ...}`` form and ignores a non-dict
        section) into a single lowest layer.

        The built-in launcher vars are standardized on the ``GB_`` prefix
        (``GB_DOCKER_*``): the composer's unconditional aliasing mirrors each
        ``LLMB_DOCKER_*`` onto a ``GB_DOCKER_*`` twin, keeping the legacy
        ``LLMB_DOCKER_*`` names for backwards compatibility.

        :param kwargs: the launch context; reads ``docker_config``
            (``config.docker`` env), ``launcher_config`` (its ``env``),
            ``launch_id`` (LLMB_DOCKER_LAUNCH_ID), and ``container_name``
            (LLMB_DOCKER_CONTAINER_NAME).
        :returns: the ordered env layers to compose.
        """
        config_env: Dict[str, str] = {}
        self._merge_env_section(config_env, (self._get_defaults() or {}).get("env"))
        self._merge_env_section(
            config_env, (kwargs.get("docker_config") or {}).get("env")
        )
        launcher_env = (kwargs.get("launcher_config") or {}).get("env")
        builtins: Dict[str, str] = {
            "LLMB_DOCKER_LAUNCH_ID": kwargs.get("launch_id", ""),
            "LLMB_DOCKER_CONTAINER_NAME": kwargs.get("container_name", ""),
        }
        return [
            config_env,
            launcher_env if isinstance(launcher_env, dict) else {},
            builtins,
        ]

    def _declared_secret_mappings(
        self: Self, **kwargs: Any
    ) -> List[EnvironmentVariableConfig]:
        """Declared Docker secret mappings for the shared launch-env composer.

        Overrides :meth:`Environment._declared_secret_mappings` so only secrets
        the step declares in ``config.docker.secrets.secret_names_to_use_as_env_variable``
        are injected into the container (least-privilege, matching
        LSF/SkyPilot). ``secret_name`` defaults to ``env_name`` verbatim.

        :param kwargs: the launch context; only ``config`` is read.
        :returns: the declared ``EnvironmentVariableConfig`` mappings.
        """
        docker_config = (kwargs.get("config") or {}).get("docker") or {}
        # Validate only the secrets section: config.docker.env also accepts plain
        # values, which StepDockerConfig.env does not model.
        return StepDockerConfig.model_validate(
            {"secrets": docker_config.get("secrets") or {}}
        ).secrets.secret_names_to_use_as_env_variable

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
            # Blocking docker-SDK calls (image pull, container run) must be
            # offloaded to a thread; running them directly in this coroutine
            # freezes the event loop, so an enclosing ``asyncio.wait_for`` can
            # never enforce its timeout and the build hangs indefinitely on a
            # slow/stalled pull or daemon.  (monitor_docker_log/cleanup_docker
            # already offload their blocking calls the same way.)
            loop = asyncio.get_running_loop()

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
            await loop.run_in_executor(
                None,
                functools.partial(
                    self._pull_image,
                    client,
                    docker,
                    image,
                    pull_policy,
                    registry_auth,
                ),
            )

            # Build container name
            step_name = step.get("name", run_metadata.get("target_name", "gb"))
            container_name = f"gb-{step_name}-{launch_id[:8]}"

            # Build environment variables (full env incl. authoritative
            # standard vars like GB_BUILD_ID; see Docker.get_launch_env_vars).
            env_vars = self.get_launch_env_vars(
                run_metadata=run_metadata,
                config=config,
                launcher_config=launcher_config,
                docker_config=docker_config,
                launch_id=launch_id,
                container_name=container_name,
            )

            # Build bind mount volumes: the shared read-only mounts, the env://
            # inputs this launch's bindings reference, and this launch's own
            # workspace. Neither of the last two is added to _extra_volumes,
            # which every later launch on this instance reads: two host dirs
            # bound to /gb-workspace make dockerd refuse the container
            # ("Duplicate mount point"), and a deleted input dir would fail
            # every later container ("bind source path does not exist").
            volumes = dict(self._extra_volumes)
            volumes.update(self._env_input_volumes_for(kwargs.get("bindings")))
            if targetsteprun_asset_dir:
                volumes[str(targetsteprun_asset_dir)] = {
                    "bind": _WORKSPACE_MOUNT,
                    "mode": "rw",
                }

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

            # Annotated Any: wrapping the call in run_in_executor hides the
            # detach=True -> Container overload, so the checker would otherwise
            # infer the bytes (logs) overload and flag container.id below.
            container: Any = await loop.run_in_executor(
                None,
                functools.partial(
                    client.containers.run,
                    image=image,
                    command=command,
                    name=container_name,
                    detach=True,
                    volumes=volumes,
                    environment=env_vars,
                    **resource_kwargs,
                ),
            )

            self._launched_containers[launch_id] = container.id
            tsr_id = run_metadata.get("targetsteprun_id")
            if targetsteprun_asset_dir and not tsr_id:
                # TargetStepRun always sets it; without it a /gb-workspace
                # output of this launch cannot be pushed.
                logger.warning(
                    "Docker launch %s has no targetsteprun_id; its /gb-workspace "
                    "outputs cannot be pushed",
                    launch_id,
                )
            if targetsteprun_asset_dir and tsr_id:
                # Absolute, so an env:// output registered under it resolves
                # from any working directory.
                self._workspaces_by_targetsteprun[tsr_id] = str(
                    Path(targetsteprun_asset_dir).absolute()
                )
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

    # ------------------------------------------------------------------
    # pushasset
    # ------------------------------------------------------------------

    def pushasset(
        self: Self,
        task_group: TaskGroup,
        binding: Any,
        uristr: str,
        binding_id: Optional[str] = None,
        additional_targetsteps_queue: Optional[Queue] = None,
        run_metadata: Optional[EntityRunMetadata] = None,
        output_config: Optional[BuildTargetOutputConfig] = None,
        unique_hash: Optional[str] = None,
    ) -> Task[URI]:
        """Push an asset, registering env:// container-path outputs at the host.

        An env:// output is a no-transfer reference, so the base renders its URI
        straight from the producer binding — which holds the in-container path
        (``/gb-workspace/...``, or an input mount) that no consumer can reach.
        For env:// outputs only, a container path is rewritten to its host path
        (see :meth:`_env_output_host_binding`) before the base renders and
        registers it. Every other store and path is passed through unchanged.

        If the host path cannot be found, the returned task raises
        :class:`LookupError`, as a ``file://`` or ``hf://`` push of such a path
        does.
        """
        if URI.get_uristr(uristr).startswith("env://"):
            try:
                binding = self._env_output_host_binding(binding, run_metadata)
            except LookupError as e:
                error = e

                async def fail() -> URI:
                    raise error

                asyncio_runner = task_group if task_group is not None else asyncio
                return asyncio_runner.create_task(fail())
        return super().pushasset(
            task_group=task_group,
            binding=binding,
            uristr=uristr,
            binding_id=binding_id,
            additional_targetsteps_queue=additional_targetsteps_queue,
            run_metadata=run_metadata,
            output_config=output_config,
            unique_hash=unique_hash,
        )

    def _env_output_host_binding(
        self: Self, binding: Any, run_metadata: Optional[EntityRunMetadata]
    ) -> Any:
        """Return a copy of *binding* with a container path mapped to the host.

        A ``/gb-workspace`` path maps into the producing launch's workspace; a
        path under a read-only input mount (``/gb-env-inputs``,
        ``/gb-hf-models``), e.g. an input re-emitted as an output, maps to that
        input's host path (see :meth:`_host_path`). Returns *binding* itself
        when its path is not a container path.

        :raises LookupError: no workspace is recorded for the producing
            targetsteprun, or no mount covers the path. Registering the
            container path would record a URI that exists on no host, and
            guessing another launch's workspace would register a wrong one.
        """
        path = binding.get("path") if isinstance(binding, dict) else None
        if not isinstance(path, str) or not _is_container_path(path):
            return binding
        host = self._host_path(path, run_metadata)
        if _is_container_path(host):
            raise LookupError(
                f"no Docker mount covers container path {path}; cannot register "
                "it as an env:// output"
            )
        return {**binding, "path": host}

    def _caches_pushed_binding(self: Self, uristr: str, binding: Any) -> bool:
        """Never cache a producer binding that holds a container path.

        Each container mounts only its own workspace, so another target's
        container would read ``/gb-workspace/...`` from its own workspace: the
        file would be missing, or the wrong one. Without the cache, a consumer's
        pull goes to the store (an ``hf://`` pull mounts the model; ``file://``
        has no Docker pull and fails with a clear error).
        """
        path = binding.get("path") if isinstance(binding, dict) else None
        if _is_container_path(path):
            return False
        return super()._caches_pushed_binding(uristr, binding)

    async def pushasset_envstore(self: Self, *args: Any, **kwargs: Any) -> URI:
        """Refuse to register an env:// output at a container path.

        :meth:`pushasset` maps a templated binding path to its host path, but a
        literal URI such as ``env:///gb-workspace/outputs/x`` still names a
        path that exists on no host.

        :raises ValueError: the URI's path is a container path.
        """
        uri = kwargs.get("uri")
        if uri:
            envuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
            if envuri.uri is not None and _is_container_path(envuri.uri.path):
                raise ValueError(
                    f"env:// output {URI.get_uristr(envuri)} names a container "
                    "path; use env://{{ binding.path }} so it is registered at "
                    "its host path"
                )
        return await super().pushasset_envstore(*args, **kwargs)

    # ------------------------------------------------------------------
    # pushasset_filestore
    # ------------------------------------------------------------------

    async def pushasset_filestore(
        self: Self,
        binding: Any,
        uri: Optional[Any] = None,
        storepush_config=None,
        run_metadata: Optional[EntityRunMetadata] = None,
        **kwargs,
    ) -> Any:
        """Push an artifact from a Docker container workspace to a file URI.

        The artifact was written inside the container at /gb-workspace/...,
        which is bind-mounted to the producing launch's host workspace. The
        container path is translated to that host path (see :meth:`_host_path`)
        and copied to the output URI location. A host path is copied as is.

        :raises LookupError: a ``/gb-workspace`` path whose producing launch
            (``run_metadata.targetsteprun_id``) is unknown.
        """
        self._warn_non_default_mode(storepush_config, uri)
        if uri is None:
            return None

        # Extract source path from binding (dict or string)
        if isinstance(binding, dict):
            source_path = binding.get("path", "")
        else:
            source_path = str(binding)

        host_path = self._host_path(source_path, run_metadata)

        uriobj = uri
        if isinstance(uri, str):
            uriobj = URI.get_uri(uri)
        assert uriobj.uri is not None, "the URI is None"
        sync_or_copy(host_path, uriobj.uri.path)
        return uri
