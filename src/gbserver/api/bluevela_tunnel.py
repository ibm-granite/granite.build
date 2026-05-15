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

"""Per-request SSH tunnel factory for the BlueVela remote-file REST API.

Opens a short-lived `SshTunnel` scoped to a single REST request. The REST
process does not share tunnels with the buildwatcher — each request pays a
handshake cost (~1s typical, more under load). This is acceptable for
interactive file inspection; if it becomes a bottleneck, a pooled tunnel
with health checks can be introduced here without changing callers.

Connection parameters are resolved from the target's `environment_uri` via
the same loader the buildwatcher uses (`Environment.load_environment_config`),
so the API always sees the same SSH config the build itself ran with. The
SSH key is pulled from the space's IBM Cloud Secret Manager group at
request time.
"""

import os
import random
import stat
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

from fastapi import HTTPException, status

from gbserver.environment.environment import Environment
from gbserver.types.constants import ENABLE_SSH_HOST_KEY_VERIFICATION
from gbserver.utils.logger import get_logger
from gbserver.utils.ssh_tunnel import SshTunnel

logger = get_logger(__name__)


@dataclass(frozen=True)
class BlueVelaConfig:
    """Connection + layout config resolved for one REST request."""

    login_node: str
    username: str
    workspace_remote_dir: str


def _pick_login_node(nodes: List[str]) -> str:
    """Pick one login node at random.

    v1 does no liveness check — asyncssh.connect will surface errors on
    handshake. Reusing Lsf._get_reachable_ssh_node here would pull in the
    Lsf instance's mutable state (self.unreachable_ssh_nodes, locks), so
    we keep it simple.
    """
    cleaned = [n.strip() for n in nodes if isinstance(n, str) and n.strip()]
    if not cleaned:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no login nodes configured in environment.yaml",
        )
    return random.choice(cleaned)


def _resolve_lsf_config(environment_uri: str) -> tuple[List[str], str, str, str]:
    """Load environment.yaml for `environment_uri` and return Lsf SSH params.

    Returns (login_nodes, username, ssh_key_secret_name, workspace_remote_dir).
    Raises 400 if the environment is not type "Lsf"; 503 if any required
    field is missing from environment.yaml.
    """
    try:
        env_config, _asset = Environment.load_environment_config(environment_uri)
    except Exception as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"failed to load environment config from {environment_uri!r}: {e}",
        ) from e
    if (env_config.type or "").lower() != "lsf":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"bluevela API only supports Lsf environments; this target uses {env_config.type!r}",
        )
    cfg: Dict = env_config.config or {}
    authentication: Dict = cfg.get("authentication", {})
    workspace: Dict = cfg.get("workspace", {})

    login_nodes = authentication.get("login_nodes", []) or []
    username = (authentication.get("login_node_username") or "").strip()
    ssh_key_secret_name = (authentication.get("login_node_ssh_key") or "").strip()
    workspace_remote_dir = (workspace.get("remote_dir") or "").strip()

    missing = [
        name
        for name, val in (
            ("authentication.login_nodes", login_nodes),
            ("authentication.login_node_username", username),
            ("authentication.login_node_ssh_key", ssh_key_secret_name),
            ("workspace.remote_dir", workspace_remote_dir),
        )
        if not val
    ]
    if missing:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"environment.yaml for {environment_uri!r} is missing required fields: {missing}",
        )
    return login_nodes, username, ssh_key_secret_name, workspace_remote_dir


def _fetch_ssh_key_for_space(space_name: str, secret_name: str) -> str:
    """Fetch the SSH private key for `space_name` from IBM Cloud Secret Manager.

    Uses IbmcloudSpaceSecretManager (non-admin), which merges secrets
    across every group whose description regex matches the space's
    git_repo_uri — including the public group. This mirrors the LSF
    runtime's view of `space_secrets`, so a key stored in gbspace-public
    (or any other matching group) is found here without invoking the
    heavyweight Space.__init__ path.

    503 if the SDK is unavailable or no groups resolve; 404 if the space
    or the secret name is not found.
    """
    try:
        from gbserver.spacesecretmanager.ibmcloudspacesecretmanager import (
            IbmcloudSpaceSecretManager,
        )
    except ImportError as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "IBM Cloud secret manager support is not installed",
        ) from e

    from gbserver.storage.singleton_storage import get_admin_storage

    stored_space = get_admin_storage().space_storage.get_by_name(space_name)
    if stored_space is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"space {space_name!r} not found"
        )

    manager = IbmcloudSpaceSecretManager(uri=stored_space.git_repo_uri)
    if not manager.secret_groups:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"no secret groups resolved for space {space_name!r}",
        )
    secrets = manager.get_space_secrets() or {}
    key_material = secrets.get(secret_name)
    if not key_material:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"ssh key secret {secret_name!r} not found in any secret group "
            f"matching space {space_name!r}",
        )
    return key_material


def _write_key_file(key_material: str) -> str:
    """Write an SSH private key to a 0600 tempfile. Caller must unlink."""
    fd, path = tempfile.mkstemp(prefix="bluevela_ssh_", suffix=".key")
    try:
        os.write(fd, key_material.encode("utf-8"))
        if not key_material.endswith("\n"):
            os.write(fd, b"\n")
    finally:
        os.close(fd)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


@asynccontextmanager
async def open_bluevela_tunnel(
    space_name: str,
    environment_uri: str,
) -> AsyncIterator[tuple[SshTunnel, BlueVelaConfig]]:
    """Open a short-lived SSH tunnel to a BlueVela login node for one request.

    Resolves SSH connection params from the target's environment.yaml (via
    Environment.load_environment_config) so the API stays in sync with the
    build's actual config. Yields (tunnel, config). On exit the tunnel is
    closed and the private-key tempfile is unlinked, including cancellation
    paths.
    """
    login_nodes, username, key_secret_name, workspace_remote_dir = _resolve_lsf_config(
        environment_uri
    )

    login_node = _pick_login_node(login_nodes)
    key_material = _fetch_ssh_key_for_space(space_name, key_secret_name)
    key_file_path: Optional[str] = None
    tunnel: Optional[SshTunnel] = None
    try:
        key_file_path = _write_key_file(key_material)
        logger.info(
            "[bluevela] opening tunnel: space=%s node=%s key_file=%s",
            space_name,
            login_node,
            key_file_path,
        )
        tunnel = SshTunnel(
            host=login_node,
            username=username,
            key_file=key_file_path,
            host_key_verification=ENABLE_SSH_HOST_KEY_VERIFICATION,
        )
        await tunnel.open()
        yield tunnel, BlueVelaConfig(
            login_node=login_node,
            username=username,
            workspace_remote_dir=workspace_remote_dir,
        )
    finally:
        if tunnel is not None:
            try:
                await tunnel.close()
            except Exception as e:
                logger.warning("[bluevela] tunnel close failed: %s", e)
        if key_file_path is not None:
            try:
                os.unlink(key_file_path)
            except OSError as e:
                logger.warning(
                    "[bluevela] failed to remove key file %s: %s", key_file_path, e
                )


__all__ = ["BlueVelaConfig", "open_bluevela_tunnel"]
