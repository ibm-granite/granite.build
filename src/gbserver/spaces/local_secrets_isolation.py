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

"""Keep spaces from sharing the default local secret store.

With no ``secrets_dir``, LocalSpaceSecretManager keeps a space's secrets in
``<gb_home>/space_secrets/<space.yaml name>/``. Keying on the space.yaml ``name``
(rather than the registered name) lets standalone's aliases -- one space
registered as ``public``, ``standalone`` and ``local`` -- share one store. But
space.yaml names, unlike registered names, are not unique: two different spaces
copied from the same template would read each other's secrets. Registration
calls check_local_secrets_collision() to refuse such a space.
"""

import glob
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from gbcommon.uri.git import GitURI
from gbcommon.uri.uri import URI
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.spaceconfig import SpaceConfig
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

SPACE_YAML = "space.yaml"


class LocalSecretsCollisionError(ValueError):
    """Two different spaces would share one default local secrets directory."""


def load_space_config(space_uri: str) -> SpaceConfig:
    """Pull a space into a temporary directory and parse its space.yaml.

    The temporary copy is removed before returning.

    Args:
        space_uri: The space's URI (a bare path is treated as ``file:``).

    Returns:
        SpaceConfig: The parsed space.yaml.

    Raises:
        ValueError: If the space has no space.yaml.
    """
    uriobj = URI.get_uri(uri=space_uri, default_scheme="file")
    with tempfile.TemporaryDirectory() as tmpdir:
        uriobj.pull(dest=Path(tmpdir))
        space_yamls = glob.glob(str(Path(tmpdir) / "**" / SPACE_YAML), recursive=True)
        if not space_yamls:
            raise ValueError(f"No '{SPACE_YAML}' found for space at {space_uri}")
        return SpaceConfig.from_yaml(Path(space_yamls[0]))


def default_local_secrets_key(space_config: SpaceConfig) -> Optional[str]:
    """Return the directory name a space uses in the default local secret store.

    Args:
        space_config: The space's parsed space.yaml.

    Returns:
        Optional[str]: The space.yaml ``name`` if the space's secret manager is
        ``local`` -- directly or inside a ``hybrid`` chain -- with no
        ``secrets_dir``; otherwise None (the space does not use the shared
        default location).
    """
    secret_manager = space_config.secret_manager
    managers: List[Dict[str, Any]] = [
        {"type": secret_manager.type, "config": secret_manager.config}
    ]
    if secret_manager.type == "hybrid":
        managers = (secret_manager.config or {}).get("managers") or []
    for manager in managers:
        config = manager.get("config") or {}
        if manager.get("type") == "local" and not config.get("secrets_dir"):
            return space_config.name
    return None


def _registered_key(space: StoredSpace) -> Optional[str]:
    """Return a registered space's default-local-secrets key, or None.

    A space whose space.yaml cannot be loaded (e.g. a stale row pointing at a
    deleted directory) is skipped with a warning rather than blocking the
    registration of an unrelated space.

    Args:
        space: The registered space.

    Returns:
        Optional[str]: See default_local_secrets_key().
    """
    try:
        config_uri = GitURI.get_gb_space_config_uri(uri=space.git_repo_uri)
        return default_local_secrets_key(load_space_config(config_uri))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Skipping local-secrets collision check against space '%s' (%s):"
            " could not read its space.yaml: %s",
            space.name,
            space.git_repo_uri,
            e,
        )
        return None


def check_local_secrets_collision(
    candidate: StoredSpace, registered: Iterable[StoredSpace]
) -> None:
    """Refuse to register a space that would share another's local secrets.

    Spaces with the same ``git_repo_uri`` are aliases of one space and may share.

    Args:
        candidate: The space about to be registered (or re-pointed).
        registered: The spaces already registered (the candidate's own row, if
            any, may be included).

    Raises:
        LocalSecretsCollisionError: If the candidate uses the default local
            secret store and a different registered space uses it under the
            same space.yaml name.
        ValueError: If the candidate's own space.yaml cannot be loaded.
    """
    key = default_local_secrets_key(
        load_space_config(GitURI.get_gb_space_config_uri(uri=candidate.git_repo_uri))
    )
    if key is None:
        return
    for other in registered:
        if other.git_repo_uri == candidate.git_repo_uri:
            continue
        if _registered_key(other) == key:
            raise LocalSecretsCollisionError(
                f"space '{candidate.name}' ({candidate.git_repo_uri}) and registered"
                f" space '{other.name}' ({other.git_repo_uri}) both use the default"
                f" local secret store with space.yaml name '{key}', so they would"
                " share secrets; give one a different space.yaml name or an explicit"
                " secret_manager.config.secrets_dir"
            )
