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

"""
The space.
"""

import glob
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Optional, Self, Union

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
from gbserver.types.constants import GBSERVER_PROCEED_WITHOUT_SECRETS, is_debug_mode
from gbserver.types.spaceconfig import SpaceConfig
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import write_local_secrets_file

logger = get_logger(__name__)

SPACE_YAML = "space.yaml"


class Space:
    """A space provides the context for a build."""

    def __init__(
        self: Self,
        uri: Union[URI, str],
        username: Optional[str] = None,
        force_fetch: bool = False,
    ):
        """Create the instance."""
        self.uristr = URI.get_uristr(uri)
        self.secrets = {}
        uriobj = URI.get_uri(uri=uri, default_scheme="file")
        tmppath = Path(tempfile.mkdtemp())
        uriobj.pull(dest=tmppath, force=force_fetch)
        space_yamls = glob.glob(str(tmppath / "**" / SPACE_YAML), recursive=True)
        builtins_uri = (Path(__file__).parent.parent / "builtins").as_uri()
        base_uris = [self.uristr, builtins_uri]
        if space_yamls is None or len(space_yamls) == 0:
            raise ValueError(f"No '{SPACE_YAML}' found at path: {tmppath}")
        self.space_config: SpaceConfig = SpaceConfig.from_yaml(Path(space_yamls[0]))
        if self.space_config is not None:
            if self.space_config.base_uris is not None:
                base_uris = base_uris + self.space_config.base_uris
            if self.space_config is not None:
                URI.set_space_config(self.space_config)
        self.secrets = self._fetch_secrets(username=username)

        SpaceURI.set_baseuris(base_uris=base_uris, space_secrets=self.secrets)
        Assetstore.load_assetstores_from_dir(tmppath, secrets=self.secrets)

        if not is_debug_mode():
            # TODO: for now only remove the experiments directory to be safe, but longterm we should be removing the whole tmppath dir
            exp_dirs = glob.glob(str(tmppath / "**" / "experiments"), recursive=True)
            for exp_dir in exp_dirs:
                shutil.rmtree(exp_dir)

    def get_secrets(self: Self) -> Dict[str, str]:
        """Returns the cached secrets for the space."""
        return self.secrets

    def _is_first_local_sync(self) -> bool:
        """Returns true if this is the first local sync from remote to local secrets
        for bootstrapping"""

        sm = self.space_config.secret_manager

        # if type is not local, no notion of syncing remote to local -> return False
        if sm.type != "local":
            return False

        assert (
            "secrets_dir" in sm.config
        ), "Local secret manager requires 'secrets_dir' in config"

        # Remote sync must be explicitly enabled
        if not sm.config.get("do_remote_sync", False):
            return False

        # Validate remote sync config
        assert (
            "remote_sync_config" in sm.config
        ), "'do_remote_sync' is true but 'remote_sync_config' is missing"

        remote_cfg = sm.config["remote_sync_config"]

        assert "type" in remote_cfg, "'remote_sync_config' must define 'type'"

        assert "config" in remote_cfg, "'remote_sync_config' must define 'config'"

        assert (
            "service_url" in remote_cfg["config"]
        ), "'remote_sync_config.config' must define 'service_url'"

        # gets the path to the secrets file
        secrets_dir = (
            Path(self.space_config.secret_manager.config["secrets_dir"])
            .expanduser()
            .resolve()
        )

        return not secrets_dir.exists() or secrets_dir.stat().st_size == 0

    def _fetch_secrets(self: Self, username: Optional[str] = None) -> Dict[str, str]:
        logger.info("_fetch_secrets start")

        sm_cfg = self.space_config.secret_manager.config

        # By default, setting sync to False
        do_sync = False

        # Checks if this is the first local sync i.e. local secrets path does not exists + assertion checks,
        #  - if yes, then we sync all the remote ibmcloud secrets to the
        #  local secrets file (`secrets_dir`) to start off with.

        if self._is_first_local_sync():
            logger.info(
                "First local secrets sync detected; bootstrapping from ibmcloud"
            )
            do_sync = True

        # If not the first local sync, i.e. secrets file exists, then fetch whether user
        # wants to do the remote sync
        elif sm_cfg.get("do_remote_sync", False):
            logger.info("Not the first local sync for bootstrapping")
            do_sync = sm_cfg.get("do_remote_sync", False)
            logger.info(
                "==== Flag to remote_sync is set to = %s, proceeding with a remote sync ====;",
                do_sync,
            )

        else:
            logger.warning(
                f"=== Remote sync is set to {do_sync}, proceeding without a remote sync ==="
            )

        # If sync is enabled -> execute the sync operation from remote to local secrets manager

        if do_sync:
            remote_cfg = sm_cfg["remote_sync_config"]
            remote_type = remote_cfg["type"]
            remote_config = remote_cfg["config"]

            logger.info(f"==== Remote secrets type: {remote_type} ====")

            remote_secrets_manager: SpaceSecretManager = (
                SpaceSecretManager.get_spacesecretmanager(
                    secret_manager_type=remote_type,
                    uri=self.uristr,
                    **remote_config,
                )
            )
            # returns secrets with payload, secret group and the labels in the required format
            remote_secrets = remote_secrets_manager.get_secrets_with_groups(
                username=username
            )
            logger.info(
                f"======== Fetched {len(remote_secrets)} secrets from remote for sync ========"
            )
            secrets_file = Path(sm_cfg["secrets_dir"])

            write_local_secrets_file(
                secrets_file=secrets_file,
                space_name=self.space_config.name,
                secrets=remote_secrets,
            )

        try:
            self.secret_manager: SpaceSecretManager = (
                SpaceSecretManager.get_spacesecretmanager(
                    secret_manager_type=self.space_config.secret_manager.type,
                    uri=self.uristr,
                    **self.space_config.secret_manager.config,
                )
            )

            logger.info("created a secret_manager: %s", self.secret_manager)
            secrets = self.secret_manager.get_secrets(username=username)
            if secrets is None:
                if not GBSERVER_PROCEED_WITHOUT_SECRETS:
                    raise ValueError("secrets is None")
                secrets = {}
            logger.info(
                "fetched %d secrets using the secret manager: %s",
                len(secrets),
                self.secret_manager,
            )
            return secrets
        except Exception as e:
            logger.error(traceback.format_exc())
            if GBSERVER_PROCEED_WITHOUT_SECRETS:
                logger.error(
                    "failed to instantiate the secret manager: %s . Continuing without secrets.",
                    e,
                )
            else:
                raise ValueError(
                    "failed to instantiate the secret manager and fetch secrets"
                ) from e
            return {}
