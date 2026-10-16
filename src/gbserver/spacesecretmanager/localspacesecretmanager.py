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
Secret manager from local directory.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Self, Union

from gbcommon.types.constants import get_gb_home_dir
from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
from gbserver.utils.logger import get_logger
from gbserver.utils.secretfile import (
    SUPPORTED_SECRET_FILE_EXTENSIONS,
    load_secret_file,
    write_secret_file,
)

logger = get_logger(__name__)


class LocalSpaceSecretManager(SpaceSecretManager):
    """Secret manager that fetches from the local filesystem secrets folder.

    Reads (get_secret/get_secrets) reload from disk on each call, and writes are
    read-modify-write on the secrets file without file locking, so concurrent
    writes to the same file could race and lose an update. Acceptable for the
    standalone single-user use case; multi-client concurrent writes would need a
    file lock (follow-up if that becomes a real scenario).
    """

    # Users should have access to all secrets in the space_name and in `public` space.
    # If space_name is empty, they should have access to only public space
    # A user can be part of many spaces, but they can access only one space's resources at a time

    SUPPORTED_EXTENSIONS = SUPPORTED_SECRET_FILE_EXTENSIONS

    def __init__(
        self: Self, uri: str, secrets_dir: Optional[Union[str, Path]] = None, **kwargs
    ) -> None:
        super().__init__(uri=uri, **kwargs)
        # secrets_dir is optional: when omitted (e.g. a space.yaml with `config: {}`),
        # default to <gb_home>/space_secrets — the sibling of the per-user secrets dir
        # (see usersecretmanager.factory). Resolved at call time via get_gb_home_dir()
        # so a GB_HOME_DIR override is honored. ~ and ${ENV} in an explicit value are
        # expanded so a committed space.yaml can carry a portable path.
        raw_dir = (
            os.path.join(get_gb_home_dir(), "space_secrets")
            if secrets_dir is None
            else str(secrets_dir)
        )
        self.dir = Path(os.path.expanduser(os.path.expandvars(raw_dir)))

    def get_secret(
        self: Self,
        secret_name: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> Any:
        """
        Gets as input the secret_name and first checks if the secret exists in the space and if it does, return it.
        If it does not exist, check if exists in the public space, and return it.
        """
        secrets = self._load_all_secrets(secrets_dir=self.dir)
        return (
            {"value": secrets[secret_name]}
            if secrets.get(secret_name) is not None
            else {}
        )

    def get_secrets(
        self: Self, username: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        # Reload from disk so reads reflect writes made via create/update/delete on
        # the same manager instance (the /space_secrets admin API does both). The
        # source is a local file, so re-reading is cheap.
        return self._load_all_secrets(secrets_dir=self.dir)

    def _load_all_secrets(self: Self, secrets_dir: Path) -> Dict[str, str]:
        """Load all the secrets from a directory by automatically detecting and loading
        one of the supported files: .env, .yaml/.yml, or .json."""
        secrets = {}
        dir_path = Path(secrets_dir)
        if not dir_path.exists():
            # A missing secrets dir is the normal "no secrets configured yet" state
            # (e.g. a fresh standalone checkout). Treat it as an empty secret set
            # rather than an error — writes (create_secret) create the dir on demand.
            logger.debug("Secrets path does not exist, treating as empty: %s", dir_path)
            return {}

        if dir_path.is_file():
            return self._load_from_file(dir_path)

        for secrets_file_path in dir_path.iterdir():
            if secrets_file_path.is_file() and (
                secrets_file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
                or secrets_file_path.name.lower() == ".env"
            ):
                new_secrets = self._load_from_file(secrets_file_path)
                if not isinstance(new_secrets, dict):
                    raise ValueError(
                        f"Invalid content in {secrets_file_path}: expected a dictionary"
                    )
                secrets.update(new_secrets)

        if not secrets:
            logger.warning(
                "No supported secrets files (%s) found in %s",
                ", ".join(self.SUPPORTED_EXTENSIONS),
                dir_path,
            )
        return secrets

    def _load_from_file(self: Self, file_path: Path) -> Dict[str, str]:
        """Load secrets from a specific file based on its extension."""
        return load_secret_file(file_path)

    def create_secret(
        self,
        secret_name: str,
        secret_value: str,
        secret_type: str = "arbitrary",
        secret_group_name: str = "",
    ) -> None:

        dir_path = self.dir
        if dir_path.is_file():
            target_file = dir_path
            if target_file.name == ".env":
                suffix = ".env"
            else:
                suffix = target_file.suffix.lower()
            if suffix not in [".yaml", ".yml", ".json", ".env"]:
                raise ValueError(
                    f"Unsupported secret file type '{suffix}'. Must be one of .env/.yaml/.yml/.json."
                )
        else:
            if not secret_group_name:
                raise ValueError(
                    f"secret_group_name cannot be empty when 'dir' {self.dir} is a directory."
                )
            dir_path.mkdir(parents=True, exist_ok=True)
            target_file = dir_path / f"{secret_group_name}.yaml"
        if target_file.exists():
            try:
                secrets = self._load_from_file(target_file)  # loads decoded secrets
            except Exception as e:
                logger.warning(
                    "Failed to load or decode secrets file %s: %s . "
                    + "Starting with an empty dictionary.",
                    target_file,
                    e,
                )
                secrets = {}
        else:
            secrets = {}

        if secret_name in secrets:
            logger.warning(
                "Secret '%s' already exists in '%s'. Overriding value.",
                secret_name,
                target_file.name,
            )
        secrets[secret_name] = secret_value
        self._write_encoded_secrets_to_file(target_file, secrets)
        logger.info("Secret '%s' saved to %s", secret_name, target_file)

    def _resolve_target_file(self: Self, secret_group_name: str) -> Path:
        """Resolve the on-disk file backing a secret group (mirrors create_secret)."""
        dir_path = self.dir
        if dir_path.is_file():
            return dir_path
        if not secret_group_name:
            raise ValueError(
                f"secret_group_name cannot be empty when 'dir' {self.dir} is a directory."
            )
        return dir_path / f"{secret_group_name}.yaml"

    def delete_secret(
        self: Self, secret_name: str, secret_group_name: str = ""
    ) -> None:
        target_file = self._resolve_target_file(secret_group_name)
        secrets = self._load_from_file(target_file) if target_file.exists() else {}
        if secret_name not in secrets:
            raise ValueError(
                f"Secret '{secret_name}' does not exist in '{target_file.name}'"
            )
        del secrets[secret_name]
        self._write_encoded_secrets_to_file(target_file, secrets)
        logger.info("Secret '%s' deleted from %s", secret_name, target_file)

    def _write_encoded_secrets_to_file(
        self, target_file: Path, secrets: Dict[str, str]
    ) -> None:
        """
        Writes secrets to a file after base64-encoding all values.

        Supports .env, .yaml/.yml, and .json file formats.
        """
        try:
            write_secret_file(target_file, secrets)
        except Exception as e:
            logger.error("Failed to write secrets to %s: %s", target_file, e)
