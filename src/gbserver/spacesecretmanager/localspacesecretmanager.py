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

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional, Self

import yaml
from dotenv import dotenv_values

from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LocalSpaceSecretManager(SpaceSecretManager):
    """Secret manager that fetches from the local filesystem secrets folder."""

    # Users should have access to all secrets in the space_name and in `public` space.
    # If space_name is empty, they should have access to only public space
    # A user can be part of many spaces, but they can access only one space's resources at a time

    SUPPORTED_EXTENSIONS = [".env", ".yaml", ".yml", ".json"]

    def __init__(self: Self, uri: str, secrets_dir: Path, **kwargs) -> None:
        super().__init__(uri=uri, **kwargs)
        space_name = ""  # How to get the space_name?
        self.secrets = self._load_all_secrets(secrets_dir=secrets_dir)
        self.dir = Path(secrets_dir)

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
        return (
            {"value": self.secrets[secret_name]}
            if self.secrets.get(secret_name) is not None
            else {}
        )

    def get_secrets(self: Self, username: Optional[str] = None) -> Optional[Dict[str, str]]:
        return self.secrets

    def _load_all_secrets(self: Self, secrets_dir: Path) -> Dict[str, str]:
        """Load all the secrets from a directory by automatically detecting and loading
        one of the supported files: .env, .yaml/.yml, or .json."""
        secrets = {}
        dir_path = Path(secrets_dir)
        if not dir_path.exists():
            logger.error("Path does not exist: %s", dir_path)
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
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        secrets = {}
        if suffix == ".env" or name == ".env":
            logger.info("Loading secrets from dotenv file: %s", file_path)
            secrets = dotenv_values(file_path)
        elif suffix in [".yaml", ".yml"]:
            logger.info("Loading secrets from YAML file: %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            secrets = data if isinstance(data, dict) else {}
        elif suffix == ".json":
            try:
                logger.info("Loading secrets from JSON file: %s", file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                secrets = data if isinstance(data, dict) else {}
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse secrets JSON file %s: %s . Returning empty dict.",
                    file_path,
                    e,
                )
                return {}
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        decoded_secrets = {}

        # NEW FORMAT: spaces -> <space> -> secrets
        if "spaces" in secrets:
            for space_name, space_cfg in secrets["spaces"].items():
                space_secrets = space_cfg.get("secrets", {})
                for name, secret in space_secrets.items():
                    try:
                        payload = secret["payload"]
                        labels = secret.get("labels")
                        if labels and "encode:base64" in labels:
                            value = base64.b64decode(payload.encode("utf-8")).decode("utf-8")
                        else:
                            value = payload
                        decoded_secrets[name] = value
                    except Exception as e:
                        logger.error(
                            "Invalid secret entry for key: %s | value: %s error: %s",
                            name,
                            secret,
                            e,
                        )

        # BACKWARD COMPATIBILITY - OLD FORMAT: flat key -> base64 (old file format, if any)
        else:
            for key, value in secrets.items():
                try:
                    decoded_value = base64.b64decode(value).decode("utf-8")
                    decoded_secrets[key] = decoded_value
                except Exception as e:
                    logger.error(
                        "Invalid base64 value for key: %s | value: %s error: %s",
                        key,
                        value,
                        e,
                    )

        return decoded_secrets

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

    def _write_encoded_secrets_to_file(self, target_file: Path, secrets: Dict[str, str]) -> None:
        """
        Writes secrets to a file after base64-encoding all values.

        Supports .env, .yaml/.yml, and .json file formats.
        """
        try:
            encoded_secrets = {
                k: base64.b64encode(v.encode("utf-8")).decode("utf-8") for k, v in secrets.items()
            }
            if target_file.name == ".env":
                suffix = ".env"
            else:
                suffix = target_file.suffix.lower()
            if suffix == ".env":
                with open(target_file, "w", encoding="utf-8") as f:
                    for k, v in encoded_secrets.items():
                        f.write(f"{k}={v}\n")
            elif suffix in [".yaml", ".yml"]:
                with open(target_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(encoded_secrets, f, default_flow_style=False)
            elif suffix == ".json":
                with open(target_file, "w", encoding="utf-8") as f:
                    json.dump(encoded_secrets, f, indent=4)
            else:
                raise ValueError(f"Unsupported file type for secrets: {suffix}")
        except Exception as e:
            logger.error("Failed to write secrets to %s: %s", target_file, e)
