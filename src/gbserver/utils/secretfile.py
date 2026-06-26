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

"""Shared helpers for reading and writing local secret files.

Secret values are stored base64-encoded so the on-disk representation is the
same regardless of the file format (.env / .yaml / .yml / .json). These helpers
are used by both the space-level ``LocalSpaceSecretManager`` and the per-user
``LocalUserSecretManager`` so the two stay in sync.
"""

import base64
import json
from pathlib import Path
from typing import Dict

import yaml
from dotenv import dotenv_values

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_SECRET_FILE_EXTENSIONS = [".env", ".yaml", ".yml", ".json"]


def _file_suffix(file_path: Path) -> str:
    """Return the effective suffix for a secrets file, treating ``.env`` specially."""
    if file_path.name.lower() == ".env":
        return ".env"
    return file_path.suffix.lower()


def load_secret_file(file_path: Path) -> Dict[str, str]:
    """Load and base64-decode secrets from a single file.

    Supports two on-disk shapes:
      * NEW: ``spaces -> <space> -> secrets -> <name> -> {payload, labels}``
      * OLD: flat ``<name>: <base64-value>``

    Returns a flat ``{name: decoded_value}`` dict. Unparseable entries are
    skipped with a logged error rather than raising.
    """
    suffix = _file_suffix(file_path)
    raw: Dict = {}
    if suffix == ".env":
        logger.info("Loading secrets from dotenv file: %s", file_path)
        raw = dict(dotenv_values(file_path))
    elif suffix in [".yaml", ".yml"]:
        logger.info("Loading secrets from YAML file: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        raw = data if isinstance(data, dict) else {}
    elif suffix == ".json":
        try:
            logger.info("Loading secrets from JSON file: %s", file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse secrets JSON file %s: %s . Returning empty dict.",
                file_path,
                e,
            )
            return {}
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    decoded_secrets: Dict[str, str] = {}

    # NEW FORMAT: spaces -> <space> -> secrets
    if "spaces" in raw:
        for _space_name, space_cfg in raw["spaces"].items():
            space_secrets = space_cfg.get("secrets", {})
            for name, secret in space_secrets.items():
                try:
                    payload = secret["payload"]
                    labels = secret.get("labels")
                    if labels and "encode:base64" in labels:
                        value = base64.b64decode(payload.encode("utf-8")).decode(
                            "utf-8"
                        )
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
    # BACKWARD COMPATIBILITY - OLD FORMAT: flat key -> base64
    else:
        for key, value in raw.items():
            try:
                decoded_secrets[key] = base64.b64decode(value).decode("utf-8")
            except Exception as e:
                logger.error(
                    "Invalid base64 value for key: %s | value: %s error: %s",
                    key,
                    value,
                    e,
                )

    return decoded_secrets


def write_secret_file(target_file: Path, secrets: Dict[str, str]) -> None:
    """Write secrets to a file after base64-encoding all values.

    Supports .env, .yaml/.yml, and .json file formats (inferred from the
    target file's suffix).
    """
    encoded_secrets = {
        k: base64.b64encode(v.encode("utf-8")).decode("utf-8")
        for k, v in secrets.items()
    }
    suffix = _file_suffix(target_file)
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
