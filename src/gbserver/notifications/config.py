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

"""Notification configuration loader for standalone mode."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATHS = [
    Path.home() / ".gbserver" / "notifications.yaml",
    Path(".gb") / "notifications.yaml",
]


def load_notification_config(
    config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load standalone notification configuration from a YAML file.

    If config_path is provided, use it directly. Otherwise search default paths:
      - ~/.gbserver/notifications.yaml
      - .gb/notifications.yaml

    If no file is found, returns an empty list.

    For each notification entry, keys ending in '_env' are resolved by reading
    the corresponding environment variable. For example:
      bot_token_env: "TELEGRAM_BOT_TOKEN"
    becomes:
      bot_token: <value of os.getenv("TELEGRAM_BOT_TOKEN")>

    Args:
        config_path: Optional explicit path to a YAML configuration file.

    Returns:
        A list of notification configuration dictionaries.
    """
    resolved_path = _find_config_file(config_path)
    if resolved_path is None:
        return []

    try:
        with open(resolved_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Failed to load notification config from %s: %s", resolved_path, exc)
        return []

    if not isinstance(data, dict):
        return []

    notifications = data.get("notifications", [])
    if not isinstance(notifications, list):
        return []

    return [_resolve_env_keys(entry) for entry in notifications]


def _find_config_file(config_path: Optional[str]) -> Optional[Path]:
    """Locate the configuration file to use."""
    if config_path is not None:
        path = Path(config_path)
        if path.is_file():
            return path
        logger.warning("Notification config file not found: %s", config_path)
        return None

    for default_path in _DEFAULT_CONFIG_PATHS:
        if default_path.is_file():
            return default_path

    return None


def _resolve_env_keys(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve keys ending in '_env' by reading the named environment variable.

    For each key like 'bot_token_env', the value is treated as an environment
    variable name. The resolved value is stored under the key without the '_env'
    suffix (e.g., 'bot_token'). The original '_env' key is removed.
    """
    resolved = {}
    for key, value in entry.items():
        if key.endswith("_env") and isinstance(value, str):
            base_key = key[: -len("_env")]
            resolved[base_key] = os.getenv(value)
        else:
            resolved[key] = value
    return resolved
