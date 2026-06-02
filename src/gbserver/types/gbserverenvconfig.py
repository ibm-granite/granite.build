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

"""Environment specific config — thin wrapper over gbcommon.types.gbenvconfig."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

from gbcommon.types.gbenvconfig import (
    DEFAULT_GB_ENVIRONMENT,
    GBEnvConfig,
    add_environment_config,
)
from gbcommon.types.gbenvconfig import (
    gb_environment_config as _common_gb_environment_config,
)

# Backwards-compatible alias
GBServerEnvConfig = GBEnvConfig


_LOADED_EXTRA_SERVER_RUNTIME_CONFIGS = False


def load_extra_server_runtime_configs() -> Optional[GBEnvConfig]:
    """
    Parse the CLI args and add another server runtime config
    to the dict of built-in ones.
    This new added one will be automatically selected unless
    the GB_ENVIRONMENT env var is specified.
    """
    global _LOADED_EXTRA_SERVER_RUNTIME_CONFIGS
    if _LOADED_EXTRA_SERVER_RUNTIME_CONFIGS:
        return None
    _LOADED_EXTRA_SERVER_RUNTIME_CONFIGS = True

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--server-runtime-config",
        dest="server_runtime_config",
        required=False,
        type=Path,
        default=None,
        help="Path to a server runtime config file.",
    )
    known_args = None
    try:
        known_args, unknown_args = parser.parse_known_args()
    except SystemExit as e:
        print("[ERROR] parse_known_args caught SystemExit, error:", e)
        return None
    except Exception as e:
        print("[ERROR] parse_known_args failed to parse, error:", e)
        return None

    server_runtime_config_path = known_args.server_runtime_config
    if server_runtime_config_path is None:
        return None
    assert isinstance(
        server_runtime_config_path, Path
    ), f"invalid server_runtime_config_path: {known_args}"
    assert (
        server_runtime_config_path.is_file()
    ), f"expected server runtime config to be a file: '{server_runtime_config_path}'"
    with open(server_runtime_config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = add_environment_config(config_dict=config_dict)
    return config


def gb_environment_config(gb_env: str = "") -> GBEnvConfig:
    """Server-specific wrapper that loads extra runtime configs first."""
    loaded_config = load_extra_server_runtime_configs()
    if gb_env == "":
        gb_env = loaded_config.env if loaded_config else None
    return _common_gb_environment_config(gb_env)
