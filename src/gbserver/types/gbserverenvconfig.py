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

"""Environment specific config."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Self

import yaml
from pydantic import BaseModel

from gbserver.types.constants_base import (
    BACKEND_SERVER_NAMESPACE_DEV,
    BACKEND_SERVER_NAMESPACE_PROD,
    BACKEND_SERVER_NAMESPACE_STAGING,
    DEFAULT_GB_ENVIRONMENT,
    DEFAULT_GH_DOMAIN,
    getenv_boolean,
)


class GBServerEnvConfig(BaseModel):
    """
    The server runtime config.
    """

    env: str
    """The GB env name. One of PROD, STAGING or DEV."""

    lakehouse_environment: str
    """The lakehouse environment to use.  One of PROD or STAGING """

    dashboard_instance: str = ""
    """The portion of the dashboard url specific to DMF PROD, STAGING or STAGING2."""

    dmf_instance: str
    """The portion of the lineage url specific to DMF PROD, STAGING or STAGING2."""

    public_space_git_uri: str
    """The uri of the public space git repo (https://...) """

    public_space_lh_subnamespace: str
    """The child name of the Lakehouse namespace under the main GB name space (i.e. granite_dot_build). For example, public. """

    buildwatcher_deployment_yaml: str
    """The location of the buildwatcher's deployment yaml for use by the BuildRunnerJob"""

    default_pod_namespace: str
    """The default namespace in which our servers are run in the cluster
       And used for testing when not running buildwatcher in the cluster.
    """

    default_sql_schema: str
    """The default schema to use in SQL storage."""

    space_config_branch_name: str
    """The name of the branch in a space repo holding the steps, assetstores, etc.
    Usually one of gspace-cofig, gbspace-config-staging or gbspace-config-dev
    """

    feature_flags: Dict[str, bool] = {}
    """Feature flags for this environment. Each flag is a string key mapped to a boolean value.
    Values are evaluated at startup via getenv_boolean() so they can be overridden by env vars.
    """

    def model_post_init(self: Self, context: Any, /) -> None:
        if self.env == "":
            raise ValueError("field env cannot be empty")


_GBSERVER_ENVIRONMENT_CONFIGS = {
    "PROD": GBServerEnvConfig(
        env="PROD",
        lakehouse_environment="PROD",
        dashboard_instance="https://api.llm-build-dev.vpc-int.res.ibm.com",
        dmf_instance="ui.dmf",
        public_space_git_uri=f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public",
        public_space_lh_subnamespace="public",
        # buildwatcher_deployment_yaml="k8s/prod/dep-gbserver-build-runner.yaml",
        buildwatcher_deployment_yaml="k8s/dep-build-runner.yaml",
        default_pod_namespace=BACKEND_SERVER_NAMESPACE_PROD,
        default_sql_schema="granite_dot_build_prod",
        space_config_branch_name="gbspace-config",
        feature_flags={
            "lakehouse_space_membership": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_SPACE_MEMBERSHIP", False
            ),
            "lakehouse_lineage": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_LINEAGE", False
            ),
        },
    ),
    "STAGING": GBServerEnvConfig(
        env="STAGING",
        lakehouse_environment="STAGING",
        dashboard_instance="https://api.llm-build-dev.vpc-int.res.ibm.com",
        dmf_instance="ui.dmf-staging",
        public_space_git_uri=f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test",
        public_space_lh_subnamespace="public",
        # buildwatcher_deployment_yaml="k8s/staging/dep-gbserver-build-runner.yaml",
        buildwatcher_deployment_yaml="k8s/dep-build-runner.yaml",
        default_pod_namespace=BACKEND_SERVER_NAMESPACE_STAGING,
        default_sql_schema="granite_dot_build_staging",
        space_config_branch_name="gbspace-config",
        feature_flags={
            "lakehouse_space_membership": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_SPACE_MEMBERSHIP", False
            ),
            "lakehouse_lineage": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_LINEAGE", False
            ),
        },
    ),
    "DEV": GBServerEnvConfig(
        env="DEV",
        lakehouse_environment="STAGING",
        dashboard_instance="https://api.llm-build-dev.vpc-int.res.ibm.com",
        dmf_instance="ui2.dmf-staging",
        public_space_git_uri=f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public-dev",
        public_space_lh_subnamespace="public_dev",
        # buildwatcher_deployment_yaml="k8s/dev/dep-gbserver-build-runner.yaml",
        buildwatcher_deployment_yaml="k8s/dep-build-runner.yaml",
        default_pod_namespace=BACKEND_SERVER_NAMESPACE_DEV,
        default_sql_schema="granite_dot_build_dev",
        # For now this is the same as staging and prod, but when we have a CI/CD with a dev environtment,
        # we will likely want to change this to be something like gbspace-config-dev.
        space_config_branch_name="gbspace-config",
        feature_flags={
            "lakehouse_space_membership": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_SPACE_MEMBERSHIP", False
            ),
            "lakehouse_lineage": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_LINEAGE", False
            ),
        },
    ),
    "STANDALONE": GBServerEnvConfig(
        env="STANDALONE",
        lakehouse_environment="",
        dmf_instance="",
        public_space_git_uri="",
        public_space_lh_subnamespace="",
        buildwatcher_deployment_yaml="",
        default_pod_namespace="default",
        default_sql_schema="standalone",
        space_config_branch_name="main",
        feature_flags={
            "lakehouse_space_membership": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_SPACE_MEMBERSHIP", False
            ),
            "lakehouse_lineage": getenv_boolean(
                "GBSERVER_FEATURE_LAKEHOUSE_LINEAGE", False
            ),
        },
    ),
}


def add_server_runtime_config(config_dict: Dict) -> GBServerEnvConfig:
    """Add another server runtime config to the dict of built-in ones."""
    print("[INFO] add_server_runtime_config config_dict:", config_dict)
    config = GBServerEnvConfig.model_validate(config_dict)
    print("[INFO] add_server_runtime_config config:", config)
    if config.env in _GBSERVER_ENVIRONMENT_CONFIGS:
        old = _GBSERVER_ENVIRONMENT_CONFIGS[config.env]
        print(
            f"[WARNING] the server runtime config '{config.env}'"
            + f" already exists: {old} , overwriting with {config}"
        )
    _GBSERVER_ENVIRONMENT_CONFIGS[config.env] = config
    return config


_LOADED_EXTRA_SERVER_RUNTIME_CONFIGS = False


def load_extra_server_runtime_configs() -> Optional[GBServerEnvConfig]:
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

    # print("[INFO] sys.argv:", sys.argv)

    parser = argparse.ArgumentParser(add_help=False)  # avoid usage info
    parser.add_argument(
        "--server-runtime-config",
        dest="server_runtime_config",
        required=False,
        type=Path,
        default=None,
        help="Path to a server runtime config file.",
    )
    # NOTE: there is a danger of sub-commands being treated as file path if the user forgets to give a path
    # This will throw usage errors because there are unknown flags and commands
    # args = parser.parse_args()
    known_args = None
    unknown_args = None
    try:
        known_args, unknown_args = parser.parse_known_args()
    except SystemExit as e:
        # Could also be triggered by the --help flag which prints usage and exits.
        # add_help=False does stop this, but keep it just in case.
        print("[ERROR] parse_known_args caught SystemExit, error:", e)
        return None
    except Exception as e:
        print("[ERROR] parse_known_args failed to parse, error:", e)
        return None
    # print(
    #     "[DEBUG] parse_known_args known_args:",
    #     known_args,
    #     "unknown_args:",
    #     unknown_args,
    # )
    server_runtime_config_path = known_args.server_runtime_config
    # print("[INFO] server_runtime_config_path:", server_runtime_config_path)
    if server_runtime_config_path is None:
        # print("[INFO] server_runtime_config_path is None")
        return None
    assert isinstance(
        server_runtime_config_path, Path
    ), f"invalid server_runtime_config_path: {known_args}"
    assert (
        server_runtime_config_path.is_file()
    ), f"expected server runtime config to be a file: '{server_runtime_config_path}'"
    with open(server_runtime_config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = add_server_runtime_config(config_dict=config_dict)
    return config


def gb_environment_config(gb_env: str) -> GBServerEnvConfig:
    """Get the config for the llm.build environment. and raise exception if bad env name is given
    Currently supported values include DEV, STAGING and PROD.
    """
    loaded_config = load_extra_server_runtime_configs()
    if gb_env == "":
        gb_env = loaded_config.env if loaded_config else DEFAULT_GB_ENVIRONMENT
    valid_keys = list(_GBSERVER_ENVIRONMENT_CONFIGS.keys())
    assert (
        gb_env in _GBSERVER_ENVIRONMENT_CONFIGS
    ), f"unknown GB environment: {gb_env} , expected one of {valid_keys}"
    return _GBSERVER_ENVIRONMENT_CONFIGS[gb_env]
