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

"""Unified environment configuration for GB (gbcli + gbserver)."""

import os
from typing import Any, Dict, Optional, Self

from pydantic import BaseModel

from gbcommon.types.constants import DEFAULT_GH_DOMAIN


def getenv_boolean(envname: str, default: bool = False) -> bool:
    """Evaluate the environment variable and return as a boolean value."""
    value = os.getenv(envname)
    if value is None:
        return default
    value_normalized = str(value).lower()
    return value_normalized not in ["false", "null", "undefined", "no", "0", ""]


class GBEnvConfig(BaseModel):
    """Unified environment configuration for gbcli and gbserver."""

    env: str
    """The GB env name. One of PROD, STAGING, DEV, or STANDALONE."""

    lakehouse_environment: str
    """The lakehouse environment to use. One of PROD or STAGING."""

    feature_flags: Dict[str, bool] = {}
    """Feature flags for this environment."""

    space_config_branch_name: str = ""
    """The branch in a space repo holding steps, assetstores, etc."""

    # --- gbcli-origin fields ---

    gbserver_host: str = ""
    """The gbserver API endpoint URL."""

    default_space: str = ""
    """The default space name."""

    dmf_ui: str = ""
    """The full DMF UI base URL (e.g. https://ui.dmf.vpc-int.res.ibm.com)."""

    config_spaces: str = ""
    """Config section name for spaces."""

    config_profile: str = ""
    """Config section name for profiles."""

    server_log_application_name: str = ""
    """Application name for logging."""

    branch_assets: str = ""
    """Git branch name for assets."""

    hf_organization: str = ""
    """HuggingFace organization."""

    # --- gbserver-origin fields ---

    dashboard_instance: str = ""
    """The dashboard URL for build status."""

    public_space_git_uri: str = ""
    """The URI of the public space git repo."""

    public_space_lh_subnamespace: str = ""
    """The child name of the Lakehouse namespace under the main GB namespace."""

    buildwatcher_deployment_yaml: str = ""
    """The location of the buildwatcher's deployment yaml."""

    default_pod_namespace: str = ""
    """The default K8s namespace for servers."""

    default_sql_schema: str = ""
    """The default schema to use in SQL storage."""

    def model_post_init(self: Self, context: Any, /) -> None:
        if self.env == "":
            raise ValueError("field env cannot be empty")


DEFAULT_GB_ENVIRONMENT = "PROD"

_GB_ENVIRONMENT_CONFIGS: Dict[str, GBEnvConfig] = {
    "PROD": GBEnvConfig(
        env="PROD",
        lakehouse_environment="PROD",
        space_config_branch_name="gbspace-config",
        # gbcli
        gbserver_host="https://api.llm-build-prod.vpc-int.res.ibm.com",
        default_space="public",
        dmf_ui="https://ui.dmf.vpc-int.res.ibm.com",
        config_spaces="gb.spaces",
        config_profile="gb.spaces.profiles",
        server_log_application_name="llm-build-prod",
        branch_assets="gbspace-config",
        hf_organization="ibm-research",
        feature_flags={
            "gbserver_build_events": getenv_boolean("GBSERVER_BUILD_EVENTS", True),
            "gbserver_artifact_filter": getenv_boolean(
                "GBSERVER_ARTIFACT_FILTER", True
            ),
            "gbserver_build_update": getenv_boolean("GBSERVER_BUILD_UPDATE", True),
        },
        # gbserver
        dashboard_instance="https://api.llm-build-dev.vpc-int.res.ibm.com",
        public_space_git_uri=f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public",
        public_space_lh_subnamespace="public",
        buildwatcher_deployment_yaml="k8s/dep-build-runner.yaml",
        default_pod_namespace=os.getenv(
            "GBSERVER_BACKEND_SERVER_NAMESPACE_PROD", "llm-build-prod"
        ),
        default_sql_schema="granite_dot_build_prod",
    ),
    "STAGING": GBEnvConfig(
        env="STAGING",
        lakehouse_environment="STAGING",
        space_config_branch_name="gbspace-config",
        # gbcli
        gbserver_host="https://api.llm-build-staging.vpc-int.res.ibm.com",
        default_space="public",
        dmf_ui="https://ui.dmf-staging.vpc-int.res.ibm.com",
        config_spaces="staging.gb.spaces",
        config_profile="staging.gb.spaces.profiles",
        server_log_application_name="llm-build-staging",
        branch_assets="gbspace-config-dev",
        hf_organization="ibm-research",
        feature_flags={
            "gbserver_build_events": getenv_boolean("GBSERVER_BUILD_EVENTS", True),
            "gbserver_artifact_filter": getenv_boolean(
                "GBSERVER_ARTIFACT_FILTER", True
            ),
            "gbserver_build_update": getenv_boolean("GBSERVER_BUILD_UPDATE", True),
        },
        # gbserver
        dashboard_instance="https://api.llm-build-dev.vpc-int.res.ibm.com",
        public_space_git_uri=f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test",
        public_space_lh_subnamespace="public",
        buildwatcher_deployment_yaml="k8s/dep-build-runner.yaml",
        default_pod_namespace=os.getenv(
            "GBSERVER_BACKEND_SERVER_NAMESPACE_STAGING", "llm-build-staging"
        ),
        default_sql_schema="granite_dot_build_staging",
    ),
    "DEV": GBEnvConfig(
        env="DEV",
        lakehouse_environment="STAGING",
        space_config_branch_name="gbspace-config",
        # gbcli
        gbserver_host="https://api.llm-build-dev.vpc-int.res.ibm.com",
        default_space="public",
        dmf_ui="https://ui2.dmf-staging.vpc-int.res.ibm.com",
        config_spaces="dev.gb.spaces",
        config_profile="dev.gb.spaces.profiles",
        server_log_application_name="llm-build-dev",
        branch_assets="gbspace-config-dev",
        hf_organization="ibm-research",
        feature_flags={
            "gbserver_build_events": getenv_boolean("GBSERVER_BUILD_EVENTS", True),
            "gbserver_artifact_filter": getenv_boolean(
                "GBSERVER_ARTIFACT_FILTER", True
            ),
            "gbserver_build_update": getenv_boolean("GBSERVER_BUILD_UPDATE", True),
        },
        # gbserver
        dashboard_instance="https://api.llm-build-dev.vpc-int.res.ibm.com",
        public_space_git_uri=f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public-dev",
        public_space_lh_subnamespace="public_dev",
        buildwatcher_deployment_yaml="k8s/dep-build-runner.yaml",
        default_pod_namespace=os.getenv(
            "GBSERVER_BACKEND_SERVER_NAMESPACE_DEV", "llm-build-dev"
        ),
        default_sql_schema="granite_dot_build_dev",
    ),
    "STANDALONE": GBEnvConfig(
        env="STANDALONE",
        lakehouse_environment="",
        space_config_branch_name="main",
        # gbcli
        gbserver_host="http://localhost:8080",
        default_space="standalone",
        dmf_ui="",
        config_spaces="",
        config_profile="",
        server_log_application_name="gbserver-standalone",
        branch_assets="",
        hf_organization="ibm-research",
        feature_flags={
            "build_start_via_github": False,
            "gbserver_build_events": True,
            "gbserver_artifact_filter": False,
            "gbserver_build_update": True,
        },
        # gbserver
        dashboard_instance="",
        public_space_git_uri="",
        public_space_lh_subnamespace="",
        buildwatcher_deployment_yaml="",
        default_pod_namespace="default",
        default_sql_schema="standalone",
    ),
}


def gb_env_normalize(value: Optional[str], source: str = "input") -> Optional[str]:
    """Normalize user-facing env name to canonical form.

    Returns None if value is None/empty. Raises ValueError on invalid input.
    """
    if not value:
        return None
    v = value.lower()
    if v in ("prod", "production"):
        return "PROD"
    elif v in ("staging",):
        return "STAGING"
    elif v in ("dev", "development"):
        return "DEV"
    elif v in ("standalone", "local"):
        return "STANDALONE"
    else:
        raise ValueError(f"Error: {source} has invalid value '{value}'")


def gb_environment() -> str:
    """Read GB_ENVIRONMENT env var, normalize, default to PROD."""
    raw = os.environ.get("GB_ENVIRONMENT")
    normalized = gb_env_normalize(raw, "Environment variable GB_ENVIRONMENT")
    return normalized if normalized else DEFAULT_GB_ENVIRONMENT


def gb_environment_config(gb_env: Optional[str] = None) -> GBEnvConfig:
    """Get the config for the given env. If gb_env is None or empty, uses gb_environment()."""
    if not gb_env:
        gb_env = gb_environment()
    if gb_env not in _GB_ENVIRONMENT_CONFIGS:
        valid_keys = list(_GB_ENVIRONMENT_CONFIGS.keys())
        raise ValueError(
            f"unknown GB environment: {gb_env}, expected one of {valid_keys}"
        )
    return _GB_ENVIRONMENT_CONFIGS[gb_env]


def is_standalone() -> bool:
    """Return True if the current environment is STANDALONE."""
    return gb_environment() == "STANDALONE"


def add_environment_config(config_dict: Dict) -> GBEnvConfig:
    """Add or overwrite a runtime config entry. Used by gbserver for --server-runtime-config."""
    config = GBEnvConfig.model_validate(config_dict)
    if config.env in _GB_ENVIRONMENT_CONFIGS:
        old = _GB_ENVIRONMENT_CONFIGS[config.env]
        print(
            f"[WARNING] the environment config '{config.env}'"
            + f" already exists: {old} , overwriting with {config}"
        )
    _GB_ENVIRONMENT_CONFIGS[config.env] = config
    return config
