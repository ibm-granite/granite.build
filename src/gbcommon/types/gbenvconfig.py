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

import logging
import os
from typing import Any, Dict, List, Optional, Self

from pydantic import BaseModel

from gbcommon.types.constants import DEFAULT_GH_DOMAIN

logger = logging.getLogger(__name__)

# The single source of truth for what counts as "false"/"true" when parsing a
# boolean from an environment-variable string. Anything set but not falsy is
# treated as true; the truthy set is used only to warn on unrecognized input.
_FALSY_TOKENS = frozenset({"false", "null", "undefined", "no", "off", "0", ""})
_TRUTHY_TOKENS = frozenset({"true", "yes", "on", "1"})


def parse_boolean(
    value: object | None, default: bool = False, *, strict: bool = False
) -> bool:
    """Parse an env-var-style string (or already-typed value) into a boolean.

    ``None`` (unset) → ``default``. Otherwise falsy tokens (``_FALSY_TOKENS``) →
    ``False``. For any other set value the two modes differ:

    - lenient (default): anything not falsy → ``True``; an unrecognized value (a
      likely typo, e.g. ``on-prod``) still → ``True`` but logs a warning.
    - ``strict=True``: only a recognized truthy token → ``True``; an unrecognized
      value → ``False`` with no warning. Use when a typo must fail *closed* — a
      flag whose safe default and safe failure mode are both ``False`` (e.g. an
      opt-in like making a HuggingFace repo ``public``).

    Accepts non-string input because YAML/JSON yields real ``bool``/``int``;
    ``str()`` maps those onto the same token sets, so a value and its quoted form
    resolve identically. Never raises — the single place the string→bool rule
    lives, shared by ``getenv_boolean`` and any caller holding the value.
    """
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _FALSY_TOKENS:
        return False
    if normalized not in _TRUTHY_TOKENS:
        if strict:
            return False
        logger.warning(
            "Unrecognized boolean value %r — treating as true; use one of %s / %s",
            value,
            sorted(_TRUTHY_TOKENS),
            sorted(_FALSY_TOKENS),
        )
    return True


def getenv_boolean(envname: str, default: bool = False) -> bool:
    """Evaluate the environment variable and return as a boolean value."""
    return parse_boolean(os.getenv(envname), default)


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

    web_ui_url: str = ""
    """The full web UI base URL."""

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

    hf_enterprise_organizations: Optional[List[str]] = None
    """HF organizations that use Enterprise resource groups.

    Used by the CLI (`gb artifact push/register --store hf`) to decide whether a
    resource group applies. ``None`` means every org is treated as Enterprise,
    preserving the behavior from before the Enterprise/non-Enterprise split.

    Keep in sync with ``config.enterprise_organizations`` in the hf asset store's
    ``store.yaml``, which the server-side push path reads. The duplication is
    deliberate: store.yaml lives in the space's git repo and is loaded
    server-side only, so the CLI cannot read it.
    """

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


def deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` applied, recursively.

    Nested dicts are merged key-by-key so an environment can override a single
    ``feature_flags`` entry without restating the rest; any other value replaces
    the base value outright.

    Neither input is rebound, but the copy is **shallow**: a value the override
    does not touch (a list, or a nested dict absent from the override) is carried
    into the result by reference, so mutating it through the result would also
    change ``base``. Every caller here feeds the result straight to
    ``GBEnvConfig``, and pydantic deep-copies on validation, so the resolved
    configs share no mutable state — but a caller that skips validation must copy
    what it intends to mutate.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


# Environment configs are defined values.yaml-style: a base dict holding the full
# set of values, plus one dict per deployed environment listing only what it
# overrides. See _GB_ENVIRONMENT_CONFIGS below for how they combine.
#
# The base carries the PROD values, so _GB_ENVIRONMENT_CONFIG_PROD is empty — it
# is kept so every deployed environment reads the same way at the call site.
# Because the base *is* PROD, editing a base value also changes PROD; to change
# it for STAGING/DEV only, put it in each of their dicts instead.
#
# STANDALONE does not use the base: it is fully local and would override nearly
# everything, so it is written out in full below.
_GB_ENVIRONMENT_CONFIG_BASE = {
    "env": "PROD",
    "lakehouse_environment": "PROD",
    "space_config_branch_name": "gbspace-config",
    # gbcli
    "gbserver_host": "https://api.llm-build-prod.vpc-int.res.ibm.com",
    "default_space": "public",
    "web_ui_url": "https://dashboard.llm-build-prod.vpc-int.res.ibm.com",
    "config_spaces": "gb.spaces",
    "config_profile": "gb.spaces.profiles",
    "server_log_application_name": "llm-build-prod",
    "branch_assets": "gbspace-config",
    "hf_organization": "ibm-research",
    "hf_enterprise_organizations": ["ibm-research", "ibm-granite"],
    "feature_flags": {
        "gbserver_build_events": getenv_boolean("GBSERVER_BUILD_EVENTS", True),
        "gbserver_artifact_filter": getenv_boolean("GBSERVER_ARTIFACT_FILTER", True),
        "gbserver_build_update": getenv_boolean("GBSERVER_BUILD_UPDATE", True),
    },
    # gbserver
    # NOTE: the DEV api host, for every deployed environment. Intentional.
    "dashboard_instance": "https://api.llm-build-dev.vpc-int.res.ibm.com",
    "public_space_git_uri": f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public",
    "public_space_lh_subnamespace": "public",
    "buildwatcher_deployment_yaml": "k8s/dep-build-runner.yaml",
    "default_pod_namespace": os.getenv(
        "GBSERVER_BACKEND_SERVER_NAMESPACE_PROD", "llm-build-prod"
    ),
    "default_sql_schema": "granite_dot_build_prod",
}

# The base already holds the PROD values, so PROD overrides nothing.
_GB_ENVIRONMENT_CONFIG_PROD: dict = {}

_GB_ENVIRONMENT_CONFIG_STAGING = {
    "env": "STAGING",
    "lakehouse_environment": "STAGING",
    "branch_assets": "gbspace-config-dev",
    # gbcli
    "gbserver_host": "https://api.llm-build-staging.vpc-int.res.ibm.com",
    "web_ui_url": "https://dashboard.llm-build-staging.vpc-int.res.ibm.com",
    "config_spaces": "staging.gb.spaces",
    "config_profile": "staging.gb.spaces.profiles",
    "server_log_application_name": "llm-build-staging",
    # gbserver
    "public_space_git_uri": f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gb-test",
    "default_pod_namespace": os.getenv(
        "GBSERVER_BACKEND_SERVER_NAMESPACE_STAGING", "llm-build-staging"
    ),
    "default_sql_schema": "granite_dot_build_staging",
}

_GB_ENVIRONMENT_CONFIG_DEV = {
    "env": "DEV",
    # DEV shares the STAGING lakehouse.
    "lakehouse_environment": "STAGING",
    "branch_assets": "gbspace-config-dev",
    # gbcli
    "gbserver_host": "https://api.llm-build-dev.vpc-int.res.ibm.com",
    "web_ui_url": "https://dashboard.llm-build-dev.vpc-int.res.ibm.com",
    "config_spaces": "dev.gb.spaces",
    "config_profile": "dev.gb.spaces.profiles",
    "server_log_application_name": "llm-build-dev",
    # gbserver
    "public_space_git_uri": f"https://{DEFAULT_GH_DOMAIN}/granite-dot-build/gbspace-public-dev",
    "public_space_lh_subnamespace": "public_dev",
    "default_pod_namespace": os.getenv(
        "GBSERVER_BACKEND_SERVER_NAMESPACE_DEV", "llm-build-dev"
    ),
    "default_sql_schema": "granite_dot_build_dev",
}

_GB_ENVIRONMENT_CONFIGS: Dict[str, GBEnvConfig] = {
    "PROD": GBEnvConfig(
        **deep_merge(_GB_ENVIRONMENT_CONFIG_BASE, _GB_ENVIRONMENT_CONFIG_PROD)
    ),
    "STAGING": GBEnvConfig(
        **deep_merge(_GB_ENVIRONMENT_CONFIG_BASE, _GB_ENVIRONMENT_CONFIG_STAGING)
    ),
    "DEV": GBEnvConfig(
        **deep_merge(_GB_ENVIRONMENT_CONFIG_BASE, _GB_ENVIRONMENT_CONFIG_DEV)
    ),
    # Spelled out rather than merged onto the base: STANDALONE is fully local
    # (no lakehouse, no GitHub-backed space, fixed feature flags) and would
    # override nearly every base value, so a diff would obscure more than it saves.
    "STANDALONE": GBEnvConfig(
        env="STANDALONE",
        lakehouse_environment="",
        space_config_branch_name="main",
        # gbcli
        gbserver_host="http://localhost:8080",
        default_space="standalone",
        web_ui_url="http://localhost:8080/dashboard",
        config_spaces="",
        config_profile="",
        server_log_application_name="gbserver-standalone",
        branch_assets="",
        hf_organization="ibm-research",
        hf_enterprise_organizations=["ibm-research", "ibm-granite"],
        feature_flags={
            "build_start_via_github": False,
            # Deliberately off in standalone, unlike the deployed environments.
            "gbserver_artifact_filter": False,
            "gbserver_build_events": True,
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
