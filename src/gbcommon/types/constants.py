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

import os
from typing import Optional

API_VERSION = "v1"
API_BASE_PATH = f"/api/{API_VERSION}"

ENV_VAR_PREFIX = "GB"
ENV_VAR_PREFIX_SERVER = "GBSERVER"
ENV_VAR_GH_DOMAIN = ENV_VAR_PREFIX + "_GH_DOMAIN"
DEFAULT_GH_DOMAIN = os.getenv(ENV_VAR_GH_DOMAIN, "github.ibm.com")
DEFAULT_WORKSPACE_DIR = os.environ.get(
    "GBSERVER_DEFAULT_WORKSPACE_DIR", "gbserverworkspace"
)
DEFAULT_TEST_SPACE_NAME = "default-test-space"  #
PUBLIC_SPACE_NAME = "default-space"
DEFAULT_REPO_DIR_TO_WATCH = "experiments"
CURRENT_BUILD_YAML_KEY = "granite.build"
BUILD_YAML_BASE_KEYS = ["llm.build", "granite.build"]
CURRENT_BUILD_YAML_VERSION_KEY = "version"
CURRENT_BUILD_YAML_VERSION = "0.0.1"
ENV_VAR_IBM_SEC_MAN_ENDPOINT = ENV_VAR_PREFIX_SERVER + "_IBM_SEC_MAN_ENDPOINT"
ENV_VAR_IBM_SEC_MAN_API_KEY = ENV_VAR_PREFIX_SERVER + "_IBM_SEC_MAN_API_KEY"
ENV_VAR_DEFAULT_LOG_LEVEL = ENV_VAR_PREFIX_SERVER + "_DEFAULT_LOG_LEVEL"
PR_TITLE_DRYRUN = "dryrun"
PR_TITLE_IGNORE = "ignore"
WORKSPACE_REPOS_DIR = "repos"
WORKSPACE_ZIPS_DIR = "zips"
WORKSPACE_BUILDS_DIR = "builds"

CONTEXT_SETTINGS = dict(auto_envvar_prefix=ENV_VAR_PREFIX_SERVER)
DEFAULT_DIR_PERMS = 0o775
DEFAULT_LOG_LEVEL = os.environ.get(ENV_VAR_DEFAULT_LOG_LEVEL, "INFO")
DEFAULT_LOG_FORMAT = (
    "[%(asctime)s %(levelname)-5s]"
    + "[%(filename)20s:%(lineno)3s %(funcName)25s()] %(message)s"
)
# Admin storage-related constants
GRANITE_DOT_BUILD_ADMIN_NAMESPACE = "granite_dot_build.admin"
GB_SPACES_TABLE_NAME = "gb_spaces"
GB_BUILDS_TABLE_NAME = "gb_builds"
GB_STEP_RUNS_TABLE_NAME = "gb_steps"
GB_ARTIFACT_REGISTRY_TABLE_NAME = "gb_artifacts"
GB_TARGET_RUNS_TABLE_NAME = "gb_targets"
GB_JOB_STATS_DETAIL_CATEGORY = "granite-dot-build"
GB_JOB_STATS_DETAIL_TYPE = "granite-dot-build"

# Artifact storage-related constants
GB_DEFAULT_LH_ARTIFACT_HOST = (
    "lake-staging.17qyd1z8hik0.us-east.codeengine.appdomain.cloud"
)


# gbserver
GRANITE_DOT_BUILD_PARENT_NAMESPACE = "granite_dot_build"
GB_PUBLIC_ARTIFACT_NAMESPACE = f"{GRANITE_DOT_BUILD_PARENT_NAMESPACE}.public"

# Used during testing to direct content to/from  staging or dev, when GB_ENVIRONMENT=STANDALONE
# Initially done for HF pushes.
ENV_VAR_GB_TEST_STANDALONE_ENVIRONMENT = ENV_VAR_PREFIX + "TEST_STANDALONE_ENVIRONMENT"
GB_TEST_STANDALONE_ENVIRONMENT = os.getenv(ENV_VAR_GB_TEST_STANDALONE_ENVIRONMENT, "STAGING")
# ---------------------------------------------------------------------------
# GitHub domain helpers
# ---------------------------------------------------------------------------


def is_public_github(domain: Optional[str] = None) -> bool:
    """Return True if the domain is public github.com."""
    if domain is None:
        domain = DEFAULT_GH_DOMAIN
    return domain.lower() == "github.com"


def get_gh_api_base(domain: Optional[str] = None) -> str:
    """Return the GitHub REST API base URL.

    Enterprise: https://{domain}/api/v3
    Public:     https://api.github.com
    """
    if domain is None:
        domain = DEFAULT_GH_DOMAIN
    if is_public_github(domain):
        return "https://api.github.com"
    return f"https://{domain}/api/v3"


def get_gh_web_base(domain: Optional[str] = None) -> str:
    """Return the GitHub web (HTML) base URL."""
    if domain is None:
        domain = DEFAULT_GH_DOMAIN
    return f"https://{domain}"


def get_gh_credentials_section(domain: Optional[str] = None) -> str:
    """Return the TOML credentials section name for the given domain."""
    if domain is None:
        domain = DEFAULT_GH_DOMAIN
    if is_public_github(domain):
        return "user.github_com"
    return "user.github"
