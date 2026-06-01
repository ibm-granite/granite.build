"""
All of the common constants.
"""

import logging
import os
import sys
from sys import platform

from gbcommon.types.constants import DEFAULT_GH_DOMAIN, is_public_github
from gbcommon.types.gbenvconfig import (
    gb_env_normalize,
    gb_environment_config,
    getenv_boolean,
)

logger = logging.getLogger(__name__)

PROJECT_NAME = "LLM.build"


# environments
GB_ENVIRONMENT_DEFAULT = "PROD"


def gb_env_formating(value: str, type: str) -> str:
    """Normalize env name with sys.exit on invalid input (CLI behavior)."""
    try:
        return gb_env_normalize(value, type)
    except ValueError as e:
        sys.exit(str(e))


def gb_environment() -> str:
    GB_ENVIRONMENT = gb_env_formating(
        os.environ.get("GB_ENVIRONMENT"), "Environment variable GB_ENVIRONMENT"
    )

    if GB_ENVIRONMENT is None:
        GB_ENVIRONMENT = GB_ENVIRONMENT_DEFAULT

    return GB_ENVIRONMENT


def hf_token() -> str:
    HF_TOKEN = os.environ.get("HF_TOKEN")

    return HF_TOKEN


# HuggingFace defaults
HF_ORGANIZATION_DEFAULT = gb_environment_config().hf_organization or "ibm-research"

# gbcli
_GBCLI_REPO_ORG = "ibm-granite" if is_public_github() else "granite-dot-build"
GBCLI_REPO_URL = os.environ.get(
    "GBCLI_REPO_URL",
    f"https://{DEFAULT_GH_DOMAIN}/{_GBCLI_REPO_ORG}/granite.build",
)

# assets
ASSETS_REPO_ORG = os.environ.get("GBCLI_ASSETS_REPO_ORG", "granite-dot-build")
ASSETS_REPO_NAME = os.environ.get("GBCLI_ASSETS_REPO_NAME", "assets")
ASSETS_REPO_URL = os.environ.get(
    "GBCLI_ASSETS_REPO_URL",
    f"https://{DEFAULT_GH_DOMAIN}/{ASSETS_REPO_ORG}/{ASSETS_REPO_NAME}.git",
)

# templates
TEMPLATES_REPO_BRANCH = "main"
TEMPLATES_REPO_FOLDER = "templates"

# steps
STEPS_REPO_BRANCH = "main"
STEPS_REPO_FOLDER = "steps"

# build
BUILD_FILENAME = "build.yaml"
CURRENT_BUILD_YAML_KEY = "granite.build"
CURRENT_BUILD_YAML_VERSION_KEY = "version"
CURRENT_BUILD_YAML_VERSION = "0.0.1"
BUILD_PARAMETERS_FILE = "parameters.yaml"
BUILD_PARAMETERS_APPLIED_FILE = "parameters-applied.yaml"
BUILD_RUN_FILE = "run.yaml"
BUILD_RUN_YAML_KEY = "targets_to_run"
BUILD_START_IGNORE = [
    ".gbconfig",
    ".gitignore",
    ".git",
    ".DS_Store",
    "*~",
    BUILD_RUN_FILE,
]
BUILD_LOGALL_PAGE_SIZE = 1000
BUILD_LOG_FOLLOW_SLEEP_TIME = 20
BUILD_LOG_SECONDS_IN_A_DAY = 86400
BUILD_LOG_DEFAULT_QUERY_RANGE = 7
BUILD_LOG_MAX_QUERY_RANGE = 8
BUILD_LOG_MAX_LOG_LIFESPAN = 14

# spaces
SPACE_REPO_ORG = "granite-dot-build"
SPACE_REPO_NAME = "gbspace-public"
SPACE_REPO_BUILD_FOLDER = "experiments"
SPACE_DEFAULT_NAME = "public"

# Models/RITS
RITS_URL = "https://rits.fmaas.res.ibm.com/"
RITS_BASE_URL = (
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/"
)
RITS_LIST_URL = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
RITS_MAX_TOKENS = 512
RITS_TEMP = 1.0
RITS_TOP_P = 0.7

# Lakehouse
LAKEHOUSE_ENVIRONMENT = gb_environment_config().lakehouse_environment
LAKEHOUSE_NAMESPACE = "granite_dot_build.public"
LAKEHOUSE_MODEL_SHARED_TABLE = "model_shared"
LAKEHOUSE_MODEL_TABLE = "model"
LAKEHOUSE_FILESET_SHARED_TABLE_NAME = "fileset_shared"
LAKEHOUSE_FILESET_TABLE_NAME = "fileset"
REVISION_DEFAULT = "granite-dot-build"
HF_REVISION_DEFAULT = "main"
GB_DMF_USE_CLASSIC_LOADER = getenv_boolean(
    "GB_DMF_USE_CLASSIC_LOADER", False
)  # True to turn off aspera use
GB_DMF_LOADER_BATCH_SIZE = 50000000
GB_DMF_LOADER_SIZE_LIMIT = 1073741824  # 1GB limit

# GBServer
GBSERVER_INSTANCE = os.environ.get(
    "GBSERVER_HOST", gb_environment_config().gbserver_host
)
GBSERVER_BUILD_API = f"{GBSERVER_INSTANCE}/api/v1/builds/"
GBSERVER_ARTIFACT_API = f"{GBSERVER_INSTANCE}/api/v1/artifacts/"
GBSERVER_SECRETS_API = f"{GBSERVER_INSTANCE}/api/v1/secrets/"
GBSERVER_SPACES_API = f"{GBSERVER_INSTANCE}/api/v1/spaces/"
GBSERVER_LINEAGE_API = f"{GBSERVER_INSTANCE}/api/v1/lineage/"

GBSERVER_LOGS_API = f"{GBSERVER_INSTANCE}/api/v1/logs/"

# Checksum
DEFAULT_CHECKSUM_CONCURRENCY = 8


# Error message
USER_NOT_LOGGED_IN_ERROR_MESSAGE = (
    "Error: User not logged in. Obtain a new token with 'auth login'"
)

VPN_CONNECTION_ERROR_MESSAGE = "Make sure you are connected to the VPN and try again."
VPN_TUNNELALL_CONNECTION_ERROR_MESSAGE = (
    "Make sure you are connected to the TUNNELALL VPN and try again."
)

ORIGIN_CERTIFY_MESSAGE = f"""🚨 New Requirement: To track artifacts from models with restricted use, you must provide one of the following when pushing an artifact.
1. If the artifact was created from existing {PROJECT_NAME} artifacts: Use --origin or --origin-list
2. If the artifact was created using a model under restricted use: Use --origin or --origin-list
3. If the artifact was created using a model under non-restricted use: Use --certify-no-restrictions (this will log your certification)
View the list of both unrestricted use and restricted-use models in the project documentation.
For any help on how to use each option in more details see `llmb artifact push --help`. For more information, ask a question in the `#llm-dot-build-users` channel."""


# DMF
DMF_URL = gb_environment_config().dmf_ui

# for tabulate
ARTIFACT_LINEAGE_DEFAULT_HEADERS = [
    "ARTIFACT_ID",
    "JOB_NAME",
    "JOB_ID",
    "JOB_STATUS",
    "SOURCE",
    "TARGET",
]

ARTIFACT_LINEAGE_FULL_HEADERS = [
    "ARTIFACT_ID",
    "CATEGORY",
    "JOB_NAME",
    "JOB_ID",
    "JOB_TYPE",
    "JOB_STARTED_AT",
    "JOB_COMPLETED_AT",
    "JOB_STATUS",
    "OWNER",
    "SOURCE",
    "SOURCE_FILTER",
    "SOURCE_TYPE",
    "SOURCE_OBJECT",
    "TARGET",
    "TARGET_FILTER",
    "TARGET_TYPE",
    "TARGET_OBJECT",
    "SOURCE_CODE_DETAILS",
    "JOB_INPUT_PARAMS",
    "EXECUTION_STATS",
    "JOB_OUTPUT_STATS",
]

ARTIFACT_LIST_HEADERS = [
    "UUID",
    "NAME",
    "URI",
    "TYPE",
    "TAGS",
    "STATUS",
    "CREATED_BY_BUILD_ID",
    "USERNAME",
    "CREATED_AT",
]

BUILD_LINEAGE_DEFAULT_HEADERS = [
    "BUILD_ID",
    "JOB_NAME",
    "JOB_ID",
    "JOB_STATUS",
    "SOURCE",
    "TARGET",
]

BUILD_LINEAGE_FULL_HEADERS = [
    "BUILD_ID",
    "CATEGORY",
    "JOB_NAME",
    "JOB_ID",
    "JOB_TYPE",
    "JOB_STARTED_AT",
    "JOB_COMPLETED_AT",
    "JOB_STATUS",
    "OWNER",
    "SOURCE",
    "SOURCE_TYPE",
    "SOURCE_OBJECT",
    "TARGET",
    "TARGET_TYPE",
    "TARGET_OBJECT",
    "SOURCE_CODE_DETAILS",
    "JOB_INPUT_PARAMS",
    "EXECUTION_STATS",
    "JOB_OUTPUT_STATS",
]

BUILD_LIST_HEADERS = ["BUILD_ID", "NAME", "USER", "TAGS", "STATUS", "START_TIME"]

BUILD_STATUS_ARTIFACTS_HEADERS = ["ARTIFACT_ID", "URI"]
BUILD_STATUS_STEPS_HEADERS = ["STEP_ID", "NAME", "STATUS", "URI"]
BUILD_STATUS_HISTORY_HEADERS = ["TIME", "DESCRIPTION"]

SPACE_LIST_HEADERS = ["NAME", "GIT REPO URI", "LAKEHOUSE NAMESPACE", "ROLE"]

STEP_LIST_HEADERS = ["STEP NAME", "DESCRIPTION", "URI"]
STEP_DESCRIBE_HEADERS = ["CONFIG", "PROPERTIES"]
STEP_FILENAME = "step.yaml"
STEP_README_FILENAME = "README.md"

TEMPLATE_LIST_HEADERS = ["TEMPLATE NAME", "DESCRIPTION"]

BUILD_DESCRIBE_ARTIFACTS_HEADERS = ["NAME", "URI"]
BUILD_DESCRIBE_STEPS_HEADERS = ["URI", "CONFIG"]

MODEL_LIST_HEADERS = ["MODEL", "FULL MODEL ID"]
MODEL_LIST_URI_HEADERS = ["MODEL", "RITS BASE URI"]

SECRET_SPACE_ADMIN_ERROR = "Only space admin can perform this operation. Run 'llmb space list --all --refresh' if the space role was recently updated."

# time delta for comparison against saved timestamp
SPACE_TIMESTAMP_DELTA_HOURS = 2


def to_int(value: str, type: str) -> str:
    if value:
        try:
            return int(value)
        except ValueError as e:
            logger.error(f"{type} does not have a valid value.")

    return None


# git
HTTP_POST_BUFFER = to_int(
    os.environ.get("GB_GIT_HTTP_POST_BUFFER"),
    "Environment variable GB_GIT_HTTP_POST_BUFFER",
)

# character compatibility
CLIPBOARD_CHAR = "📋 " if platform == "darwin" else ""
