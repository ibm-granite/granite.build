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

"""Contants and env vars that are used by many other modules."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from gbserver.types.constants_base import (
    DEFAULT_GH_DOMAIN,
    ENV_VAR_IBMID_AUTHORIZE_URL,
    ENV_VAR_IBMID_CALLBACK_URL,
    ENV_VAR_IBMID_CLIENT_ID,
    ENV_VAR_IBMID_CLIENT_SECRET,
    ENV_VAR_IBMID_ISSUER,
    ENV_VAR_IBMID_JWKS_URI,
    ENV_VAR_IBMID_TOKEN_URL,
    ENV_VAR_IBMID_USERINFO_URL,
    ENV_VAR_PREFIX,
    getenv_boolean,
)
from gbserver.types.gbserverenvconfig import gb_environment_config

load_dotenv(override=False)

API_VERSION = "v1"
API_BASE_PATH = f"/api/{API_VERSION}"

BUILD_YAML_BASE_KEYS = ["llm.build", "granite.build"]
CURRENT_BUILD_YAML_VERSION_KEY = "version"
CURRENT_BUILD_YAML_VERSION = "0.0.1"
DEFAULT_REPO_DIR_TO_WATCH = "experiments"
FULL_CONFIG_RUN_METADATA_KEY = "run_metadata"
GBSERVER_SECRET_NAME_SEPARATOR = "___"
STEP_FILE_NAME = "step.yaml"

ENV_URI_SCHEME = "env"
FILE_SCHEME = "file"

CODE_GBSERVER_DIR = Path(__file__).parent.parent
CODE_GBSERVER_BUILTINS_DIR = CODE_GBSERVER_DIR / "builtins"
CODE_GBSERVER_BUILTINS_STEPS_DIR = CODE_GBSERVER_BUILTINS_DIR / "steps"
CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_DIR = CODE_GBSERVER_BUILTINS_STEPS_DIR / "gbstep"
CODE_GBSERVER_BUILTINS_STEPS_HFPULL_DIR = CODE_GBSERVER_BUILTINS_STEPS_DIR / "hfpull"
CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_DIR = CODE_GBSERVER_BUILTINS_STEPS_DIR / "hfpush"
CODE_GBSERVER_BUILTINS_STEPS_LHPULL_DIR = CODE_GBSERVER_BUILTINS_STEPS_DIR / "lhpull"
CODE_GBSERVER_BUILTINS_STEPS_LHPUSH_DIR = CODE_GBSERVER_BUILTINS_STEPS_DIR / "lhpush"
CODE_GBSERVER_BUILTINS_STEPS_COSRCLONE_DIR = (
    CODE_GBSERVER_BUILTINS_STEPS_DIR / "cosrclone"
)

CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI = f"{FILE_SCHEME}://" + str(
    CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_DIR
)
CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI = f"{FILE_SCHEME}://" + str(
    CODE_GBSERVER_BUILTINS_STEPS_HFPULL_DIR
)
CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI = f"{FILE_SCHEME}://" + str(
    CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_DIR
)
CODE_GBSERVER_BUILTINS_STEPS_LHPULL_URI = f"{FILE_SCHEME}://" + str(
    CODE_GBSERVER_BUILTINS_STEPS_LHPULL_DIR
)
CODE_GBSERVER_BUILTINS_STEPS_LHPUSH_URI = f"{FILE_SCHEME}://" + str(
    CODE_GBSERVER_BUILTINS_STEPS_LHPUSH_DIR
)
CODE_GBSERVER_BUILTINS_STEPS_COSRCLONE_URI = f"{FILE_SCHEME}://" + str(
    CODE_GBSERVER_BUILTINS_STEPS_COSRCLONE_DIR
)

# ---------------------------------------------------------
# Environment variables


ENV_VAR_TRUNCATE_LENGTH = ENV_VAR_PREFIX + "_TRUNCATE_LENGTH"
# The env var for admin table prefix, to cascade it to child processes (especiallly for rest-server multiworker)
# Once we migrate to env-based SQL schemas we won't need it.
ENV_VAR_GBSERVER_ADMIN_TABLE_PREFIX = ENV_VAR_PREFIX + "_ADMIN_TABLE_PREFIX"
ENV_VAR_IBM_SEC_MAN_ENDPOINT = ENV_VAR_PREFIX + "_IBM_SEC_MAN_ENDPOINT"
ENV_VAR_IBM_SEC_MAN_API_KEY = ENV_VAR_PREFIX + "_IBM_SEC_MAN_API_KEY"
ENV_VAR_DEFAULT_LOG_LEVEL = ENV_VAR_PREFIX + "_DEFAULT_LOG_LEVEL"
ENV_VAR_PYICEBERG_LOG_LEVEL = ENV_VAR_PREFIX + "_PYICEBERG_LOG_LEVEL"
ENV_VAR_DEFAULT_GITHUB_TOKEN = ENV_VAR_PREFIX + "_GITHUB_TOKEN"
ENV_VAR_DEBUG_MODE = ENV_VAR_PREFIX + "_DEBUG_MODE"
ENV_VAR_METADATA_STORAGE = ENV_VAR_PREFIX + "_METADATA_STORAGE"
ENV_VAR_AUTH_MODE = ENV_VAR_PREFIX + "_AUTH_MODE"
ENV_VAR_API_KEY = ENV_VAR_PREFIX + "_API_KEY"
ENV_VAR_API_USER = ENV_VAR_PREFIX + "_API_USER"
ENV_VAR_USE_LESS_COMPUTE_ON_DRY_RUN = ENV_VAR_PREFIX + "_USE_LESS_COMPUTE_ON_DRY_RUN"
# This is set in the buildwatcher pod so the BuildRunnerJob can be sure to run in the same namespace
ENV_VAR_BUILDRUNNERJOB_NAMESPACE = ENV_VAR_PREFIX + "_BUILDRUNNERJOB_NAMESPACE"
ENV_VAR_BUILDRUNNERJOB_IMAGE = ENV_VAR_PREFIX + "_BUILDRUNNERJOB_IMAGE_OVERRIDE"
ENV_VAR_BUILDRUNNERJOB_SECRET_NAME = ENV_VAR_PREFIX + "_BUILDRUNNERJOB_SECRET_NAME"
ENV_VAR_BUILDRUNNERJOB_BUILD_WORKSPACE_PVC_NAME = (
    ENV_VAR_PREFIX + "_BUILDRUNNERJOB_BUILD_WORKSPACE_PVC_NAME"
)
ENV_VAR_BUILDRUNNERJOB_CONFIGMAP_NAME = (
    ENV_VAR_PREFIX + "_BUILDRUNNERJOB_CONFIGMAP_NAME"
)
ENV_VAR_DEFAULT_BUILDRUNNER_TYPE = ENV_VAR_PREFIX + "_DEFAULT_BUILDRUNNER_TYPE"

ENV_VAR_GBSERVER_K8S_USE_ASPERA = ENV_VAR_PREFIX + "_K8S_USE_ASPERA"
ENV_VAR_GBSERVER_LSF_USE_ASPERA = ENV_VAR_PREFIX + "_LSF_USE_ASPERA"
ENV_VAR_GBSERVER_ENABLE_SSH_HOST_KEY_VERIFICATION = (
    ENV_VAR_PREFIX + "_ENABLE_SSH_HOST_KEY_VERIFICATION"
)
ENV_VAR_GBSERVER_ENABLE_STEP_RETRY = ENV_VAR_PREFIX + "_ENABLE_STEP_RETRY"
ENV_VAR_BUILDRUNNERJOB_SLEEP_ON_END = ENV_VAR_PREFIX + "_BUILDRUNNERJOB_SLEEP_ON_END"
ENV_VAR_BUILTIN_STEP_IMAGE = ENV_VAR_PREFIX + "_BUILTIN_STEP_IMAGE"

ENV_VAR_GBSERVER_SQL_SCHEME = ENV_VAR_PREFIX + "_SQL_SCHEME"  # postgresql, mysql, etc.
ENV_VAR_GBSERVER_SQL_DBNAME = ENV_VAR_PREFIX + "_SQL_DBNAME"
ENV_VAR_GBSERVER_SQL_SCHEMA = ENV_VAR_PREFIX + "_SQL_SCHEMA"
ENV_VAR_GBSERVER_SQL_HOST = ENV_VAR_PREFIX + "_SQL_HOST"
ENV_VAR_GBSERVER_SQL_PORT = ENV_VAR_PREFIX + "_SQL_PORT"
ENV_VAR_GBSERVER_SQL_USER = ENV_VAR_PREFIX + "_SQL_USER"
ENV_VAR_GBSERVER_SQL_PASSWD = ENV_VAR_PREFIX + "_SQL_PASSWD"
ENV_VAR_GBSERVER_SQL_SSLROOT_CERT = (
    ENV_VAR_PREFIX + "_SQL_SSLROOT_CERT"
)  # deprected in favor of _FILE
ENV_VAR_GBSERVER_SQL_SSLROOT_CERT_FILE = ENV_VAR_PREFIX + "_SQL_SSLROOT_CERT_FILE"
ENV_VAR_GBSERVER_SQL_SSLROOT_CERT_BASE64 = ENV_VAR_PREFIX + "_SQL_SSLROOT_CERT_BASE64"
ENV_VAR_GBSERVER_SQL_ECHO = ENV_VAR_PREFIX + "_SQL_ECHO"
ENV_VAR_SIDECAR_MONITORING_IMAGE_TAG = ENV_VAR_PREFIX + "_SIDECAR_MONITORING_IMAGE_TAG"
ENV_VAR_GBSERVER_IMAGE_TAG = ENV_VAR_PREFIX + "_IMAGE_TAG"
ENV_VAR_GBSERVER_METRICS_ENDPOINT = ENV_VAR_PREFIX + "_METRICS_ENDPOINT"
ENV_VAR_GBSERVER_METRICS_AUTH_TOKEN = ENV_VAR_PREFIX + "_METRICS_AUTH_TOKEN"
# Node Health Alerting
ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_WEBHOOK_URL = (
    ENV_VAR_PREFIX + "_NODE_HEALTH_ALERT_WEBHOOK_URL"
)
ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_WEBHOOK_URL = (
    ENV_VAR_PREFIX + "_NODE_HEALTH_ALERT_SLACK_WEBHOOK_URL"
)
ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_CHANNEL = (
    ENV_VAR_PREFIX + "_NODE_HEALTH_ALERT_SLACK_CHANNEL"
)
ENV_VAR_GBSERVER_NODE_HEALTH_ALERT_SLACK_MENTION_USERS = (
    ENV_VAR_PREFIX + "_NODE_HEALTH_ALERT_SLACK_MENTION_USERS"
)

ENV_VAR_LSF_LOGIN_NODE_ROTATION = ENV_VAR_PREFIX + "_LSF_LOGIN_NODE_ROTATION"

ENV_VAR_GBSERVER_DEFAULT_GH_REQUEST_TIMEOUT = (
    ENV_VAR_PREFIX + "_DEFAULT_GH_REQUEST_TIMEOUT"
)
ENV_VAR_GBSERVER_PUSH_METRICS_TIMEOUT = ENV_VAR_PREFIX + "_PUSH_METRICS_TIMEOUT"

ENV_VAR_GBSERVER_RAISE_BUILD_EXCEPTIONS = ENV_VAR_PREFIX + "_RAISE_BUILD_EXCEPTIONS"
GBSERVER_RAISE_BUILD_EXCEPTIONS = (
    os.getenv(ENV_VAR_GBSERVER_RAISE_BUILD_EXCEPTIONS, "false").lower() == "true"
)

# Hugging Face Hub Configuration
ENV_VAR_HF_TOKEN = ENV_VAR_PREFIX + "_HF_TOKEN"

# Hugging Face Hub default values from environment
GBSERVER_HF_TOKEN = os.getenv(ENV_VAR_HF_TOKEN, os.getenv("HF_TOKEN", None))

DEFAULT_GH_API_ENDPOINT = f"https://api.{DEFAULT_GH_DOMAIN}"
# NOTE: To do multiple dmf pushes with aspera, the aspera daemon needs to be kept running.
# This causes an issue where the LSF job doesn't end because the daemon is still running.
K8S_USE_ASPERA = os.getenv(ENV_VAR_GBSERVER_K8S_USE_ASPERA, "true").lower() == "true"
LSF_USE_ASPERA = os.getenv(ENV_VAR_GBSERVER_LSF_USE_ASPERA, "false").lower() == "true"
ENABLE_SSH_HOST_KEY_VERIFICATION = (
    os.getenv(ENV_VAR_GBSERVER_ENABLE_SSH_HOST_KEY_VERIFICATION, "false").lower()
    == "true"
)
GBSERVER_ENABLE_STEP_RETRY = (
    os.getenv(ENV_VAR_GBSERVER_ENABLE_STEP_RETRY, "true").lower() == "true"
)
# Metrics
# Endpoint to push metrics to http://gb-metrics-gb-metrics:8081/api/metrics
GBSERVER_METRICS_ENDPOINT = os.getenv(ENV_VAR_GBSERVER_METRICS_ENDPOINT, "")
GBSERVER_METRICS_AUTH_TOKEN = os.getenv(ENV_VAR_GBSERVER_METRICS_AUTH_TOKEN, "")
# Metrics
DEFAULT_LOG_LEVEL = os.getenv(ENV_VAR_DEFAULT_LOG_LEVEL, "info").lower()
PYICEBERG_LOG_LEVEL = os.getenv(ENV_VAR_PYICEBERG_LOG_LEVEL, "WARNING").upper()
GBSERVER_TRUNCATE_LENGTH = int(os.getenv(ENV_VAR_TRUNCATE_LENGTH, "-1"), base=10)
DEFAULT_GH_REQUEST_TIMEOUT = int(
    os.getenv(ENV_VAR_GBSERVER_DEFAULT_GH_REQUEST_TIMEOUT, "60"), base=10
)
PUSH_METRICS_TIMEOUT = int(
    os.getenv(ENV_VAR_GBSERVER_PUSH_METRICS_TIMEOUT, "10"), base=10
)
DEFAULT_WORKSPACE_DIR = os.getenv(
    ENV_VAR_PREFIX + "_DEFAULT_WORKSPACE_DIR", "gbserverworkspace"
)
"""deprecated in favor of  DEFAULT_ROOT_WORKSPACE_DIR"""
DEFAULT_ROOT_WORKSPACE_DIR = os.getenv(
    ENV_VAR_PREFIX + "_DEFAULT_ROOT_WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR
)
DEFAULT_ROOT_BUILDWATCHER_WORKSPACE_DIR = (
    DEFAULT_ROOT_WORKSPACE_DIR + "/gbserver-buildwatcher-workspace"
)
DEFAULT_ROOT_PRWATCHER_WORKSPACE_DIR = (
    DEFAULT_ROOT_WORKSPACE_DIR + "/gbserver-prwatcher-workspace"
)
GBSERVER_FUNCTIONAL_IDS = json.loads(
    os.getenv(
        ENV_VAR_PREFIX + "_FUNCTIONAL_IDS",
        '["Granite-Dot-Build-Test", "Granitebuild", "aibs"]',
    )
)
DEFAULT_BUILDWATCHER_COMMITTER_NAME = os.getenv(
    ENV_VAR_PREFIX + "_DEFAULT_BUILDWATCHER_COMMITTER_NAME", "Granitebuild"
)
DEFAULT_BUILDWATCHER_COMMITTER_EMAIL = os.getenv(
    ENV_VAR_PREFIX + "_DEFAULT_BUILDWATCHER_COMMITTER_EMAIL", "granitebuild@ibm.com"
)
MAX_PR_CREATION_TRIES = int(
    os.getenv(ENV_VAR_PREFIX + "_MAX_PR_CREATION_TRIES", "100"), base=10
)
FETCH_CLOUD_LOGS_MAX_RETRIES = int(
    os.getenv(ENV_VAR_PREFIX + "_FETCH_CLOUD_LOGS_MAX_RETRIES", "10"), base=10
)
FETCH_CLOUD_LOGS_RETRY_INTERVAL = int(
    os.getenv(ENV_VAR_PREFIX + "_FETCH_CLOUD_LOGS_RETRY_INTERVAL", "5"), base=10
)
FETCH_CLOUD_LOGS_MAX_PAGE_SIZE = int(
    os.getenv(ENV_VAR_PREFIX + "_FETCH_CLOUD_LOGS_MAX_PAGE_SIZE", "10000"), base=10
)
FETCH_CLOUD_LOGS_PR_MAX_CHARS = int(
    os.getenv(ENV_VAR_PREFIX + "_FETCH_CLOUD_LOGS_PR_MAX_CHARS", str(65536 // 2)),
    base=10,
)
FETCH_CLOUD_LOGS_TIME_RANGE = int(
    os.getenv(ENV_VAR_PREFIX + "_FETCH_CLOUD_LOGS_TIME_RANGE", str(5 * 24 * 3600)),
    base=10,
)  # last 5 days
LAKEHOUSE_MAX_RETRIES = int(
    os.getenv(ENV_VAR_PREFIX + "_LAKEHOUSE_MAX_RETRIES", "100"), base=10
)
LAKEHOUSE_BASE_DELAY = float(os.getenv(ENV_VAR_PREFIX + "_LAKEHOUSE_BASE_DELAY", "0.5"))
GIT_CLONE_MAX_RETRIES = int(
    os.getenv(ENV_VAR_PREFIX + "_GIT_CLONE_MAX_RETRIES", "5"), base=10
)
GIT_CLONE_RETRY_MIN_WAIT = float(
    os.getenv(ENV_VAR_PREFIX + "_GIT_CLONE_RETRY_MIN_WAIT", "1")
)
GIT_CLONE_RETRY_MAX_WAIT = float(
    os.getenv(ENV_VAR_PREFIX + "_GIT_CLONE_RETRY_MAX_WAIT", "30")
)
# GitHub API retry configuration
GITHUB_API_MAX_RETRIES = int(
    os.getenv(ENV_VAR_PREFIX + "_GITHUB_API_MAX_RETRIES", "10"), base=10
)
GITHUB_API_RETRY_BASE_DELAY = float(
    os.getenv(ENV_VAR_PREFIX + "_GITHUB_API_RETRY_BASE_DELAY", "1.0")
)
GITHUB_API_RETRY_MAX_DELAY = float(
    os.getenv(ENV_VAR_PREFIX + "_GITHUB_API_RETRY_MAX_DELAY", "60.0")
)
GBSERVER_GITHUB_TOKEN = os.getenv(
    ENV_VAR_DEFAULT_GITHUB_TOKEN, os.getenv("GITHUB_TOKEN", "")
)
GBSERVER_IBM_CLOUD_LOGS_API_KEY = os.getenv("IBM_CLOUD_LOGS_API_KEY", "")
GBSERVER_IBM_CLOUD_LOGS_API_URL = os.getenv("IBM_CLOUD_LOGS_API_URL", "")
GBSERVER_IBM_CLOUD_SERVER_LOGS_API_KEY = os.getenv("IBM_CLOUD_SERVER_LOGS_API_KEY", "")
GBSERVER_IBM_CLOUD_SERVER_LOGS_API_URL = os.getenv("IBM_CLOUD_SERVER_LOGS_API_URL", "")
GBSERVER_DEBUG_MODE = os.getenv(ENV_VAR_DEBUG_MODE, None)
GBSERVER_GIT_COMMIT = os.getenv(ENV_VAR_PREFIX + "_GIT_COMMIT", "")
# Standalone mode — must be evaluated before constants that read the env vars
# it sets (PROCEED_WITHOUT_SECRETS, METADATA_STORAGE, etc.). Uses setdefault()
# so explicit user overrides are preserved.
ENV_VAR_STANDALONE_MODE = ENV_VAR_PREFIX + "_STANDALONE_MODE"
GBSERVER_STANDALONE_MODE = getenv_boolean(ENV_VAR_STANDALONE_MODE, False)

if GBSERVER_STANDALONE_MODE:
    for _k, _v in {
        ENV_VAR_METADATA_STORAGE: "sqlite",
        ENV_VAR_DEFAULT_BUILDRUNNER_TYPE: "thread",
        ENV_VAR_PREFIX + "_PROCEED_WITHOUT_SECRETS": "true",
        "GB_ENVIRONMENT": "STANDALONE",
    }.items():
        os.environ.setdefault(_k, _v)

GBSERVER_PROCEED_WITHOUT_SECRETS = getenv_boolean(
    ENV_VAR_PREFIX + "_PROCEED_WITHOUT_SECRETS", False
)  # default False

# NATS JetStream configuration
ENV_VAR_NATS_URL = ENV_VAR_PREFIX + "_NATS_URL"
ENV_VAR_NATS_STREAM_MAX_AGE = ENV_VAR_PREFIX + "_NATS_STREAM_MAX_AGE"
ENV_VAR_NATS_MAX_DELIVER = ENV_VAR_PREFIX + "_NATS_MAX_DELIVER"
ENV_VAR_NATS_ACK_WAIT = ENV_VAR_PREFIX + "_NATS_ACK_WAIT"
ENV_VAR_NATS_EMBEDDED = ENV_VAR_PREFIX + "_NATS_EMBEDDED"

GBSERVER_NATS_URL = os.getenv(ENV_VAR_NATS_URL, "nats://localhost:4222")
GBSERVER_NATS_STREAM_MAX_AGE = int(os.getenv(ENV_VAR_NATS_STREAM_MAX_AGE, "604800"))
GBSERVER_NATS_MAX_DELIVER = int(os.getenv(ENV_VAR_NATS_MAX_DELIVER, "5"))
GBSERVER_NATS_ACK_WAIT = int(os.getenv(ENV_VAR_NATS_ACK_WAIT, "30"))
GBSERVER_NATS_EMBEDDED = getenv_boolean(ENV_VAR_NATS_EMBEDDED, True)

GBSERVER_REST_SERVER_WORKERS = int(
    os.getenv(ENV_VAR_PREFIX + "_REST_SERVER_WORKERS", "1"), base=10
)
GBSERVER_REST_SERVER_TIMEOUT_KEEP_ALIVE = int(
    os.getenv(ENV_VAR_PREFIX + "_REST_SERVER_TIMEOUT_KEEP_ALIVE", "120"), base=10
)
# Build Runner
BUILDRUNNERJOB_SLEEP_ON_END = (
    os.getenv(ENV_VAR_BUILDRUNNERJOB_SLEEP_ON_END, "false").lower() == "true"
)
BUILDRUNNERJOB_SECRET_NAME = os.getenv(
    ENV_VAR_BUILDRUNNERJOB_SECRET_NAME, "vela-414-granite-dot-build-svc-acc-secret2"
)
BUILDRUNNERJOB_BUILD_WORKSPACE_PVC_NAME = os.getenv(
    ENV_VAR_BUILDRUNNERJOB_BUILD_WORKSPACE_PVC_NAME, "gb-buildws-pvc"
)
BUILDRUNNERJOB_CONFIGMAP_NAME = os.getenv(
    ENV_VAR_BUILDRUNNERJOB_CONFIGMAP_NAME, "granite-dot-build-configmap"
)
# Environment LSF
# Maximum number of retries for transient LSF errors (e.g., "Cannot open your job file")
GBSERVER_LSF_TRANSIENT_ERROR_MAX_RETRIES = int(
    os.getenv(ENV_VAR_PREFIX + "_LSF_TRANSIENT_ERROR_MAX_RETRIES", "3"), base=10
)
# Delay between retries for transient LSF errors (in seconds)
GBSERVER_LSF_TRANSIENT_ERROR_RETRY_DELAY = int(
    os.getenv(ENV_VAR_PREFIX + "_LSF_TRANSIENT_ERROR_RETRY_DELAY", "30"), base=10
)
# Used by the build framework monitoring to allow the consumption of all the events
GBSERVER_MONITORING_GRACE_PERIOD = int(
    os.getenv(ENV_VAR_PREFIX + "_MONITORING_GRACE_PERIOD", "30"), base=10
)
USE_LESS_COMPUTE_ON_DRY_RUN = (
    os.getenv(ENV_VAR_USE_LESS_COMPUTE_ON_DRY_RUN, "True").lower() == "true"
)
GBSERVER_GBSERVER_IMAGE_TAG = os.getenv(ENV_VAR_GBSERVER_IMAGE_TAG, None)
GBSERVER_SIDECAR_MONITORING_IMAGE_TAG = os.getenv(
    ENV_VAR_SIDECAR_MONITORING_IMAGE_TAG, "latest"
)
GBSERVER_DEFAULT_BUILDRUNNER_TYPE = os.getenv(
    ENV_VAR_DEFAULT_BUILDRUNNER_TYPE, "job"
)  # One of job, process or thread.
GB_ENVIRONMENT_FROM_ENV = os.getenv("GB_ENVIRONMENT", "").upper()
GB_ENVIRONMENT_CONFIG = gb_environment_config(GB_ENVIRONMENT_FROM_ENV)
GB_ENVIRONMENT = GB_ENVIRONMENT_CONFIG.env
_default_gbserver_image_tag = (
    GBSERVER_GBSERVER_IMAGE_TAG if GBSERVER_GBSERVER_IMAGE_TAG else "latest"
)
GBSERVER_IMAGE = f"us.icr.io/cil15-shared-registry/gb-{GB_ENVIRONMENT.lower()}/gbserver:{_default_gbserver_image_tag}"
# Override the image for the BuildRunnerProcess to use when running BuildRunner CLI.
# If not set, then the image from the buildwatcher deployment yaml is used
BUILDRUNNERJOB_IMAGE_OVERRIDE = os.getenv(ENV_VAR_BUILDRUNNERJOB_IMAGE, GBSERVER_IMAGE)
GBSERVER_BUILTIN_STEP_IMAGE = os.getenv(ENV_VAR_BUILTIN_STEP_IMAGE, GBSERVER_IMAGE)

BUILDRUNNERJOB_NAMESPACE = os.getenv(
    ENV_VAR_BUILDRUNNERJOB_NAMESPACE, GB_ENVIRONMENT_CONFIG.default_pod_namespace
)
LAKEHOUSE_ENVIRONMENT = os.getenv(
    "LAKEHOUSE_ENVIRONMENT", GB_ENVIRONMENT_CONFIG.lakehouse_environment
)

# SQL ------------------------------
GB_METADATA_STORAGE = os.getenv(ENV_VAR_METADATA_STORAGE, "sql").lower()

# Auth
GBSERVER_AUTH_MODE = os.getenv(ENV_VAR_AUTH_MODE, "github")
GBSERVER_API_KEY = os.getenv(ENV_VAR_API_KEY, "")
GBSERVER_API_USER = os.getenv(ENV_VAR_API_USER, "standalone")

# IBMid OIDC
GBSERVER_IBMID_ISSUER = os.getenv(
    ENV_VAR_IBMID_ISSUER, "https://login.ibm.com/oidc/endpoint/default"
)
GBSERVER_IBMID_JWKS_URI = os.getenv(
    ENV_VAR_IBMID_JWKS_URI, "https://login.ibm.com/oidc/endpoint/default/jwks"
)
GBSERVER_IBMID_CLIENT_ID = os.getenv(ENV_VAR_IBMID_CLIENT_ID, "")
GBSERVER_IBMID_CLIENT_SECRET = os.getenv(ENV_VAR_IBMID_CLIENT_SECRET, "")
GBSERVER_IBMID_AUTHORIZE_URL = os.getenv(
    ENV_VAR_IBMID_AUTHORIZE_URL,
    "https://login.ibm.com/v1.0/endpoint/default/authorize",
)
GBSERVER_IBMID_TOKEN_URL = os.getenv(
    ENV_VAR_IBMID_TOKEN_URL,
    "https://login.ibm.com/v1.0/endpoint/default/token",
)
GBSERVER_IBMID_USERINFO_URL = os.getenv(
    ENV_VAR_IBMID_USERINFO_URL,
    "https://login.ibm.com/v1.0/endpoint/default/userinfo",
)
GBSERVER_IBMID_CALLBACK_URL = os.getenv(ENV_VAR_IBMID_CALLBACK_URL, "")

# OpenLineage / WandB lineage provider
GBSERVER_LINEAGE_PROVIDER = os.getenv(ENV_VAR_PREFIX + "_LINEAGE_PROVIDER", "wandb")
GBSERVER_WANDB_API_KEY = os.getenv(ENV_VAR_PREFIX + "_WANDB_API_KEY", "")
GBSERVER_WANDB_PROJECT = os.getenv(
    ENV_VAR_PREFIX + "_WANDB_PROJECT", "lineage-tracking"
)
GBSERVER_WANDB_ENTITY = os.getenv(ENV_VAR_PREFIX + "_WANDB_ENTITY", "dmf-testing")
GBSERVER_WANDB_BASE_URL = os.getenv(
    ENV_VAR_PREFIX + "_WANDB_BASE_URL", "https://ibm.wandb.io"
)

GBSERVER_SQL_SCHEME = os.getenv(ENV_VAR_GBSERVER_SQL_SCHEME, "postgresql")
GBSERVER_SQL_HOST = os.getenv(
    ENV_VAR_GBSERVER_SQL_HOST,
    "05ed7d0c-3027-412e-bc75-23351a34b8fa.blrrvkdw0thh68l98t20.databases.appdomain.cloud",
)
GBSERVER_SQL_PORT = os.getenv(ENV_VAR_GBSERVER_SQL_PORT, "31842")
GBSERVER_SQL_DBNAME = os.getenv(ENV_VAR_GBSERVER_SQL_DBNAME, "ibmclouddb")
GBSERVER_SQL_SCHEMA = os.getenv(
    ENV_VAR_GBSERVER_SQL_SCHEMA, GB_ENVIRONMENT_CONFIG.default_sql_schema
)
GBSERVER_SQL_USER = os.getenv(
    ENV_VAR_GBSERVER_SQL_USER, "ibm_cloud_60dd7591_25f4_48a5_840d_4239660d304c"
)
GBSERVER_SQL_PASSWD = os.getenv(ENV_VAR_GBSERVER_SQL_PASSWD, "")
GBSERVER_SQL_SSLROOT_CERT_FILE = os.getenv(
    ENV_VAR_GBSERVER_SQL_SSLROOT_CERT_FILE,
    os.getenv(ENV_VAR_GBSERVER_SQL_SSLROOT_CERT, None),
)
# A base64 encoding of an ssl cert file.
GBSERVER_SQL_SSLROOT_CERT_BASE64 = os.getenv(
    ENV_VAR_GBSERVER_SQL_SSLROOT_CERT_BASE64, None
)
GBSERVER_SQL_ECHO = getenv_boolean(ENV_VAR_GBSERVER_SQL_ECHO, False)  # default False
GBSERVER_SECRET_GROUP_FOR_USERS = "_gbuser-" + GB_ENVIRONMENT

# -------------------------------------------------


HELP_INSTRUCTIONS_FOR_BUILD = """We are going to launch a build with the id `{build_id}`

Once the build is running, you can use the `gb` CLI to get more information.

To get the build status:

```shell
llmb build status {build_id}
```

To get all of the lines from the logs:

```shell
llmb build log --all {build_id}
```

To only get the last 10k lines of the logs:

```shell
llmb build log --tail 10000 {build_id}
```

By default this gives you the logs from all the steps in the build.

To only get the logs of a particular step you can use:

```shell
llmb build log --all {build_id} --build-step-id <step id>
```

If you have admin access, you can access the build-runner logs as well:

```shell
llmb admin log gbserver-build-runner --all --build-id {build_id}
```
"""

LINEAGE_LINK_MESSAGE_FOR_BUILD = """
Use the following link to see the lineage/status of the build and its artifacts:

{build_status_link}
"""

DASHBOARD_LINK_MESSAGE_FOR_BUILD = """
Dashboard: {dashboard_link}
"""

STARTING_BUILD_MESSAGE = """
Build is starting.  Use the following link to see the lineage/status of the build and its artifacts:

{build_status_link}
"""


def is_debug_mode() -> bool:
    """Returns True if debug mode is enabled."""
    return GBSERVER_DEBUG_MODE is not None


PR_TITLE_DRYRUN = "dryrun"
PR_TITLE_IGNORE = "ignore"
WORKSPACE_REPOS_DIR = "repos"
WORKSPACE_PRS_DIR = "pullrequests"
WORKSPACE_ZIPS_DIR = "zips"
WORKSPACE_BUILDS_DIR = "builds"

CONTEXT_SETTINGS = {"auto_envvar_prefix": ENV_VAR_PREFIX}
DEFAULT_DIR_PERMS = 0o775
DEFAULT_LOG_FORMAT = (
    "[%(asctime)s %(levelname)-5s]"
    + "[%(filename)20s:%(lineno)3s %(funcName)25s()] %(message)s"
)
# Admin storage-related constants
GRANITE_DOT_BUILD_PARENT_NAMESPACE = "granite_dot_build"

GRANITE_DOT_BUILD_ADMIN_NAMESPACE = f"{GRANITE_DOT_BUILD_PARENT_NAMESPACE}.admin"
GB_SPACES_TABLE_NAME = "gb_spaces"
GB_BUILDS_TABLE_NAME = "gb_builds"
GB_EVENTS_TABLE_NAME = "gb_events"
GB_STEP_RUNS_TABLE_NAME = "gb_steps"
GB_ARTIFACT_REGISTRY_TABLE_NAME = "gb_artifacts"
GB_TARGET_RUNS_TABLE_NAME = "gb_targets"
GB_NODE_FAILURES_TABLE_NAME = "gb_ndfail"
GB_SPACE_USERS_TABLE_NAME = "gb_space_users"

GB_JOB_STATS_DETAIL_CATEGORY = "granite-dot-build"
GB_JOB_STATS_DETAIL_TYPE = "granite-dot-build"
GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_TYPE = "registration"
GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_JOB_NAME = "register"

# Artifact storage-related constants
GB_PUBLIC_ARTIFACT_NAMESPACE = f"{GRANITE_DOT_BUILD_PARENT_NAMESPACE}.public"


COMMAND_RUN_BUILD_WATCH_BUILD_NAME = "build-for-a-local-dir"

PUBLIC_SPACE_NAME = "public"
SPACE_REPO_CONFIG_BRANCH_NAME = GB_ENVIRONMENT_CONFIG.space_config_branch_name
SPACE_REPO_BUILD_BRANCH_NAME = "main"  # tentative- may move to a different branch name
PUBLIC_SPACE_GIT_URI = GB_ENVIRONMENT_CONFIG.public_space_git_uri
PUBLIC_SPACE_LH_NAMESPACE = f"{GRANITE_DOT_BUILD_PARENT_NAMESPACE}.{GB_ENVIRONMENT_CONFIG.public_space_lh_subnamespace}"

# Leaving the below contants for now as they seem to be used by tests
PUBLIC_STAGING_SPACE_GIT_URI = gb_environment_config("STAGING").public_space_git_uri
PUBLIC_PROD_SPACE_GIT_URI = gb_environment_config("PROD").public_space_git_uri


def truncate(s: str, l: int = GBSERVER_TRUNCATE_LENGTH) -> str:
    """Truncate the string to the given length"""
    if l < 0 or len(s) <= l:
        return s
    return s[:l] + "..."


# Tags that begin with this are only editable via the super admin
SYSTEM_TAG_PREFIX = "sys-"
