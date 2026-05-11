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
Static constants for gbserver tests.

This module contains only static constants that do not depend on environment
variables being set at import time. This allows safe importing during pytest
collection before pytest_sessionstart sets up the test environment.
"""

import os

import pytest

from gbserver.types.constants import GB_ENVIRONMENT_CONFIG

TEST_ENV_VAR_PREFIX = "GBTEST_"

# Environment variable names for GB cluster configuration
ENV_VAR_GBTEST_GB_CLUSTER_SERVER_URI = TEST_ENV_VAR_PREFIX + "GB_CLUSTER_SERVER_URI"
ENV_VAR_GBTEST_GB_CLUSTER_API_KEY = TEST_ENV_VAR_PREFIX + "GB_CLUSTER_API_KEY"
ENV_VAR_GBTEST_GB_CLUSTER_TOKEN = TEST_ENV_VAR_PREFIX + "GB_CLUSTER_TOKEN"

# Environment variable name for build teardown skip
ENV_VAR_GBTEST_SKIP_BUILD_TEARDOWN = TEST_ENV_VAR_PREFIX + "SKIP_BUILD_TEARDOWN"

# Environment variable names for compute cluster configuration
ENV_VAR_GBTEST_COMPUTE_CLUSTER_SERVER_URI = (
    TEST_ENV_VAR_PREFIX + "COMPUTE_CLUSTER_SERVER_URI"
)
ENV_VAR_GBTEST_COMPUTE_CLUSTER_PROJECT = TEST_ENV_VAR_PREFIX + "COMPUTE_CLUSTER_PROJECT"
ENV_VAR_GBTEST_COMPUTE_CLUSTER_API_KEY = TEST_ENV_VAR_PREFIX + "COMPUTE_CLUSTER_API_KEY"
ENV_VAR_GBTEST_COMPUTE_CLUSTER_TOKEN = TEST_ENV_VAR_PREFIX + "COMPUTE_CLUSTER_TOKEN"
ENV_VAR_GBTEST_ADMIN_GITHUB_TOKEN = TEST_ENV_VAR_PREFIX + "ADMIN_GITHUB_TOKEN"
ENV_VAR_GBTEST_NON_ADMIN_GITHUB_TOKEN = TEST_ENV_VAR_PREFIX + "NON_ADMIN_GITHUB_TOKEN"

# Environment variable name for job termination timeout
ENV_VAR_GBTEST_JOB_TERMINATION_TIMEOUT_SECONDS = (
    TEST_ENV_VAR_PREFIX + "JOB_TERMINATION_TIMEOUT_SECONDS"
)

# Static test configuration
GBTEST_SPACE_NAME = "public"
GBTEST_USER_NAME = "Granite-Dot-Build-Test"

# GB cluster configuration and settings (e.g., RIS3) - where gb servers and buildrunnerjob pod are run
GBTEST_GB_CLUSTER_SERVER_URI = os.getenv(
    ENV_VAR_GBTEST_GB_CLUSTER_SERVER_URI,
    "https://c111-e.us-east.containers.cloud.ibm.com:30767",
)
GBTEST_GB_CLUSTER_PROJECT = GB_ENVIRONMENT_CONFIG.default_pod_namespace
GBTEST_GB_CLUSTER_API_KEY = os.getenv(ENV_VAR_GBTEST_GB_CLUSTER_API_KEY)
GBTEST_GB_CLUSTER_TOKEN = os.getenv(ENV_VAR_GBTEST_GB_CLUSTER_TOKEN)

GBTEST_SKIP_BUILD_TEARDOWN = (
    os.getenv(ENV_VAR_GBTEST_SKIP_BUILD_TEARDOWN, "false").lower() == "true"
)

# Compute cluster configuration and settings (e.g., Openshift) - where build steps are run
GBTEST_COMPUTE_CLUSTER_SERVER_URI = os.getenv(
    ENV_VAR_GBTEST_COMPUTE_CLUSTER_SERVER_URI, "https://api.dmf.dipc.res.ibm.com:6443"
)
GBTEST_COMPUTE_CLUSTER_PROJECT = os.getenv(
    ENV_VAR_GBTEST_COMPUTE_CLUSTER_PROJECT, "granite-build"
)
GBTEST_COMPUTE_CLUSTER_API_KEY = os.getenv(ENV_VAR_GBTEST_COMPUTE_CLUSTER_API_KEY)
GBTEST_COMPUTE_CLUSTER_TOKEN = os.getenv(ENV_VAR_GBTEST_COMPUTE_CLUSTER_TOKEN)
GBTEST_ADMIN_GITHUB_TOKEN = os.getenv(ENV_VAR_GBTEST_ADMIN_GITHUB_TOKEN)
GBTEST_NON_ADMIN_GITHUB_TOKEN = os.getenv(ENV_VAR_GBTEST_NON_ADMIN_GITHUB_TOKEN)

GBTEST_JOB_TERMINATION_TIMEOUT_SECONDS = int(
    os.getenv(ENV_VAR_GBTEST_JOB_TERMINATION_TIMEOUT_SECONDS, "300"), base=10
)

# Regex pattern to extract build ID from assertion messages formatted by failed_build_assert_message
BUILD_ID_PATTERN = r"\[Build: ([^\]]+)\]"

ENV_VAR_GBTEST_ENABLE_EXTENDED_TESTS = TEST_ENV_VAR_PREFIX + "ENABLE_EXTENDED_TESTS"

# Decorator that skips a test unless GBTEST_ENABLE_EXTENDED_TESTS=true.
# Apply to any test class or method that should only run in the extended CI suite.
#
# Usage:
#   @extended_testing_only
#   class TestMySlowTest(unittest.TestCase): ...
#
#   @extended_testing_only
#   def test_something_slow(self): ...

is_extended_testing_enabled = os.environ.get(ENV_VAR_GBTEST_ENABLE_EXTENDED_TESTS, "false").lower() == "true"


def extended_testing_only(func):
    """Mark a test to only run in the extended (merge) CI suite."""
    func = pytest.mark.extended_testing_only(func)
    func = pytest.mark.skipif(
        not is_extended_testing_enabled,
        reason=f"{ENV_VAR_GBTEST_ENABLE_EXTENDED_TESTS} is set to false",
    )(func)
    return func


def failed_build_assert_message(build_id: str, message: str) -> str:
    """Format an assertion message with the build ID for easier debugging."""
    return f"[Build: {build_id}] {message}"
