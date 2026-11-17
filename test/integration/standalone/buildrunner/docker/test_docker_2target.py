#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Integration test for cross-target mem:// binding on the Docker environment.

Two docker `command`-step targets: the first emits an
`GB_ARTIFACT_ID:… GB_ARTIFACT_STATE:…` marker (a mem:// output), the second
binds it and asserts the value arrived verbatim via the mem:// store — proving
the docker monitor now recognizes the STATE marker and mem:// bindings work on
the docker backend. Mirrors the bash mem 2-target test. The Docker daemon must
be available.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.docker_required


# Real-infra build test (launches local Docker containers) — extended suite only.
@extended_testing_only
# TODO: disable this skip when image pulling is supported during the build.
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in CI/CD until we have automatic image pulling during the build",
)
class TestDockerMem2Target(AbstractYamlBuildRunnerTest):
    """Cross-target mem:// output → input binding over two docker command steps."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "2target"
