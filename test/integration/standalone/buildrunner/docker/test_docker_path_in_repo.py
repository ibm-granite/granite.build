# SPDX-License-Identifier: Apache-2.0

"""A docker build with two hf:// inputs naming different files in one repo.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below. The step checks that each input is bound at its own
downloaded file inside the container.

The Docker daemon must be available.
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


# Extended/nightly test: it launches a local Docker container and the fixture
# performs a REAL HuggingFace pull, so it runs under `make extended-tests`.
@extended_testing_only
# Same CI skip as docker/1step: no automatic image pulling during the build yet.
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in CI/CD until we have automatic image pulling during the build",
)
class TestDockerPathInRepo(AbstractYamlBuildRunnerTest):
    """Docker build bound at two files pulled from the same hf:// repo."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "path-in-repo"
