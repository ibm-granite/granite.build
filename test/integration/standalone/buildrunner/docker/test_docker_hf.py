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

"""Integration test for the Docker image step with HF URI input/output.

Validates the full pipeline:
  HF URI input  →  Docker container (image step)  →  HF URI output

The Docker daemon must be available.

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

pytestmark = pytest.mark.docker_required


# TODO: We need to disable this skip when image pulling is supported
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in CI/CD until we have automatic image pulling during the build",
)
class TestDockerImageBuild(AbstractYamlBuildRunnerTest):
    """Integration test: HF input → Docker image step → HF output.

    Runs a real local Docker container with HF I/O mocked so no actual
    HuggingFace network calls are made.
    """

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "docker-hf"
