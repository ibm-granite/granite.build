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

HF network calls are mocked so no real HuggingFace traffic is required.
The Docker daemon must be available (gated by skipif_no_docker).
"""

# import os
import os
from pathlib import Path
from typing import Self

import pytest

# from unittest.mock import patch

pytest.importorskip("kubernetes_asyncio")
from lib.buildwatcher.buildtest import (
    AbstractBuildRunnerTest,
    BuildTestSpecification,
    ExpectedTarget,
)

pytestmark = pytest.mark.ibm

# from gbcommon.uri.hf import HfURI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEST_DATA_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "test-data"
    / "integration"
    / "standalone"
    / "buildrunner"
    / "docker"
    / "docker-hf"
).resolve()

# ---------------------------------------------------------------------------
# Docker availability guard
# ---------------------------------------------------------------------------


# def _docker_available() -> bool:
#     """Return True if a Docker/Podman daemon is reachable via the Python SDK."""
#     try:
#         import docker

#         client = docker.from_env()
#         client.ping()
#         return True
#     except Exception:
#         return False


# skipif_no_docker = pytest.mark.skipif(
#     not _docker_available(),
#     reason="Docker/Podman daemon not available",
# )

# ---------------------------------------------------------------------------
# Test specification
# ---------------------------------------------------------------------------

_SPEC = BuildTestSpecification(
    build_yaml=str(_TEST_DATA_DIR / "build.yaml"),
    space_uri=f"file://{_TEST_DATA_DIR}/space",
    targets=["image-run"],
    target_expections=[
        ExpectedTarget(
            target_name="image-run",
            step_count=1,
            input_artifact_count=1,
            output_artifact_count=2,
            jobstats_count=3,
        ),
    ],
    simulate_failure=False,
    timeout_minutes=10,
)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


# TODO: We need to disable this skip when image pulling is supported
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in CI/CD until we have automatic image pulling during the build",
)
class TestDockerImageBuild(AbstractBuildRunnerTest):
    """Integration test: HF input → Docker image step → HF output.

    Runs a real local Docker container with HF I/O mocked so no actual
    HuggingFace network calls are made.
    """

    # @pytest.fixture(autouse=True)
    # def _hf_mocks(self: Self):
    #     """Mock HfURI.sync and HfURI.push to avoid real HuggingFace network calls."""
    #     with (
    #         patch.object(HfURI, "sync", return_value=True),
    #         patch.object(HfURI, "push", return_value=True),
    #         patch.object(HfURI, "exists", return_value=True),   # TODO: use an existing input artifact in HF
    #     ):
    #         yield

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        """Return the build spec for the Docker image HF pipeline test."""
        return _SPEC
