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

"""Integration test: the eval step runs end to end on the local Docker daemon.

This is a **step-level** test: it lives beside the eval step's Makefile in a
per-cluster subdir (``steps/eval/skypilot/test/docker/``, for the local Docker
environment), with its fixtures in the matching ``test-data/docker/``, and is
developed and run independently of the repository's central test suite (it is
not in ``testpaths``). Run it via ``make test`` with the repo-root ``.venv``
activated::

    make -C steps/eval/skypilot test

``make test`` first renders the Space (``make space``) and builds the image
locally (``make image``), so this test assumes both already exist and does no
rendering or building of its own:

  * The step is consumed from the **generated Space**; the build references it
    by the stable ``space://steps/eval`` URI, and the ``docker`` environment and
    monitors resolve through the generated ``space.yaml``'s ``base_uris`` chain
    to ``configurations/assets``.
  * The eval step's rendered ``Docker`` launcher references the **locally built**
    image (``${IMAGE_REF}``). The docker environment's ``pull_policy`` is
    ``if-not-present``, so that local image is used as-is — **no registry pull
    and no publish**. (Running pytest directly, without ``make image`` first,
    would make the launcher try to pull the placeholder registry and fail.)

Exercises the custom-image ``eval`` step as a real build: the container runs
``eval.sh``, which writes a stub ``results.json`` (HF-free — nothing is
downloaded) that the step registers as a single ``env://`` output artifact.

Requires a running Docker daemon. Auto-skips when not in the extended suite or
when running in CI/CD (until automatic image pulling is supported there).
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


# Real-infra build test (launches a local Docker container) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
# TODO: enable in CI/CD once automatic image pulling during the build is
# supported (mirrors test/integration/standalone/buildrunner/docker).
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in CI/CD until we have automatic image pulling during the build",
)
class TestDockerEval(AbstractYamlBuildRunnerTest):
    """eval runs its locally built image end to end on the Docker environment."""

    def _get_yaml_spec_dir(self) -> Path:
        # Fixtures (build.yaml/buildtest.yaml) live in the test-data/ dir that
        # mirrors this file, resolved by the repo's test/ ↔ test-data/ helper —
        # which keys off the first `test/` segment, so it works from both homes
        # of this test (see steps/README.md, "Two test modes"):
        #   Mode 1 (authoring)  steps/eval/skypilot/test/docker/
        #       -> steps/eval/skypilot/test-data/docker/  (co-located)
        #   Mode 2 (published)  test/steps/eval/skypilot/docker/
        #       -> test-data/steps/eval/skypilot/docker/  (parallel top-level tree)
        # In Mode 1 the Space is rendered and the image built by `make test`
        # (its `space`/`image` prerequisites) before this test runs.
        return get_test_data_dir_for(__file__)
