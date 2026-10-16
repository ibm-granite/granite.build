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

"""Integration test: the byoc step runs end to end on Skypilot/slurm.

This is a **step-level** test: it lives beside the byoc step's Makefile in a
per-cluster subdir (``steps/byoc/skypilot/test/slurm/``, for the SkyPilot/slurm
cluster), with its fixtures in the matching ``test-data/slurm/``, and is
developed and run independently of the repository's central test suite (it is
not in ``testpaths``). Run it via ``make test`` with the repo-root ``.venv``
activated::

    make -C steps/byoc/skypilot test

``make test`` depends on ``make space``, so the git-ignored ``space/`` directory
that the ``buildtest.yaml``'s ``space_uri`` points at is always rendered before
pytest runs — this test therefore assumes the Space already exists and does no
rendering of its own. (pytest still discovers the repo-root ``pyproject.toml``
as its rootdir, so the project ``pythonpath`` — ``libgbtest``, ``integration.*``
— and marker registrations apply regardless of the invocation directory.)

Exercises the public-image ``byoc`` step as a real build: it git-clones a tiny
public repo (``octocat/Hello-World``) in ``setup`` and runs a command in
``run``, registering one output artifact.

Design notes:
  * The step is consumed from the **generated Space** produced by ``make space``
    (the ``test`` target's prerequisite). The build references the step by the
    stable ``space://steps/byoc`` URI; everything else (environment, monitor)
    resolves through the generated ``space.yaml``'s ``base_uris`` chain to
    ``configurations/assets``. This validates the real framework output.
  * **Bare node** (``image: ""``) so no Pyxis is required — the local Docker
    slurm cluster runs it directly.
  * **HF-free** (``env://`` in/out) so it is fast and needs no HF_TOKEN.

Requires a running Docker SLURM cluster (see scripts/slurm/setup-slurm.sh).
Auto-skips when the cluster is not reachable via SSH or when not in the extended
suite.
"""

from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import (
    _slurm_cluster_reachable,
)
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches a SLURM job via Skypilot) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
class TestSkypilotSlurmByoc(AbstractYamlBuildRunnerTest):
    """byoc clones a public repo and runs a command end to end on slurm."""

    def _get_yaml_spec_dir(self) -> Path:
        # Fixtures (build.yaml/buildtest.yaml) live in the test-data/ dir that
        # mirrors this file, resolved by the repo's test/ ↔ test-data/ helper —
        # which keys off the first `test/` segment, so it works from both homes
        # of this test (see steps/README.md, "Two test modes"):
        #   Mode 1 (authoring)  steps/byoc/skypilot/test/slurm/
        #       -> steps/byoc/skypilot/test-data/slurm/  (co-located)
        #   Mode 2 (published)  test/steps/byoc/skypilot/slurm/
        #       -> test-data/steps/byoc/skypilot/slurm/  (parallel top-level tree)
        # In Mode 1 the Space is rendered by `make space` (the `make test`
        # prerequisite) before this test runs.
        return get_test_data_dir_for(__file__)
