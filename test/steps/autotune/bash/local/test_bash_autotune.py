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

"""Integration test: the autotune step runs end to end on the bash environment.

This is a **step-level** test: it lives beside the autotune step's Makefile in a
per-cluster subdir (``steps/autotune/bash/test/local/``, "local" being the bash
environment's only target), with its fixtures in the matching ``test-data/local/``,
and is developed and run independently of the repository's central test suite (it
is not in ``testpaths``). Run it via ``make test`` with the repo-root ``.venv``
activated::

    make -C steps/autotune/bash test

``make test`` depends on ``make space``, so the git-ignored ``space/`` directory
that the ``buildtest.yaml``'s ``space_uri`` points at is always rendered before
pytest runs.

What it proves that the unit tests cannot: that ``run.py``'s ``GB_ARTIFACT_ID``
marker is actually scraped by the bash monitor and binds the ``custom`` output.
The step sets ``skip_finding_output_artifacts``, so the marker is the *only*
source of that binding -- if it lands mid-line, the build "succeeds" with no
output, which ``output_artifact_count: 1`` catches.

Cost: heavyweight. The first run builds a venv and pip-installs torch + ray
(fm-tune's ``core`` extra; ``main.py`` imports ray unconditionally), then runs a
single short LoRA pass with HPO disabled. Extended suite only.

Requires the fm-tune copy vendored at ``autotunex/src/fm-tune``; auto-skips when
it is absent or when not in the extended suite.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

# Repo root: steps/autotune/bash/test/local/ -> up five.
REPO_ROOT = Path(__file__).resolve().parents[5]
FM_TUNE_ROOT = REPO_ROOT / "autotunex/src/fm-tune"


@extended_testing_only
@pytest.mark.skipif(
    not (FM_TUNE_ROOT / "main.py").is_file(),
    reason=f"vendored fm-tune not found at {FM_TUNE_ROOT}",
)
class TestBashAutotune(AbstractYamlBuildRunnerTest):
    """autotune trains once and registers its output via the artifact marker."""

    def _get_yaml_spec_dir(self) -> Path:
        # Fixtures (build.yaml/buildtest.yaml) live in the test-data/ dir that
        # mirrors this file, resolved by the repo's test/ <-> test-data/ helper --
        # which keys off the first `test/` segment, so it works from both homes of
        # this test (see steps/README.md, "Two test modes"):
        #   Mode 1 (authoring)  steps/autotune/bash/test/local/
        #       -> steps/autotune/bash/test-data/local/   (co-located)
        #   Mode 2 (published)  test/steps/autotune/bash/local/
        #       -> test-data/steps/autotune/bash/local/   (parallel top-level tree)
        return get_test_data_dir_for(__file__)
