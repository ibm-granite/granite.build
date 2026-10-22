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

SCOPE: a pass proves the step runs and that its marker is scraped and bound.
``output_artifact_count`` is satisfied by registration, so it does not additionally
assert that files landed at the output URI -- and do NOT add such an assertion here.
Under this in-process harness the push reads an ephemeral workspace path that is torn
down with the run, so the pushed files are not observable from the test even though
delivery works. Verified against a real gbserver build (marker pointing at
``~/.granite.build/workdir/.../launch-<id>/outputs``, ~85MB of adapter weights
delivered to the declared absolute file: URI). Use gbserver, not this test, to check
delivery.

Cost: heavyweight. The first run builds a venv and pip-installs torch + ray
(fm-tune's ``core`` extra; ``main.py`` imports ray unconditionally), then runs a
single short LoRA pass with HPO disabled. Extended suite only.

Requires the fm-tune copy vendored at ``autotunex/src/fm-tune``; auto-skips when
it is absent or when not in the extended suite.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
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

    def _out_dir(self) -> Path:
        """Absolute, durable output dir for this run.

        `outputs.custom.uri` MUST be absolute: gbserver resolves a relative file: URI
        against its own CWD, which for the harness is an ephemeral workspace torn down
        after the run -- so the pushed artifact would vanish and the recorded URI would
        point at nothing.
        """
        return Path(tempfile.mkdtemp(prefix="autotune-out-"))

    def _get_yaml_spec_dir(self) -> Path:
        """Render the committed fixture into a temp spec dir with absolute paths.

        The harness reads build.yaml/buildtest.yaml verbatim and has no parameter
        substitution of its own -- but the spec dir is whatever this method returns,
        so the two absolute paths a static fixture cannot express are filled in here:

          * ``@FM_TUNE_ROOT@`` -- the vendored fm-tune checkout. The step runs with
            its CWD in the build workdir, so a relative path would not resolve.
          * ``@DATASET_DIR@`` -- the fixture's dataset. gbserver resolves a relative
            ``file:`` URI against its OWN CWD, not this directory.

        Both are derived from this file's location, and ``parents[5]`` is the repo
        root in *either* test mode, so the same fixture works from both homes (see
        steps/README.md, "Two test modes"):
          Mode 1 (authoring)  steps/autotune/bash/test/local/
              -> steps/autotune/bash/test-data/local/   (co-located)
          Mode 2 (published)  test/steps/autotune/bash/local/
              -> test-data/steps/autotune/bash/local/   (parallel top-level tree)

        ``space_uri`` is resolved against the REAL fixture dir before being written
        out absolute, so publish-step's Mode-2 rewrite of that field still governs
        which Space is used.
        """
        fixture = get_test_data_dir_for(__file__)
        spec = Path(tempfile.mkdtemp(prefix="autotune-buildtest-"))

        build = (fixture / "build.yaml").read_text()
        build = build.replace("@FM_TUNE_ROOT@", str(FM_TUNE_ROOT))
        build = build.replace("@DATASET_DIR@", str(fixture / "dataset"))
        build = build.replace("@OUTPUT_DIR@", str(self._out_dir()))
        assert "@" not in build.split("granite.build:", 1)[1], "unsubstituted token"
        (spec / "build.yaml").write_text(build)

        bt = yaml.safe_load((fixture / "buildtest.yaml").read_text())
        raw = str(bt.get("space_uri", "../../space")).removeprefix("file://")
        if not Path(raw).is_absolute():
            raw = str((fixture / raw).resolve())
        bt["space_uri"] = f"file://{raw}"
        (spec / "buildtest.yaml").write_text(yaml.safe_dump(bt, sort_keys=False))

        # Deliberately not cleaned up: on failure the rendered build.yaml is the
        # first thing you want to read, and it is a couple of KB under $TMPDIR.
        return spec
