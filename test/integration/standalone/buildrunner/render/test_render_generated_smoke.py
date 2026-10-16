# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Self-contained integration test for the issue #278 workflow.

Exercises the full chain end-to-end on the in-process Bash environment (no cloud,
no HF, no external gbserver):

  1. ``gb build describe --raw`` (build_describe) resolves a PARAMETERIZED build.yaml
     into an executable one.
  2. ``gbtest render`` (generate_skeleton) emits a skeleton buildtest.yaml.
  3. The skeleton's FIXME placeholders are filled (simulating the human edit),
     and ``space_uri`` is added so space:// URIs resolve from the local space.
  4. The build runs through the harness with the ``-f`` override
     (build_yaml_override) pointing at the rendered executable build, and must
     reach SUCCESS with the derived artifact counts.

No cross-branch dependency (does not use the DPK template).
"""

from pathlib import Path

import pytest
import yaml
from libgbtest.buildrunner.buildtest import AbstractYamlBuildRunnerTest
from libgbtest.buildrunner.buildtest_gen import generate_skeleton

from gbcli.services.service_build import build_describe

pytestmark = pytest.mark.standalone

_REPO_ROOT = Path(__file__).resolve().parents[5]
_LOCAL_SPACE = _REPO_ROOT / "configurations" / "spaces" / "local"

# A PARAMETERIZED build: one Bash target, one command step that echoes the
# MESSAGE param and registers a single env:// output from its GB_ARTIFACT marker.
_PARAM_BUILD = """\
granite.build:
  name: render-smoke-$${SUFFIX}
  targets:
    command:
      allow_unknown: true
      environment_uri: space://environments/bash
      outputs:
        out:
          uri: "env:///tmp/gb-render-smoke-out.txt"
          type: fileset
      steps:
        - step_uri: space://steps/command
          config:
            command_config:
              command: >-
                echo "message: $${MESSAGE}";
                echo "GB_ARTIFACT_ID:out GB_ARTIFACT_PATH:/tmp/gb-render-smoke-out.txt"
            compute_config:
              num_nodes: 1
"""


@pytest.mark.xdist_group(name="buildtest_local")
class TestRenderGeneratedBashSmoke(AbstractYamlBuildRunnerTest):
    """render -> gbtest render -> fill -> run (via -f) on the Bash environment."""

    @pytest.fixture(autouse=True)
    def _prepare_generated_fixture(self, tmp_path):
        # 1. Render the parameterized build into an executable build.yaml via the
        #    offline `gb build describe --raw` path. Keep a distinct name so the
        #    run exercises the -f override (not the sibling).
        src = tmp_path / "build.yaml.in"
        src.write_text(_PARAM_BUILD, encoding="utf-8")
        exec_build = tmp_path / "exec-build.yaml"
        exec_build.write_text(
            build_describe(
                github_token="",
                filename=str(src),
                format="yaml",
                raw=True,
                params=["MESSAGE=hello", "SUFFIX=1"],
            ),
            encoding="utf-8",
        )

        # 2. Generate the skeleton buildtest.yaml from the executable build.
        doc = yaml.safe_load(generate_skeleton(exec_build))

        # 3. Fill the FIXME placeholders (the human edit) and point the harness at
        #    the local space so space:// URIs resolve without cloning.
        exp = doc["target_expectations"][0]
        exp["step_count"] = 1  # one command step; env push is a no-op (no extra step)
        exp["jobstats_count"] = 1  # not asserted under standalone NoopLineageStore
        doc["space_uri"] = str(_LOCAL_SPACE)
        (tmp_path / "buildtest.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

        # 4. Drive the run against the executable build via the -f override.
        self._spec_dir = tmp_path
        self._build_yaml_override = str(exec_build)

    def _get_yaml_spec_dir(self) -> Path:
        return self._spec_dir
