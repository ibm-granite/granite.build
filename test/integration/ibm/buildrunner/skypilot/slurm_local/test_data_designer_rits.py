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

"""data-designer generation against RITS on the local Docker SLURM cluster.

One target: a sampler + ``llm-text`` config, 4 rows, ``min_records: 4`` and a jsonl
export, so reaching SUCCESS proves the install, the spec hand-off, the space secret
becoming the RITS header, a live generation and the artifact marker — see the
fixture's build.yaml for the full list.

**IBM-only** (``pytest.mark.ibm``): RITS is IBM-internal, so this lives here rather
than in the step's own ``test/`` tree, which ``make publish-step`` ships. The step's
shipped build tests use an open model endpoint instead.

**Two gates.** The Docker SLURM cluster must be reachable (``make slurm-setup``), and
``RITS_API_KEY`` must be exported. That variable only decides whether the test runs:
the build reads the space secret ``RITS_API_KEY`` from the Space it resolves — the
step's locally rendered ``steps/data/data-designer/skypilot/space`` (space name
``public``), so render it first and store the secret there; see the step's
README.md, "IBM-only build tests".

Fixtures (build.yaml/buildtest.yaml) are in the ``test-data/`` dir mirroring this
file, under ``data-designer-rits/``.
"""

import os
from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = [pytest.mark.ibm, pytest.mark.skypilot_integration]


@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
@pytest.mark.skipif(
    not os.environ.get("RITS_API_KEY"),
    reason="RITS_API_KEY not exported (the build also needs it as a space secret)",
)
class TestSkypilotSlurmDataDesignerRits(AbstractYamlBuildRunnerTest):
    """data-designer step: a live RITS generation, end to end on SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "data-designer-rits"
