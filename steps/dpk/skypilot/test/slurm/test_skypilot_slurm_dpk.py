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

"""DPK tokenize -> validate on skypilot/slurm, via the `dpk` step.

Exercises **both** of the step's modes in one build, which is the point of the
fixture:

* ``tokenize`` — **transform mode**. The build names one DPK detail
  (``transform: tokenization2arrow``) and the step derives the python module and
  the pip extra. Reaching SUCCESS proves those derivations work against real DPK,
  not just in the render tests.
* ``validate`` — **command mode**, running the step's own bundled
  ``src/validate_tokens.py``. Proves ``file_mounts: {src: src}`` actually lands
  the script on the node — the mechanism that replaces the unsupported relative
  ``file_mounts`` the DPK template used to rely on (#294).

It also covers the cross-node handoff: ``validate`` binds ``tokenize.tokens``, and
the two targets may be scheduled on different compute nodes. ``env://`` moves no
bytes, so the tokens survive the hop only because the fixture writes them under
``/shared`` (the slurm environment's ``shared_workdir``), mounted on every node. A
node-local ``/tmp`` path would fail here.

Input is a public ``hf://`` dataset, so no HF_TOKEN is needed — the launcher's
``hf download`` runs anonymously.

Real-infra test, gated on a reachable Docker SLURM cluster, so it auto-skips in CI
and on machines without one (``make test-setup`` brings up SLURM + MinIO). Extended
suite only.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
``_get_yaml_spec_dir`` below, resolved by the repo's ``test/`` <-> ``test-data/``
helper so the same file works in both test modes (see steps/README.md).
"""

from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.skypilot_integration


@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make test-setup)",
)
class TestSkypilotSlurmDpk(AbstractYamlBuildRunnerTest):
    """dpk step: transform mode -> env:///shared handoff -> command mode."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
