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

"""eval/unitxt on skypilot/slurm, CPU only.

One target evaluates a tiny public model (``SmolLM2-135M-Instruct``, bound as the
target's ``model`` input) on two ``cards.mmlu_pro.engineering`` instances with the
CPU build of torch. The score is meaningless; reaching SUCCESS proves the plumbing
on a real cluster:

* the ``hf://`` model input is staged and its path reaches
  ``--model_args pretrained=...``;
* setup's two-phase install works: torch from ``torch_index_url`` (the PyTorch CPU
  index), then unitxt and ``hf_packages`` from PyPI;
* ``unitxt-evaluate --model hf`` runs to completion with ``--trust_remote_code``;
* the output directory is registered as ``results`` from the artifact marker.

GPU evaluation is not covered here: the local cluster has no GPUs. A GPU test
against an internal GPU SLURM cluster belongs under ``test/integration/ibm/``.

The model and the dataset are public, so no HF_TOKEN is needed.

Real-infra test, gated on a reachable Docker SLURM cluster, so it auto-skips in CI
and on machines without one (``make slurm-setup`` brings up SLURM). Extended
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
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
class TestSkypilotSlurmUnitxtCpu(AbstractYamlBuildRunnerTest):
    """eval/unitxt step: hf mode on CPU against a bound hf:// model input."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
