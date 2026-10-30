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

"""data-designer with a model served on the node, on the local Docker SLURM cluster.

One target: a sampler + ``llm-text`` config, 4 rows, ``min_records: 4``, against
``granite4:350m`` served by the step's own ``local_model: {runtime: ollama}``. No API
key, no space secret and no external model endpoint — only the downloads — so anyone
with the local cluster can run it. See the fixture's build.yaml for what SUCCESS proves.

Real-infra test, gated on a reachable Docker SLURM cluster, so it auto-skips in CI and
on machines without one (``make test-setup`` brings it up). Extended suite only.

Fixtures (build.yaml/buildtest.yaml) live in the ``test-data/`` dir mirroring this
file, resolved by the repo's ``test/`` <-> ``test-data/`` helper so the pairing holds
in both test modes (see steps/README.md).
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
class TestSkypilotSlurmDataDesignerOllama(AbstractYamlBuildRunnerTest):
    """data-designer step: generation through a model served on the node, on SLURM."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
