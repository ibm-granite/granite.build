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

"""fm-tune on skypilot/slurm: one tiny CPU LoRA run end to end.

The fixture clones fm-tune from the public upstream (source.repo), so this tests
the step against published fm-tune, not local edits to autotunex/src/fm-tune.
Real HF pulls (public repos, no token). Gated on a reachable Docker SLURM
cluster (`make test-setup`). Extended suite only.
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
class TestSkypilotSlurmFmTune(AbstractYamlBuildRunnerTest):
    """fm-tune step: a --no_autotune LoRA run registers its model output."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(60 * 60 + 300)
    def test_runner(self):
        super().test_runner()
