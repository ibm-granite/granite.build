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

"""eval/lm-eval runs aime25 end to end on the local CPU SkyPilot/slurm cluster.

One target: an ``hf://`` model input (a tiny public test model), the step installs
CPU torch and ``lm-eval[hf]`` on the bare node, runs ``lm_eval run --tasks aime25
--limit 1``, and pushes the results directory to an ``hf://`` dataset with a
``test_`` prefix (cleaned up by ``scripts/rm_hf_repos.py``). Reaching SUCCESS with
one output artifact proves the install, the argv, the results file and the marker
on a real node; the score itself is meaningless for a model this size.

Needs a write ``HF_TOKEN`` for the push, and a reachable Docker SLURM cluster
(``make test-setup``). Extended suite only.

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
class TestSkypilotSlurmLmEvalCpu(AbstractYamlBuildRunnerTest):
    """eval/lm-eval: aime25 on CPU, hf:// model in, hf:// results out."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
