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

"""DPK tokenization with in-step validation on skypilot/slurm.

One target. ``transform: tokenization2arrow`` is the only DPK detail the build
gives — the step derives the python module and the pip extra — and
``validate: true`` is the only thing it says about checking the result. The step
finds ``src/validate_tokenization2arrow.py`` by the ``validate_<transform>.py``
rule and runs it on the node after the transform and *before* emitting the
artifact marker, so reaching SUCCESS proves:

* the derivations work against real DPK, not just in the render tests;
* the validator ran and accepted the output — a failure would fail the target and
  register nothing;
* ``validation.json`` was written into the registered output directory.

This replaced a two-target ``tokenize`` -> ``validate`` build, where the second
target bound the first's output and drove the validator through command mode. The
flag removed roughly 25 lines of build.yaml. It also retired this fixture's
cluster-level coverage of the cross-node ``env:///shared`` handoff and of command
mode; both are covered by unit tests now, and the sibling build.yaml records where
to look if a handoff regression is ever suspected.

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
