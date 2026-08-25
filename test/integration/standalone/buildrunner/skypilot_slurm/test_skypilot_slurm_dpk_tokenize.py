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

"""End-to-end DPK tokenization + validation on the local Docker SLURM cluster.

Runs the DPK_Tokenize_Skypilot template's two-target pipeline on the SkyPilot
SLURM backend: `tokenize` installs Data Prep Kit from PyPI and runs
``dpk_tokenization2arrow.runtime`` over the committed sample Parquet, then
`validate` binds the tokenize output and runs ``validate_tokens.py`` against it,
failing the build on any inconsistency. Reaching SUCCESS proves the template runs
end to end on skypilot/slurm AND that validate's real checks pass on the
known-good fixture (3 arrow files, 6 documents, 85 tokens).

IMPORTANT — cross-node artifact handoff:
    `tokenize` and `validate` are separate workloads and may be scheduled on
    different SLURM compute nodes (c1..c4). ``env://`` moves no bytes and the
    consumer's ``{{ bindings.tokens.binding.path }}`` resolves to the declared
    URI path — so the tokens survive the hop only because they are written under
    ``/shared`` (the slurm environment's ``shared_workdir``, mounted on every
    node) and the ``env:///`` URI points at that same absolute path. This is the
    shared-filesystem counterpart to the aws fixture's ``s3://`` handoff (aws
    provisions a separate EC2 per target with no shared FS, so it needs s3://).
    The shipped template's ``env:///tmp/...`` default would fail here across
    nodes (produce=c4 / consume=c3 -> "No such file or directory"). See the
    fixture build.yaml.

Like the sibling skypilot_slurm build tests this is a real-infra test gated on a
reachable Docker SLURM cluster, so it auto-skips in CI and on machines without
one (run ``make slurm-setup`` to bring it up). It runs in the extended suite only
(``make extended-tests``).

The fixture's build.yaml and buildtest.yaml live in the directory returned by
``_get_yaml_spec_dir`` below.
"""

from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import (
    _slurm_cluster_reachable,
)
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches SLURM jobs via Skypilot) — extended suite only.
@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
class TestSkypilotSlurmDpkTokenize(AbstractYamlBuildRunnerTest):
    """DPK tokenize -> validate on skypilot/slurm; validate binds tokenize's env:///shared output."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "dpk-tokenize"
