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

"""PII redaction on skypilot/slurm, through the same `dpk` step as tokenization.

**This test exists to prove the step is general.** The sibling
``test/slurm/test_skypilot_slurm_dpk.py`` already shows tokenization working; this
one runs an unrelated DPK transform — ``pii_redactor``, which shares no code, no
flags, and no dependencies with tokenization — and the only things that differ are
in the build.yaml: ``transform``, ``args``, and the artifact names.
``step-template.yaml`` is untouched.

Reaching SUCCESS therefore demonstrates the two derivations hold for a second
transform:

* module — ``transform: pii_redactor`` → ``python -m dpk_pii_redactor.runtime``
* pip    — → ``data-prep-toolkit-transforms[pii-redactor]==<dpk_version>``

so adding transform #3 (and #31) is a build.yaml change, never a step change.

**This is the slow one.** The ``[pii-redactor]`` extra resolves to 125 packages
including torch, flair, and presidio (with numpy pinned below 1.29 by DPK's own
constraint), and the transform downloads a flair NER model on first use. Locally
the install alone took several minutes, so the fixture allows 60. It is the
strongest argument for supporting a prebaked ``dpk_config.dpk_image`` later; an image
is deliberately not used here so the fixture stays credential-free and runs on the
local cluster, which has no Pyxis.

Single target — a generality proof, not a handoff test; the tokenization fixture
already covers the cross-node ``env:///shared`` path.

Real-infra test, gated on a reachable Docker SLURM cluster, so it auto-skips in CI
and on machines without one (``make test-setup``). Extended suite only.
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
class TestSkypilotSlurmDpkPii(AbstractYamlBuildRunnerTest):
    """dpk step, second transform: pii_redactor with no step change."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
