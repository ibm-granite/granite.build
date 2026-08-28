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

"""DPK tokenization with a DEFAULTED ``output_path``, on skypilot/slurm.

``dpk_config.output_path`` is optional: omitted, the step writes to ``./output`` in
its working directory and absolutizes that before emitting the artifact marker.
This fixture is the only one that exercises the default — the sibling fixtures both
set it explicitly, because their declared output URIs name a path
(``test-data/slurm`` additionally needs a shared path for its cross-target
handoff).

The render tests in ``test/test_dpk_step_render.py`` already pin the *rendering*
(``OUT='./output'``, then ``OUT="$(cd "$OUT" && pwd)"`` before the marker) and even
run that fragment under a local bash. What they cannot show is what this test does:

* the relative default resolves inside SkyPilot's actual working directory on a
  remote compute node, not just in a local temp dir;
* the absolute path it produces is one the skypilot monitor accepts, registering
  the ``tokens`` output from the marker — the monitor takes the marker path
  verbatim, so a relative or nonexistent path would register silently-wrong data
  rather than fail;
* the transform genuinely wrote there, which ``output_artifact_count`` asserts.

Single target by design. The default is safe only for output no other target reads:
the working directory is the per-run workdir
(``${shared_workdir}/builds/<build_id>/runs/<targetrun_id>``), minted per *target*
by ``setup_skypilot`` and ``rm -rf``'d by ``teardown_skypilot`` when the target
finishes, so a consumer in another target would read a deleted directory. The
cross-target path is covered by ``test/slurm/``, which sets an explicit ``/shared``
``output_path``.

Input is a public ``hf://`` dataset, so no HF_TOKEN is needed.

Real-infra test, gated on a reachable Docker SLURM cluster, so it auto-skips in CI
and on machines without one (``make test-setup`` brings up SLURM + MinIO). Extended
suite only.
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
class TestSkypilotSlurmDpkDefaultOutput(AbstractYamlBuildRunnerTest):
    """dpk step: omitted output_path -> ./output, absolutized in the marker."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
