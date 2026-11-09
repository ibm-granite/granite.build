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

"""eval/livecodebench-generate, then eval/livecodebench-score, on an internal GPU SLURM cluster.

Two targets joined by a binding: ``generate`` runs ``ibm-granite/granite-4.2-3b``
with vLLM on one H200, on one contest day of LiveCodeBench ``release_v6``
(2025-04-05, 6 problems, one sample each, thinking off), and pushes its
``completions``; ``score`` reads them by binding and runs LiveCodeBench's scorer on
the cpu partition, offline and without credentials, and pushes ``results``. The
score is not checked (GPU sampling varies between runs); reaching SUCCESS proves:

* setup installs vLLM and LiveCodeBench on the bare GPU node, and the model
  generates on the GPU;
* the completions reach the score step through the binding;
* the score step installs on a CPU node and scores them;
* both outputs are pushed. :class:`libgbtest.hfpush.HfPushCleanupMixin` names the
  datasets per run, checks they landed on the Hub, and deletes them afterwards.

The exact-score check is the score step's own ``slurm-cpu`` test, against a fixture
with known answers.

It uses the generate step's local space and the in-repo environment
``space://environments/skypilot/slurm/bluevela``, so the steps resolve from this
source tree. That environment logs in as the shared functional ID with the key in
the ``BV_SSH_PRIVATE_KEY`` space secret, created once in the local space from
``~/.ssh/ibm-bluevela.key``
(``gb secret create BV_SSH_PRIVATE_KEY --from-file ~/.ssh/ibm-bluevela.key``). It
also needs an ``HF_TOKEN`` that can write ``GBTEST_HF_PUSH_ORG`` (default
ibm-research).

IBM infrastructure through the local space, so marked ``ibm``; extended suite only.
Skips unless ``~/.ssh/ibm-bluevela.key`` exists on the host, and in SPS CI/CD,
which has no key on the host. Run it by hand from the repo's IBM venv
(``make ibm-venv``)::

    make -C steps/eval/livecodebench-generate/skypilot space
    cd steps/eval/livecodebench-generate/skypilot
    GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live GBTEST_STANDALONE_ENVIRONMENT=STAGING \\
      PYTHONPATH=src ../../../../.venv/bin/python -m pytest -s -v test/slurm-bluevela

The fixture's build.yaml and buildtest.yaml live in the directory returned by
``_get_yaml_spec_dir`` below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.hfpush import HfPushCleanupMixin

pytestmark = [pytest.mark.skypilot_integration, pytest.mark.ibm]

# The key the BV_SSH_PRIVATE_KEY space secret is created from; the environment reads
# the secret (cluster_ssh_configs IdentityKey), not this path.
_BLUEVELA_KEY = Path("~/.ssh/ibm-bluevela.key").expanduser()


@extended_testing_only
# No xdist_group: skypilot_integration already puts it in the serialized
# "skypilot" group, and under --dist=loadgroup any other group is a UsageError.
@pytest.mark.skipif(
    not _BLUEVELA_KEY.is_file(),
    reason=f"BlueVela granitebuild SSH key not found at {_BLUEVELA_KEY}",
)
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="No BlueVela SSH key on SPS CI/CD hosts",
)
class TestSkypilotSlurmBlueVelaLivecodebenchGpu(
    HfPushCleanupMixin, AbstractYamlBuildRunnerTest
):
    """eval/livecodebench-generate on one GPU, then eval/livecodebench-score on CPU."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    # buildtest.yaml's timeout_minutes is 60, above pytest's global 1200s cap
    # (pyproject.toml), so give the runner 60 minutes plus 5 of headroom.
    @pytest.mark.timeout(60 * 60 + 300)
    def test_runner(self):
        super().test_runner()
