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
``space://environments/skypilot/slurm/ibm-bluevela``, which logs in as the shared
functional ID with the key file ``~/.ssh/ibm-bluevela.key`` (mode 600, no
passphrase). So the steps resolve from this source tree, and the test needs no
secret manager. It needs an ``HF_TOKEN`` that can write ``GBTEST_HF_PUSH_ORG``
(default ibm-research).

Marked ``ibm`` (IBM-only cluster), extended suite only, and skipped in SPS CI/CD,
which has no key file. Run it by hand from the repo's IBM venv (``make ibm-venv``)::

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

pytestmark = pytest.mark.ibm


@extended_testing_only
# Serialize with the other tests on this cluster (same login nodes, same SSH
# ControlMaster socket).
@pytest.mark.xdist_group(name="buildtest_bv")
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Needs the cluster's SSH key file, which SPS CI/CD does not have",
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
