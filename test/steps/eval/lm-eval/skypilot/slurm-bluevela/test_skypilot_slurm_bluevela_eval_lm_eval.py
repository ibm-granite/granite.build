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

"""eval/lm-eval runs aime25 on an internal GPU SLURM cluster, on one H200.

The GPU counterpart of ``../slurm-cpu``: the same example benchmark, but with a
real granite model (``ibm-granite/granite-4.2-3b``, bound as the target's ``model``
input), PyPI's default CUDA torch, ``device=cuda``, and 3 of aime25's 30 problems
with a 16384-token generation budget. The score is noisy and isn't asserted;
reaching SUCCESS proves:

* the ``hf://`` model input is pulled onto the cluster's shared filesystem and its
  path reaches ``--model_args pretrained=...``;
* setup installs CUDA torch and ``lm-eval[hf]`` on the bare GPU node;
* ``lm_eval run`` generates on the GPU and writes its results file;
* the results directory is registered as ``results``, and the queued hfpush uploads
  it. :class:`libgbtest.hfpush.HfPushCleanupMixin` names that dataset per run,
  checks it landed on the Hub, and deletes it afterwards.

It uses the step's local space and the in-repo environment
``space://environments/skypilot/slurm/ibm-bluevela``, which logs in as the shared
functional ID with the key file ``~/.ssh/ibm-bluevela.key`` (mode 600, no
passphrase). So the step resolves from this source tree, and the test needs no
secret manager. It needs an ``HF_TOKEN`` that can write ``GBTEST_HF_PUSH_ORG``
(default ibm-research).

Marked ``ibm`` (IBM-only cluster), extended suite only, and skipped in SPS CI/CD,
which has no key file. Run it by hand from the repo's IBM venv (``make ibm-venv``)::

    make -C steps/eval/lm-eval/skypilot space
    cd steps/eval/lm-eval/skypilot
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
class TestSkypilotSlurmBlueVelaLmEvalGpu(
    HfPushCleanupMixin, AbstractYamlBuildRunnerTest
):
    """eval/lm-eval: aime25 on one GPU, hf:// model in, hf:// results out."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    # buildtest.yaml's timeout_minutes is 60, above pytest's global 1200s cap
    # (pyproject.toml), so give the runner 60 minutes plus 5 of headroom.
    @pytest.mark.timeout(60 * 60 + 300)
    def test_runner(self):
        super().test_runner()
