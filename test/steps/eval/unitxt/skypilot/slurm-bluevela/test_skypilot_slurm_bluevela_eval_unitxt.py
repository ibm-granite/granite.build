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

"""eval/unitxt on an internal GPU SLURM cluster, on one H200.

The GPU counterpart of ``../slurm-cpu``: the same card
(``cards.mmlu_pro.engineering``, 10 instances), but a real granite model
(``ibm-granite/granite-4.2-3b``, bound as the target's ``model`` input) that
unitxt's hf mode runs on the GPU (``device=cuda``) with PyPI's default CUDA build
of torch. Ten instances make the score noisy; reaching SUCCESS proves:

* the ``hf://`` model input is pulled onto the cluster's shared filesystem and its
  path reaches ``--model_args pretrained=...``;
* setup installs the CUDA torch build, then unitxt and ``hf_packages``, on the bare
  GPU node;
* ``unitxt-evaluate --model hf`` runs the model on the GPU (torch fails with
  ``device=cuda`` if the job has no usable GPU);
* the output directory is registered as ``results``, and the queued hfpush uploads
  it. :class:`libgbtest.hfpush.HfPushCleanupMixin` names that dataset per run,
  checks it landed on the Hub, and deletes it afterwards.

It uses the local space and the in-repo environment
``space://environments/skypilot/slurm/ibm-bluevela``, which logs in as the shared
functional ID with the key file ``~/.ssh/ibm-bluevela.key`` (mode 600, no
passphrase). So the step resolves from this source tree, and the test needs no
secret manager. It needs an ``HF_TOKEN`` that can write ``GBTEST_HF_PUSH_ORG``
(default ibm-research).

Marked ``ibm`` (IBM-only cluster), extended suite only, and skipped in SPS CI/CD,
which has no key file. Run it by hand from the repo's IBM venv (``make ibm-venv``)::

    make -C steps/eval/unitxt/skypilot space
    cd steps/eval/unitxt/skypilot
    GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live PYTHONPATH=src \\
      ../../../../.venv/bin/python -m pytest -s -v test/slurm-bluevela

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
class TestSkypilotSlurmBlueVelaUnitxtGpu(
    HfPushCleanupMixin, AbstractYamlBuildRunnerTest
):
    """eval/unitxt step: hf mode on one GPU against a bound hf:// model input."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    # buildtest.yaml's timeout_minutes is 45, above pytest's global 1200s cap
    # (pyproject.toml), so give the runner 45 minutes plus 5 of headroom.
    @pytest.mark.timeout(45 * 60 + 300)
    def test_runner(self):
        super().test_runner()
