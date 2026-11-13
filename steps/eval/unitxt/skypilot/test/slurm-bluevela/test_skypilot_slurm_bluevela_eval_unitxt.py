# Copyright Granite.Build Authors
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

It uses the step's local space and the in-repo environment
``space://environments/skypilot/slurm/bluevela``, so the step resolves from this
source tree. That environment logs in as the shared functional ID with the key in
the ``BV_SSH_PRIVATE_KEY`` space secret, created once in the local space from
``~/.ssh/ibm-bluevela.key``
(``gb secret create BV_SSH_PRIVATE_KEY --from-file ~/.ssh/ibm-bluevela.key``). It
also needs an ``HF_TOKEN`` that can write ``GBTEST_HF_PUSH_ORG`` (default
ibm-research).

IBM infrastructure through the local space, so marked ``ibm``; extended suite only.
Skips unless the local space holds the ``BV_SSH_PRIVATE_KEY`` secret (the environment
authenticates with it, not with the key file), and in SPS CI/CD. Run it by hand from the repo's IBM venv
(``make ibm-venv``)::

    make -C steps/eval/unitxt/skypilot space
    cd steps/eval/unitxt/skypilot
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

# The environment authenticates with this space secret (cluster_ssh_configs
# IdentityKey), not with a key file, so the test skips when the secret is missing.
_SECRET = "BV_SSH_PRIVATE_KEY"
# The local space the test runs in: its space.yaml `name`, which picks the local
# secrets directory (~/.granite.build/space_secrets/<name>/).
_SPACE_NAME = "public"


def _bluevela_secret_present() -> bool:
    """Whether the local space can resolve the environment's SSH key secret."""
    from gbserver.spacesecretmanager.localspacesecretmanager import (
        LocalSpaceSecretManager,
    )

    try:
        manager = LocalSpaceSecretManager(uri="local", space_name=_SPACE_NAME)
        return bool(manager.get_secrets().get(_SECRET))
    except Exception:  # pylint: disable=broad-except  # an unreadable store == absent
        return False


@extended_testing_only
# No xdist_group: skypilot_integration already puts it in the serialized
# "skypilot" group, and under --dist=loadgroup any other group is a UsageError.
@pytest.mark.skipif(
    not _bluevela_secret_present(),
    reason=(
        f"no {_SECRET} secret in the local '{_SPACE_NAME}' space "
        f"(gb secret create {_SECRET} --from-file ~/.ssh/ibm-bluevela.key)"
    ),
)
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="No BlueVela SSH key on SPS CI/CD hosts",
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
