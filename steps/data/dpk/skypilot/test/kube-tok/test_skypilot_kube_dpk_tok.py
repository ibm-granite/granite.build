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

"""DPK tokenization with in-step validation in a pod on skypilot/kubernetes.

The kube counterpart of ``test/aws-tok/test_skypilot_aws_dpk_tok.py``. Same single
target — ``transform: tokenization2arrow`` with ``validate: true`` — run in a pod
on a local kind cluster. Only the environment
(``space://environments/skypilot/kubernetes/local``) and the skip gate differ; the step
(``space://steps/data/dpk``) and the transform config are the same, so reaching SUCCESS
proves the derivations and the in-step validator hook hold on kubernetes too.

The env mounts a shared volume at /shared in every pod (its shared_workdir), so —
as on slurm — the ``hf://`` input is pulled by a separate hfpull step into
/shared/hf_cache, and the bundled ``src/`` reaches the pod via file_mounts. The
``tokens`` output is an ``hf://`` dataset pushed by a separate hfpush pod, which
reads it from the same volume: :class:`libgbtest.hfpush.HfPushCleanupMixin` names it
per run in ``GBTEST_HF_PUSH_ORG``, checks it landed on the Hub, and deletes it, so
the run needs an ``HF_TOKEN`` that can write that org (the input stays public).

**Never runs against a shared cluster by accident.** Extended-suite only AND skips
unless the local kind cluster and its shared volume are up
(:func:`libgbtest.kube.kind_cluster_reachable`; bring them up with
``make test-setup-kube``); the env pins SkyPilot to the kind context.

Fixtures (build.yaml/buildtest.yaml) live in the ``test-data/`` dir mirroring this
file, resolved by the repo's ``test/`` <-> ``test-data/`` helper so the pairing holds
in both test modes (see steps/README.md).
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.hfpush import HfPushCleanupMixin
from libgbtest.kube import (
    KUBE_XDIST_GROUP,
    kind_cluster_reachable,
    kind_skip_reason,
)

pytestmark = pytest.mark.skypilot_integration


@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKubeDpkTok(HfPushCleanupMixin, AbstractYamlBuildRunnerTest):
    """dpk step: tokenization with in-step validation, end to end in a kube pod."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
