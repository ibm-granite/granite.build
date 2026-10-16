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

"""DPK tokenization with in-step validation on skypilot/aws (real EC2).

The aws counterpart of ``test/slurm-tok/test_skypilot_slurm_dpk_tok.py``. Same
single target — ``transform: tokenization2arrow`` with ``validate: true`` — proven
on a real EC2 instance instead of the local Docker SLURM cluster. Only the
environment (``space://environments/skypilot/aws``) and the credential gate differ;
the step (``space://steps/dpk``) and the transform config are the same, so reaching
SUCCESS proves the derivations and the in-step validator hook hold on aws too.

**Real EC2 — never runs by accident.** It is extended-suite only AND skips unless
AWS credentials are present (:func:`gbserver.environment.skypilot.aws_credentials_present`),
so no instance is provisioned without credentials explicitly exported. The ``hf://``
input is public, so no HF_TOKEN is needed — the launcher's inline ``hf download`` runs
anonymously.

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

from gbserver.environment.skypilot import aws_credentials_present

pytestmark = pytest.mark.skypilot_integration


@extended_testing_only
@pytest.mark.skipif(
    not aws_credentials_present(),
    reason=(
        "AWS credentials not in environment "
        "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)"
    ),
)
class TestSkypilotAwsDpkTok(AbstractYamlBuildRunnerTest):
    """dpk step: tokenization with in-step validation, end to end on aws (EC2)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
