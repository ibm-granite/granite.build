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

"""PII redaction on skypilot/aws (real EC2), through the same `dpk` step.

The aws counterpart of ``test/slurm-pii/test_skypilot_slurm_dpk_pii.py`` and the
generality proof on aws: the sibling ``aws-tok`` already shows tokenization working
on EC2, and this runs an unrelated DPK transform — ``pii_redactor``, which shares no
code, flags, or dependencies with tokenization — with the ONLY differences living in
the build.yaml (``transform``, ``args``, artifact names). ``step-template.yaml`` is
untouched, so reaching SUCCESS demonstrates the module and pip-extra derivations hold
for a second transform on aws:

* module — ``transform: pii_redactor`` -> ``python -m dpk_pii_redactor.runtime``
* pip    — -> ``data-prep-toolkit-transforms[pii-redactor]==<dpk_version>``

**This is the slow one.** The ``[pii-redactor]`` extra resolves to ~125 packages
(torch, flair, presidio) and the transform downloads a flair NER model on first use,
so the fixture allows a generous timeout. Sequential (no ``runtime_num_processors``):
each worker would load its own flair + presidio, so a pool multiplies peak memory;
the parallel path is covered by ``aws-tok``.

**Real EC2 — never runs by accident.** Extended-suite only AND skips unless AWS
credentials are present (:func:`gbserver.environment.skypilot.aws_credentials_present`).
The ``hf://`` input is public, so no HF_TOKEN is needed.
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
class TestSkypilotAwsDpkPii(AbstractYamlBuildRunnerTest):
    """dpk step, second transform on aws: pii_redactor with no step change."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
