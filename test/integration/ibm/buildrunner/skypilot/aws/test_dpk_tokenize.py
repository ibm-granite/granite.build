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

"""End-to-end DPK tokenization + validation on AWS EC2 (via SkyPilot).

Runs the DPK_Tokenize_Skypilot template's two-target pipeline on a real EC2
backend: `tokenize` installs Data Prep Kit from PyPI and runs
``dpk_tokenization2arrow.runtime`` over the committed sample Parquet, then
`validate` binds the tokenize output and runs ``validate_tokens.py`` against it,
failing the build on any inconsistency. Reaching SUCCESS proves the template runs
end to end on skypilot/aws AND that validate's real checks pass on the known-good
fixture (3 arrow files, 6 documents, 85 tokens).

IMPORTANT — cross-node artifact handoff:
    Unlike the sibling aws/2target fixture (whose second target only ECHOES the
    bound path), `validate` READS the bound path. On AWS each target provisions
    its own EC2 instance with no shared filesystem, and ``env://`` moves no bytes
    — so the shipped template's ``env:///tmp/...`` handoff would leave `validate`
    with an absent path (confirmed: produce=cN / consume=cM on a multi-node
    SLURM run fails with "No such file or directory"). This fixture therefore
    routes the `tokens` artifact through the S3 assetstore (``s3://``): tokenize
    pushes it, validate pulls it locally, so the bound path resolves on the
    validate node regardless of placement. See the fixture build.yaml.

Like the sibling aws build tests this is intentionally NOT marked ``ibm``: it
needs AWS credentials + SkyPilot, not the IBM cloud secret bundle that the
``ibm`` marker's ``check_cloud_config()`` gate enforces. It is gated on AWS
credentials being present, so it auto-skips in CI and on machines without AWS
access.

Prerequisites to actually run (locally, in the extended suite):
  1. AWS credentials configured (env vars or ``~/.aws/credentials``).
  2. SkyPilot installed and ``sky check aws`` passing.
  3. Outbound PyPI reachable from the launched instance (the step pip-installs
     DPK, transformers, and pyarrow at runtime).
  4. An S3 bucket writable with the same AWS credentials, provided to the build
     via the ``S3_TEST_BUCKET`` space variable, and the S3 assetstore secrets
     (COS_ACCESS_KEY_ID / COS_SECRET_ACCESS_KEY) available to gbserver.

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


def _aws_credentials_available() -> bool:
    """True if AWS credentials look configured (env vars or ~/.aws/credentials)."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    return (Path.home() / ".aws" / "credentials").is_file()


# Real-infra build test (SkyPilot provisions EC2 instances) — only run in the
# extended suite (make extended-tests), not the fast quick-tests suite. Shares the
# same xdist group as the sibling AWS tests so concurrent AWS provisions don't race
# on SkyPilot's local state.
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_aws")
@pytest.mark.skipif(
    not _aws_credentials_available(),
    reason="AWS credentials not configured (set AWS_ACCESS_KEY_ID/"
    "AWS_SECRET_ACCESS_KEY or provide ~/.aws/credentials); SkyPilot cannot "
    "provision an EC2 instance. Also requires `sky check aws` to pass, outbound "
    "PyPI on the instance, and a writable S3 bucket for the tokens handoff.",
)
class TestSkypilotAwsDpkTokenize(AbstractYamlBuildRunnerTest):
    """DPK tokenize -> validate on AWS EC2 via SkyPilot; validate binds tokenize's S3 output."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "dpk-tokenize"
