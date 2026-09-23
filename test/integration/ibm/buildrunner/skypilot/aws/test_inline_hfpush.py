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

"""SkyPilot-on-AWS inline hfpush: single-instance pull -> compute -> push (#390).

The no-shared-filesystem counterpart of the sibling ``test_shared_fs.py``. That
test proves an hf:// input/output flows ACROSS instances over EFS; this one
proves the opposite arrangement — that WITHOUT any ``shared_filesystem`` the
whole build runs on ONE EC2 instance, no admin-provisioned EFS required.

A single ``command`` step with a REAL ``hf://`` input and ``hf://`` output on an
environment whose hf assetstore is configured ``inline: true`` for BOTH pull and
push (and has NO ``shared_filesystem`` block). Under inline mode buildrunner does
NOT queue separate hfpull/hfpush steps:
  * inline PULL folds the ``hf download`` into the command step's setup (the
    model is cached instance-locally, there is no shared mount to stage through);
  * inline PUSH folds the upload into the command step's run epilogue, keyed off
    the ``GB_ARTIFACT_PATH`` marker the command emits.
So the target runs as a SINGLE step on a SINGLE instance (``step_count: 1``): the
command reads the inline-pulled input (``test -e`` under ``set -eu`` fails the
build if it did not arrive), writes a real file, and emits the artifact marker
the inline push epilogue uploads. Reaching SUCCESS proves the inline
pull->compute->push path works with no shared FS and no admin (a push to an
individual user namespace skips HF Enterprise resource groups).

Like the sibling aws build tests this is intentionally NOT marked ``ibm``: it
needs AWS credentials + SkyPilot, not the IBM cloud secret bundle the ``ibm``
marker's ``check_cloud_config()`` gate enforces. It auto-skips in CI and on
machines without AWS access.

It ADDITIONALLY needs, and self-skips without, an HF token: the hf:// input is
pulled and the hf:// output is pushed to a personal HF namespace (which skips HF
Enterprise resource groups), so ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``)
with write access to that namespace is required. There is NO EFS placeholder gate
here — the inline path uses no shared filesystem.

Prerequisites to actually run (locally, in the extended suite):
  1. AWS credentials configured (env vars or ``~/.aws/credentials``).
  2. SkyPilot installed and ``sky check aws`` passing.
  3. ``HF_TOKEN`` with write access to the hf:// output namespace.

The build.yaml, buildtest.yaml, and the co-located test Space live under the
directory returned by ``_get_yaml_spec_dir`` below.
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


def _hf_token_available() -> bool:
    """True if an HF token is in the environment (the hf:// I/O needs write access)."""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


# Real-infra build test (SkyPilot provisions ONE EC2 instance; no EFS) — only run
# in the extended suite (make extended-tests), and only with AWS creds + an HF
# write token. Shares the AWS xdist group with the other AWS tests so concurrent
# AWS provisions don't race on SkyPilot's local state. No EFS placeholder gate:
# the inline path uses no shared_filesystem.
pytestmark = [
    extended_testing_only,
    pytest.mark.xdist_group(name="buildtest_aws"),
    pytest.mark.skipif(
        not _aws_credentials_available(),
        reason="AWS credentials not configured (set AWS_ACCESS_KEY_ID/"
        "AWS_SECRET_ACCESS_KEY or provide ~/.aws/credentials); SkyPilot cannot "
        "provision an EC2 instance. Also requires `sky check aws` to pass.",
    ),
    pytest.mark.skipif(
        not _hf_token_available(),
        reason=(
            "no HF token in the environment (set HF_TOKEN or "
            "HUGGING_FACE_HUB_TOKEN with write access to the hf:// output "
            "namespace); the hf:// input is pulled and the output is pushed."
        ),
    ),
]


class TestSkypilotAwsInlineHfpush(AbstractYamlBuildRunnerTest):
    """Single EC2 instance: inline hf pull -> command -> inline hf push, no shared
    FS. The command reads the inline-pulled hf:// input (folded into its setup),
    writes a file, and emits the artifact marker the inline push epilogue uploads.
    SUCCESS proves the no-shared-FS, no-admin inline pull->compute->push path."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "inline-hfpush"
