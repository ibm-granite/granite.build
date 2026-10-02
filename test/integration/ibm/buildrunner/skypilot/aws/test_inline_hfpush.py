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

"""SkyPilot-on-AWS inline-pull + dispatched hfpush over a shared FS (#390).

A sibling of ``test_shared_fs.py``. Both prove an hf:// artifact flows ACROSS EC2
instances over a ``shared_filesystem`` (BYO EFS); the difference is the INPUT
side: there the hf pull runs as its own hfpull step, here the hf pull is
``inline: true`` so its download folds into the command step's setup (no separate
hfpull step). The PUSH side is the same in both after #390: hf push is ALWAYS a
DISPATCHED real step (the single-instance inline-push fold is gone), placed by
SkyPilot on its OWN EC2 instance, which reads the produced artifact back over the
EFS-backed shared workdir and uploads it to HF.

A single ``command`` step with a REAL ``hf://`` input and ``hf://`` output(s) on
the ``aws-inline`` environment (hf assetstore: ``inline: true`` PULL, dispatched
PUSH, with a ``shared_filesystem`` EFS block). So a single-output target runs as
TWO steps across TWO instances:
  1. the command step (inline pull folded into its setup; ``test -e`` under
     ``set -eu`` fails the build if the input did not arrive; it writes a real
     file under ``$pwd/out`` on the shared workdir and emits the ``GB_ARTIFACT``
     marker), then
  2. a dispatched hfpush step on a separate instance that reads ``$pwd/out`` back
     over EFS and uploads it.
Reaching SUCCESS proves the inline-pull -> compute -> dispatched-push (over
shared FS) path. Each emitted artifact adds one hfpush step, so the two-output
variant runs as THREE steps (command + hfpush x2).

Like the sibling aws build tests this is intentionally NOT marked ``ibm``: it
needs AWS credentials + SkyPilot, not the IBM cloud secret bundle the ``ibm``
marker's ``check_cloud_config()`` gate enforces. It auto-skips in CI and on
machines without AWS access.

It ADDITIONALLY needs, and self-skips without, an HF token: the hf:// input is
pulled and the hf:// output is pushed to a personal HF namespace (which skips HF
Enterprise resource groups), so ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``)
with write access to that namespace is required. The push additionally requires a
real EFS: the committed ``aws-inline`` environment ships a STAND-IN
``file_system_id`` (``fs-0abc123``), so a live run must point it at a real
pre-provisioned EFS (see that environment.yaml's header + #391 for the future
auto-provisioned path).

Prerequisites to actually run (locally, in the extended suite):
  1. AWS credentials configured (env vars or ``~/.aws/credentials``).
  2. SkyPilot installed and ``sky check aws`` passing.
  3. ``HF_TOKEN`` with write access to the hf:// output namespace.
  4. a real ``shared_filesystem`` EFS id in the ``aws-inline`` environment.

Four fixtures, each with its build.yaml + buildtest.yaml under
``inline-hfpush/<variant>/`` and sharing the one test Space at
``inline-hfpush/space`` (resolved via each buildtest.yaml's ``space_uri:
../space``):
  * :class:`TestSkypilotAwsInlineHfpushBare` — ``command_config.image: ""`` runs
    the command on the bare EC2 VM; command + dispatched hfpush = 2 steps.
  * :class:`TestSkypilotAwsInlineHfpushContainerized` — an image is set
    (``image_id: docker:python:3.12-slim``) so the command's setup+run (incl. the
    inline pull) execute inside the container; command + dispatched hfpush = 2.
  * :class:`TestSkypilotAwsInlineHfpushTwoOutput` — two hf:// outputs, so two
    dispatched hfpush steps (one per output); command + hfpush x2 = 3. Proves the
    multi-output push path (#390 §4.5) lands BOTH artifacts.
  * :class:`TestSkypilotAwsInlineHfpushRetry` — same 2-step topology run under the
    framework's step-failure simulation (buildtest.yaml
    ``simulate_step_failure: true``): the environment injects one simulated step
    failure and the RetryHandler absorbs it and retries in process, so the build
    still reaches SUCCESS. Proves the dispatched-push topology survives a step
    failure + in-process retry (#390 §4.6).

Harness limitation (deferred assertion). The brief's retry case wanted to assert
"only the push re-ran, the producer was not re-run"
(``producer_run_count == 1 and push_run_count == 2``). The YAML-driven harness
(``AbstractYamlBuildRunnerTest`` + ``buildtest.yaml``) exposes only per-TARGET
observables — ``step_count``, input/output artifact counts, ``target_failure_count``
(FAILED target runs), ``jobstats_count`` — plus optional per-step
metadata/config assertions; it does NOT expose per-step RUN counts or which step
a retry re-ran, and the framework step-failure simulation is absorbed within the
same target run (so it records no FAILED target run to count). There is therefore
no observable signal for "only the push re-ran". The retry fixture asserts the
closest available signal instead: the build reaches SUCCESS through a simulated
step failure + retry with the dispatched-push step count. See the task report.
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


# Real-infra build test (SkyPilot provisions the command instance PLUS a separate
# dispatched-push instance over a shared EFS) — only run in the extended suite
# (make extended-tests), and only with AWS creds + an HF write token. Shares the
# AWS xdist group with the other AWS tests so concurrent AWS provisions don't race
# on SkyPilot's local state. The push additionally requires a real EFS: the
# committed aws-inline environment ships a STAND-IN file_system_id, so a live run
# must point it at a real pre-provisioned EFS (see that environment.yaml + #391).
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


class TestSkypilotAwsInlineHfpushBare(AbstractYamlBuildRunnerTest):
    """BARE (command_config.image empty): inline hf pull -> command -> DISPATCHED
    hf push over a shared EFS. The command runs on the bare EC2 VM, reads the
    inline-pulled hf:// input (folded into its setup), writes a file to the shared
    workdir, and emits the artifact marker; a separate dispatched hfpush step on
    its own instance reads that file back over EFS and uploads it. SUCCESS proves
    the inline-pull -> compute -> dispatched-push (over shared FS) path on the bare
    host (command + hfpush = 2 steps)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "inline-hfpush" / "bare"


class TestSkypilotAwsInlineHfpushContainerized(AbstractYamlBuildRunnerTest):
    """CONTAINERIZED (image set -> image_id: docker:python:3.12-slim): the same
    flow, but the command's setup+run execute INSIDE the container. Exercises the
    container-boundary path the bare variant does not: the inline pull downloads
    into the container fs (the image supplies python3 + pip for the pull's
    huggingface_hub[cli]). The push is still a separate DISPATCHED step over EFS.
    SUCCESS proves inline-pull (in-container) -> compute -> dispatched-push works
    (command + hfpush = 2 steps)."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "inline-hfpush" / "containerized"


class TestSkypilotAwsInlineHfpushTwoOutput(AbstractYamlBuildRunnerTest):
    """TWO-OUTPUT (#390 §4.5): one command step with an inline hf:// input and TWO
    hf:// outputs. The command emits two GB_ARTIFACT markers, so buildrunner queues
    TWO dispatched hfpush steps (one per output), each on its own instance reading
    its artifact back over EFS. SUCCESS with output_artifact_count 2 proves BOTH
    artifacts push (command + hfpush x2 = 3 steps) — the multi-output push path the
    plan guards against."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "inline-hfpush" / "two-output"


class TestSkypilotAwsInlineHfpushRetry(AbstractYamlBuildRunnerTest):
    """RETRY (#390 §4.6): the same 2-step inline-pull -> command -> dispatched-push
    topology run under the framework's step-failure simulation
    (buildtest.yaml simulate_step_failure: true). The environment injects one
    simulated step failure and the RetryHandler absorbs it and retries in process,
    so the build still reaches SUCCESS. Proves the dispatched-over-shared-FS
    topology survives a step failure + in-process retry.

    Deferred assertion: the brief wanted "only the push re-ran, the producer was
    not re-run" (producer_run_count == 1 and push_run_count == 2). The harness
    exposes no per-step run count and the simulated retry is absorbed within the
    same target run (no FAILED target run to count), so that signal is not
    observable; this fixture asserts the closest available one — SUCCESS through a
    simulated failure + retry at the dispatched-push step count. See the module
    docstring and the task report."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "inline-hfpush" / "retry"
