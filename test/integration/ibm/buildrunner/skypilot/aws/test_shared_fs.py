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

"""SkyPilot-on-AWS shared_filesystem (EFS): hfpull result reaches the step (#378).

The AWS/EFS analog of the sibling
``skypilot/slurm_bluevela/1step`` test (which uses a SLURM ``shared_workdir``): a
single ``command`` step with a REAL ``hf://`` input. buildrunner auto-queues a
hidden hfpull step for the non-env input, and under ``shared_filesystem`` that
hfpull runs on its OWN EC2 instance and caches the download onto the EFS-backed
per-run ``$GB_BUILD_WORKDIR``. SkyPilot places the ``command`` step on a DIFFERENT
EC2 instance, so it can only read the pulled input if the EFS mount carried it
across instances. The command runs ``test -e {{ bindings.hf_input.binding.path }}``
under ``set -eu`` (failing the build if the input did not arrive) and writes a
real file that the hf:// output's hfpush step uploads. Reaching SUCCESS proves the
feature's real intent: an hfpull (lhpull, etc.) result is available to the
referenced step across instances over EFS.

Why this and not a hand-written sentinel: ``env://`` I/O is a no-op (no transfer),
so it never exercises the assetstore pull path. A non-env input is what triggers
the hidden pull step whose output must land on the shared FS — that is what this
test drives.

Two fixtures exercise the two mount paths:
  * :class:`TestSkypilotAwsSharedFsBare` — ``command_config.image: ""`` runs on
    the bare EC2 instance; the EFS mount is read on the host.
  * :class:`TestSkypilotAwsSharedFsContainerized` — an image is set
    (``image_id: docker:<image>``) so the command runs INSIDE the container
    against the in-container NFS mount (SkyPilot's default SYS_ADMIN/--net=host/
    fuse) and the 1777/uid path.

Like the sibling aws build tests this is intentionally NOT marked ``ibm``: it
needs AWS credentials + SkyPilot, not the IBM cloud secret bundle the ``ibm``
marker's ``check_cloud_config()`` gate enforces. It auto-skips in CI and on
machines without AWS access.

It ADDITIONALLY needs, and self-skips without:
  * a real, pre-provisioned BYO EFS — gbserver never creates or destroys the
    filesystem. The fixture Space ships the documented PLACEHOLDER EFS
    ``file_system_id`` (read here, cloud-free), so a run never provisions against
    a bogus id until an operator points it at a real one.
  * an HF token — the hf:// input is pulled and the hf:// output is pushed to a
    personal HF namespace (which skips HF Enterprise resource groups), so
    ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) with write access to that
    namespace is required.

Prerequisites to actually run (locally, in the extended suite):
  1. AWS credentials configured (env vars or ``~/.aws/credentials``).
  2. SkyPilot installed and ``sky check aws`` passing.
  3. A pre-provisioned BYO EFS (mount targets per worker AZ, an SG allowing NFS
     2049, root chmod 1777 — see docs/environments/skypilot-aws.md), with its
     ``file_system_id``/``region`` written into the fixture Space's
     ``environments/skypilot/aws-shared-fs/environment.yaml``.
  4. ``HF_TOKEN`` with write access to the hf:// output namespace.

Each fixture's build.yaml, buildtest.yaml, and the shared test Space live under
the directory returned by ``_get_yaml_spec_dir`` below.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import pytest
import yaml
from libgbtest.aws_efs_noleak import list_gb_ephemeral, reap_leaked
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

# No-leak settle window: EFS/EC2 deletes are eventually consistent, so after
# teardown poll up to _NO_LEAK_SETTLE_S (every _NO_LEAK_POLL_S) for the tagged
# resources to disappear before declaring a leak.
_NO_LEAK_SETTLE_S = 180
_NO_LEAK_POLL_S = 10

# The documented placeholder file_system_id the fixture Space ships (mirrors the
# commented example in configurations/assets/environments/skypilot/aws/
# environment.yaml). A real run replaces it with a validated BYO EFS id; until
# then the tests self-skip.
_PLACEHOLDER_EFS_FS_ID = "fs-0abc123"

# The committed fixture Space's aws-shared-fs environment (its shared_filesystem
# efs block); the build.yaml fixtures reference it as
# space://environments/skypilot/aws-shared-fs.
_ENV_YAML = (
    get_test_data_dir_for(__file__)
    / "shared-fs"
    / "space"
    / "environments"
    / "skypilot"
    / "aws-shared-fs"
    / "environment.yaml"
)


def _aws_credentials_available() -> bool:
    """True if AWS credentials look configured (env vars or ~/.aws/credentials)."""
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    return (Path.home() / ".aws" / "credentials").is_file()


def _hf_token_available() -> bool:
    """True if an HF token is in the environment (the hf:// I/O needs write access)."""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _fixture_ships_placeholder_efs() -> bool:
    """True while the fixture environment.yaml still ships the placeholder EFS id.

    A cloud-free read of the committed shared_filesystem.efs block. Returns True
    (=> skip) whenever the file_system_id is the placeholder or cannot be read,
    so a run never provisions EC2/EFS against a bogus filesystem.
    """
    try:
        data = yaml.safe_load(_ENV_YAML.read_text(encoding="utf-8")) or {}
    except OSError:
        return True
    efs = (((data.get("config") or {}).get("shared_filesystem") or {}).get("efs")) or {}
    return efs.get("file_system_id", _PLACEHOLDER_EFS_FS_ID) == _PLACEHOLDER_EFS_FS_ID


# The committed ephemeral fixture Spaces' environments; read (cloud-free) to drive
# the no-leak check's boto3 session at the SAME region/profile the build
# provisions against. One single-mount fixture and one two-ephemeral-mount fixture.
_EPHEMERAL_ENV_YAML = (
    get_test_data_dir_for(__file__)
    / "ephemeral-efs"
    / "space"
    / "environments"
    / "skypilot"
    / "aws-ephemeral"
    / "environment.yaml"
)
_MULTI_EPHEMERAL_ENV_YAML = (
    get_test_data_dir_for(__file__)
    / "multi-ephemeral-efs"
    / "space"
    / "environments"
    / "skypilot"
    / "aws-multi-ephemeral"
    / "environment.yaml"
)


def _ephemeral_efs_env(env_yaml: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(region, aws_profile)`` for the first ephemeral mount in ``env_yaml``.

    Reads the committed environment.yaml (``shared_filesystem`` is a list; find the
    first ``provision: ephemeral`` mount) so the no-leak boto3 session targets the
    same region and AWS profile the build uses. All ephemeral mounts in a fixture
    share one region, so the first is representative. Profile falls back to
    ``$AWS_PROFILE``.
    """
    data = yaml.safe_load(env_yaml.read_text(encoding="utf-8")) or {}
    config = data.get("config") or {}
    sfs = config.get("shared_filesystem")
    mounts = sfs if isinstance(sfs, list) else [sfs] if sfs else []
    region = None
    for m in mounts:
        efs = (m or {}).get("efs") or {}
        if efs.get("provision") == "ephemeral":
            region = efs.get("region")
            break
    profile = (
        (((config.get("cloud_config") or {}).get("workspaces") or {}).get("default"))
        or {}
    ).get("aws", {}).get("profile") or os.environ.get("AWS_PROFILE")
    return region, profile


def _assert_ephemeral_no_leak(env_yaml: Path, run_build) -> None:
    """Run ``run_build`` and assert its teardown left no ``gb-ephemeral`` resources.

    Snapshots the ``gb-ephemeral=true`` EFS filesystems + NFS security groups in
    the fixture's region BEFORE the build, runs it (setup provisions, teardown
    deprovisions), then diffs AFTER. Deletes are eventually consistent, so poll a
    bounded settle window; any net-new resource that survives is a leak, which is
    best-effort reaped (so the failing test doesn't itself leave billable AWS
    resources) before failing. Shared by the single- and multi-mount ephemeral
    tests; covers every mount since the diff is by tag, not by count.
    """
    boto3 = pytest.importorskip("boto3")
    region, profile = _ephemeral_efs_env(env_yaml)
    assert region, f"no ephemeral efs region in {env_yaml}"
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()

    before_fs, before_sg = list_gb_ephemeral(session, region)
    run_build()  # provisions, runs to SUCCESS, then deprovisions

    deadline = time.monotonic() + _NO_LEAK_SETTLE_S
    while True:
        after_fs, after_sg = list_gb_ephemeral(session, region)
        leaked_fs = after_fs - before_fs
        leaked_sg = after_sg - before_sg
        if not leaked_fs and not leaked_sg:
            break
        if time.monotonic() >= deadline:
            reap_failures = reap_leaked(session, region, leaked_fs, leaked_sg)
            reap_note = (
                "reaped"
                if not reap_failures
                else "reap incomplete: " + "; ".join(reap_failures)
            )
            pytest.fail(
                "ephemeral EFS leaked after teardown in "
                f"{region}: filesystems={sorted(leaked_fs)} "
                f"security_groups={sorted(leaked_sg)} ({reap_note})"
            )
        time.sleep(_NO_LEAK_POLL_S)


# Real-infra build test (SkyPilot provisions EC2 instances) — only run in the
# extended suite (make extended-tests). Shares the same xdist group as the other
# AWS tests so concurrent AWS provisions don't race on SkyPilot's local state.
# These marks apply to EVERY test in the module (BYO and ephemeral alike).
pytestmark = [
    extended_testing_only,
    pytest.mark.xdist_group(name="buildtest_aws"),
    pytest.mark.skipif(
        not _aws_credentials_available(),
        reason="AWS credentials not configured (set AWS_ACCESS_KEY_ID/"
        "AWS_SECRET_ACCESS_KEY or provide ~/.aws/credentials); SkyPilot cannot "
        "provision an EC2 instance. Also requires `sky check aws` to pass.",
    ),
]

# BYO-only gates (the ephemeral test references no BYO EFS id and needs no HF
# token, so these must NOT gate it — they decorate the two BYO classes instead).
_skip_placeholder_efs = pytest.mark.skipif(
    _fixture_ships_placeholder_efs(),
    reason=(
        "fixture Space still ships the placeholder EFS id "
        f"({_PLACEHOLDER_EFS_FS_ID}); set shared_filesystem.efs "
        f"file_system_id/region in {_ENV_YAML} to a validated BYO EFS to run "
        "(see docs/environments/skypilot-aws.md)."
    ),
)
_skip_no_hf_token = pytest.mark.skipif(
    not _hf_token_available(),
    reason=(
        "no HF token in the environment (set HF_TOKEN or "
        "HUGGING_FACE_HUB_TOKEN with write access to the hf:// output "
        "namespace); the hf:// input is pulled and the output is pushed."
    ),
)


@_skip_placeholder_efs
@_skip_no_hf_token
class TestSkypilotAwsSharedFsBare(AbstractYamlBuildRunnerTest):
    """Bare EC2: the command step verifies the hf:// input the hidden hfpull step
    cached onto EFS (on a separate instance) is present, reading the mount on the
    host. SUCCESS proves the pulled input crossed instances over EFS."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "shared-fs" / "bare"


@_skip_placeholder_efs
@_skip_no_hf_token
class TestSkypilotAwsSharedFsContainerized(AbstractYamlBuildRunnerTest):
    """Containerized: the same hfpull -> command -> hfpush flow, but the command
    runs inside the container and reads the hf:// input over the in-container NFS
    mount + 1777/uid path."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "shared-fs" / "containerized"


class TestSkypilotAwsEphemeralEfs(AbstractYamlBuildRunnerTest):
    """Ephemeral auto-provisioned EFS (#391), producer -> consumer across instances.

    Unlike the BYO tests above, the ``aws-ephemeral`` environment's
    ``shared_filesystem`` sets ``provision: ephemeral``, so gbserver CREATES the EFS
    at target-run setup (boto3: filesystem + a mount target per subnet + an NFS
    security group), mounts it on every worker, and DESTROYS it at teardown. The
    2-step ``command`` build's producer writes ``probe.txt`` into the per-run
    ``$GB_BUILD_WORKDIR`` and the consumer -- on a SEPARATE EC2 instance -- reads it
    back under ``set -eu``; SUCCESS proves the auto-provisioned EFS carried state
    across instances. No BYO ``file_system_id`` is referenced, so (unlike the BYO
    tests) there is no placeholder-EFS skip, and no HF token is needed (pure
    command I/O).

    **No-leak.** ``test_runner`` is overridden to assert, over boto3, that the
    build's teardown left nothing behind: it snapshots the ``gb-ephemeral=true``
    EFS filesystems + NFS security groups in the region BEFORE the build, runs it
    to SUCCESS (setup provisions, teardown deprovisions), then diffs AFTER. Any
    net-new resource is a leak; the assertion best-effort reaps it (so the failing
    test does not itself leave billable AWS resources) and then fails naming the
    ``fsid``/``sg``. The before/after diff is reliable because the AWS build tests
    share one ``xdist_group`` (serialized, no concurrent ephemeral run). The
    deprovision path itself is also unit-tested in
    ``test/unit/environment/test_skypilot_teardown.py`` and the no-leak helpers in
    ``test/unit/environment/test_ephemeral_efs_noleak.py``. The two-mount pairing is
    covered by :class:`TestSkypilotAwsMultiEphemeralEfs` below.

    Gated like the BYO tests (``extended`` + AWS credentials; auto-skips in CI and
    without ``sky check aws``). Run it with ``AWS_PROFILE=gb-skypilot`` and
    ``PYTEST_ADDOPTS=-s``.
    """

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "ephemeral-efs" / "bare"

    def test_runner(self):
        """Run the producer->consumer build and assert nothing leaked afterward.

        Wraps the inherited real build with a before/after boto3 tag diff so a
        regression in the ephemeral EFS teardown/rollback surfaces as a test
        failure rather than silent billable orphans.
        """
        _assert_ephemeral_no_leak(_EPHEMERAL_ENV_YAML, super().test_runner)


class TestSkypilotAwsMultiEphemeralEfs(AbstractYamlBuildRunnerTest):
    """Two ephemeral auto-provisioned EFS mounts (#391/#422) -- the "multiple mounts
    + ephemeral EFS" pairing the PR headlines.

    The ``aws-multi-ephemeral`` environment lists TWO ``provision: ephemeral``
    mounts (distinct mount_points), so gbserver creates a SEPARATE EFS + NFS
    security group per mount at setup. This is the exact case that regressed: both
    mounts share one ``gb-targetrun-id``, so without a per-mount SG name the second
    ``create_security_group`` collided on ``InvalidGroup.Duplicate`` and the whole
    setup failed. The producer writes a probe onto BOTH mounts (``probe.txt`` in the
    per-run ``$GB_BUILD_WORKDIR`` on mount A, an absolute file on mount B); the
    consumer -- on a SEPARATE instance -- reads BOTH back under ``set -eu``. SUCCESS
    proves the two ephemeral EFS were provisioned together (no SG collision) and
    each carried state across instances.

    **No-leak.** ``test_runner`` wraps the build in the same before/after boto3 tag
    diff as the single-mount test; because the diff is by ``gb-ephemeral`` tag it
    covers both filesystems and both security groups. Gated identically (``extended``
    + AWS credentials; run with ``AWS_PROFILE=gb-skypilot`` and ``PYTEST_ADDOPTS=-s``).
    """

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "multi-ephemeral-efs" / "bare"

    def test_runner(self):
        """Run the two-mount producer->consumer build and assert no leak afterward."""
        _assert_ephemeral_no_leak(_MULTI_EPHEMERAL_ENV_YAML, super().test_runner)


@_skip_no_hf_token
class TestSkypilotAwsEphemeralEfsHfIO(AbstractYamlBuildRunnerTest):
    """Full E2E (#391/#422): the realistic hfpull -> EFS -> command -> hfpush data
    path on an AUTO-PROVISIONED ephemeral EFS -- the production use of the feature,
    vs the probe-file smokes above.

    The ``aws-ephemeral`` environment auto-provisions the EFS at setup. A real
    hf:// input is pulled onto it on its own EC2 instance; the ``command`` step --
    on a SEPARATE instance -- verifies the pulled input is present over the mount
    (``test -e`` under ``set -eu``) and writes a file; an hf:// output is pushed
    from a THIRD instance; teardown deprovisions the filesystem. SUCCESS proves the
    auto-provisioned EFS carried a real assetstore pull/push across instances.

    Unlike the BYO ``shared-fs`` hf tests this references no ``file_system_id`` (so
    no placeholder-EFS skip), but it DOES need an HF token for the pull/push
    (``@_skip_no_hf_token``). No-leak is asserted like the sibling ephemeral tests
    (same ``aws-ephemeral`` env -> ``_EPHEMERAL_ENV_YAML``). Gated ``extended`` +
    AWS credentials; run with ``AWS_PROFILE=gb-skypilot`` and ``PYTEST_ADDOPTS=-s``.
    """

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "ephemeral-efs" / "hf"

    def test_runner(self):
        """Run the hfpull->EFS->command->hfpush build and assert no leak afterward."""
        _assert_ephemeral_no_leak(_EPHEMERAL_ENV_YAML, super().test_runner)
