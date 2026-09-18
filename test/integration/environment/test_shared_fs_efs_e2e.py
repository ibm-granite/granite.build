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

"""Gated real-infra E2E: shared_filesystem (EFS) producer -> consumer (#378).

This is the OPT-IN, skip-GATED real-AWS proof for the ``shared_filesystem`` EFS
provider (issue #378). It provisions real EC2 instances (via SkyPilot) plus talks
to a real, pre-provisioned EFS filesystem, so it COSTS MONEY and MUST NEVER run in
normal CI. It self-skips unless every one of these is set:

* ``GB_RUN_SHARED_FS_E2E=1``   -- the explicit opt-in gate (spins up EC2/EFS; $).
* ``GB_TEST_EFS_FS_ID``        -- a validated BYO EFS filesystem id (fs-...).
* ``GB_TEST_EFS_REGION``       -- that filesystem's region (e.g. us-east-1).
* AWS credentials in the environment (``aws_credentials_present()``), matching the
  gate the shipped byoc/dpk skypilot-aws step tests use so no instance is ever
  provisioned by accident.

**What it proves.** Two steps in ONE target both start with CWD ``$GB_BUILD_WORKDIR``
(the per-target-run prefix ``${mount_point}/builds/<build_id>/runs/<targetrun_id>/``
that the ``shared_filesystem`` provider mounts on EFS and ``chmod 1777``s). SkyPilot
allocates a SEPARATE EC2 instance per step, so:

* ``test_efs_producer_consumer_bare`` -- step 1 writes a sentinel under
  ``$GB_BUILD_WORKDIR``; step 2, on a different instance, reads it back. A SUCCESS
  proves the EFS mount carried step 1's bytes across instances, and teardown
  removed the per-run prefix.
* ``test_efs_producer_consumer_containerized`` -- the same 2-step flow but each
  step sets a container image (rendered to SkyPilot ``image_id: docker:<image>``),
  exercising the in-container NFS mount (SkyPilot's default SYS_ADMIN / --net=host
  / fuse) and the 1777/uid path from inside the container.

**Harness pattern.** The shipped SkyPilot step tests
(``steps/byoc/skypilot/test/aws/test_skypilot_aws_byoc.py``,
``steps/dpk/skypilot/test/aws-tok/test_skypilot_aws_dpk_tok.py``) drive a real
build through :class:`libgbtest.buildrunner.buildtest.AbstractYamlBuildRunnerTest`
with a static ``build.yaml`` + ``buildtest.yaml`` fixture pair resolved from a
rendered Space. That fixture-tree pattern is a good fit for a step's own config,
but it can't carry the runtime EFS coordinates (``GB_TEST_EFS_FS_ID`` /
``GB_TEST_EFS_REGION``) that this test must read from the environment. So this
module instead MATERIALIZES the same three spec files (environment.yaml with the
``shared_filesystem`` efs block, a 2-step build.yaml, and a buildtest.yaml) at
runtime from the fixture env vars, then hands them to the SAME build-submission
harness. The concrete file CONTENTS and the exact assertions are complete below;
the final Space-render + submit call is left as a single clearly-marked scaffold
(:func:`_submit_and_wait`) so this file COLLECTS cleanly and SKIPS with the gate
off -- see that function's TODO before running for real.

Run it (opt-in, real AWS) exactly like the other skypilot-aws step tests -- the
``-s`` is required (otherwise click sees a bad fd):

    GB_RUN_SHARED_FS_E2E=1 \\
    GB_TEST_EFS_FS_ID=fs-03bbdc96a5fdcb873 \\
    GB_TEST_EFS_REGION=us-east-1 \\
    AWS_PROFILE=gb-skypilot \\
    PYTEST_ADDOPTS=-s \\
    .venv/bin/python -m pytest \\
      test/integration/environment/test_shared_fs_efs_e2e.py -q

Confirm the gate skips in a normal run (this is what CI does):

    .venv/bin/python -m pytest \\
      test/integration/environment/test_shared_fs_efs_e2e.py -q
    # -> 2 skipped
"""

import os
import textwrap
from pathlib import Path

import pytest

# --- Skip gate -------------------------------------------------------------
#
# NOTE: everything below the gate (importing the harness, provisioning AWS)
# lives inside the test bodies, so a normal collection run never touches AWS or
# any optional harness import -- the module imports cleanly and both tests just
# report SKIPPED. The gate is intentionally the explicit opt-in env var; the
# EFS-coordinate and AWS-credential checks are asserted at the top of each test
# body (with a clear pytest.skip) so a mis-set opt-in fails loudly rather than
# silently provisioning nothing.
_GATE_ENV = "GB_RUN_SHARED_FS_E2E"

pytestmark = [
    pytest.mark.skypilot_integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get(_GATE_ENV) != "1",
        reason=(
            f"real-infra E2E: set {_GATE_ENV}=1 (spins up EC2/EFS; costs $). "
            "Also requires GB_TEST_EFS_FS_ID, GB_TEST_EFS_REGION and AWS creds."
        ),
    ),
]

# Where step 1 writes / step 2 reads, relative to $GB_BUILD_WORKDIR (the CWD both
# steps start in). Kept in one place so the producer command, the consumer
# command, and the post-run assertion all agree.
_SENTINEL_NAME = "sentinel"
_SENTINEL_VALUE = "hello-from-efs-378"

# The EFS mount point the provider mounts on each instance (matches the shipped
# docs/environment example, docs/environments/skypilot-aws.md).
_MOUNT_POINT = "/mnt/gb-shared"

# Container image for the containerized variant. A small public image that ships
# an NFS client-capable base; byoc renders `image_id: docker:<image>`.
_CONTAINER_IMAGE = "debian:12-slim"


def _require_efs_fixture() -> tuple[str, str]:
    """Return (fs_id, region) from the fixture env vars, or skip.

    Called at the top of each test body (past the opt-in gate) so a run that
    sets GB_RUN_SHARED_FS_E2E=1 but forgets the EFS coordinates skips with a
    precise reason instead of trying to build an environment.yaml with a
    ``None`` file_system_id.
    """
    fs_id = os.environ.get("GB_TEST_EFS_FS_ID")
    region = os.environ.get("GB_TEST_EFS_REGION")
    if not fs_id or not region:
        pytest.skip(
            "shared-fs E2E opted in but GB_TEST_EFS_FS_ID / GB_TEST_EFS_REGION "
            "are not both set (need a validated BYO EFS fixture)."
        )
    return fs_id, region


def _require_aws_credentials() -> None:
    """Skip unless AWS creds are present, reusing the shipped gbserver predicate.

    Mirrors the credential gate on the shipped skypilot-aws step tests
    (steps/byoc|dpk/skypilot/test/aws*) so this test can never provision EC2
    without credentials explicitly exported. Imported lazily so a normal
    (skipped) collection never imports gbserver.environment.skypilot.
    """
    from gbserver.environment.skypilot import aws_credentials_present

    if not aws_credentials_present():
        pytest.skip(
            "AWS credentials not in environment "
            "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)."
        )


def _environment_yaml(fs_id: str, region: str) -> str:
    """Render a skypilot/aws environment.yaml with the shared_filesystem efs block.

    This is the environment-level knob issue #378 adds: setting
    ``shared_filesystem`` PRODUCES the ``shared_workdir`` root (do NOT also set
    ``shared_workdir``). The efs block carries the runtime fixture coordinates —
    ``file_system_id`` + ``region`` — which is why the environment must be
    materialized here rather than committed as a static fixture. Matches
    :class:`gbserver.environment.shared_fs.config.SharedFilesystemConfig` /
    ``EfsConfig`` (provider=efs, absolute mount_point, efs block required) and the
    commented example in configurations/assets/environments/skypilot/aws/environment.yaml.
    """
    return textwrap.dedent(f"""\
        name: skypilot-aws-shared-fs-e2e
        type: Skypilot
        subtype: aws
        config:
          default_cloud: aws
          idle_minutes_to_autostop: 5
          aws_credentials:
            - profile: gb-skypilot
              aws_access_key_id: GB_AWS_ACCESS_KEY_ID
              aws_secret_access_key: GB_AWS_SECRET_ACCESS_KEY
          cloud_config:
            workspaces:
              default:
                aws:
                  profile: gb-skypilot
          # Issue #378: PRODUCES the shared_workdir root from a BYO EFS.
          shared_filesystem:
            provider: efs
            mount_point: {_MOUNT_POINT}
            efs:
              file_system_id: {fs_id}
              region: {region}
              tls: true
        assetstores:
          - store_uri: space://assetstores/s3/
            pull:
              - mode: default
                config: {{}}
            push:
              - mode: default
                config: {{}}
        """)


def _build_yaml(image: str = "") -> str:
    """Render the 2-step producer->consumer build.yaml.

    Both steps run the byoc step (space://steps/byoc) on the shared-fs aws
    environment. byoc runs a bare command from ``$GB_BUILD_WORKDIR`` (its CWD),
    so:

      * producer: ``echo <value> > $GB_BUILD_WORKDIR/<sentinel>``
      * consumer: ``test "$(cat $GB_BUILD_WORKDIR/<sentinel>)" = <value>``  (on a
        SEPARATE EC2 instance) -- SUCCESS proves the byte crossed instances on EFS.

    ``image`` empty  -> bare EC2 node (byoc renders image_id: "").
    ``image`` set    -> byoc renders ``image_id: docker:<image>``, so the same
                        commands run INSIDE the container against the in-container
                        NFS mount (the containerized variant).

    byoc requires a non-empty ``repo`` (it git-clones during setup), so we point
    both steps at the canonical tiny public repo used by the shipped byoc aws test
    (octocat/Hello-World); the clone is incidental — the sentinel logic uses the
    absolute ``$GB_BUILD_WORKDIR`` path, independent of the cloned ``code/`` dir.
    """
    producer_cmd = (
        f'echo "{_SENTINEL_VALUE}" > "$GB_BUILD_WORKDIR/{_SENTINEL_NAME}"; '
        f'echo "wrote sentinel to $GB_BUILD_WORKDIR/{_SENTINEL_NAME}"; '
        'echo "GB_ARTIFACT_ID:produced GB_ARTIFACT_PATH:/tmp/gb-produced.txt"'
    )
    consumer_cmd = (
        f'test "$(cat "$GB_BUILD_WORKDIR/{_SENTINEL_NAME}")" = "{_SENTINEL_VALUE}"; '
        f'echo "consumer read sentinel OK from a separate instance"; '
        'echo "GB_ARTIFACT_ID:consumed GB_ARTIFACT_PATH:/tmp/gb-consumed.txt"'
    )
    return textwrap.dedent(f"""\
        granite.build:
          name: skypilot-aws-shared-fs-e2e
          targets:
            efs-producer-consumer:
              environment_uri: space://environments/skypilot/aws-shared-fs
              inputs:
                seed:
                  uri: "env:///tmp/gb-seed.txt"
              outputs:
                produced:
                  uri: "env:///tmp/gb-produced.txt"
                  type: fileset
                consumed:
                  uri: "env:///tmp/gb-consumed.txt"
                  type: fileset
              steps:
                # Step 1 (producer): writes the sentinel under $GB_BUILD_WORKDIR.
                - step_uri: space://steps/byoc
                  config:
                    poll_interval_seconds: 5
                    byoc_config:
                      image: "{image}"
                      repo: "https://github.com/octocat/Hello-World.git"
                      ref: ""
                      workdir: "code"
                      command: >-
                        {producer_cmd}
                    compute_config:
                      num_gpus_per_node: 0
                      total_memory_per_node: 1Gi
                # Step 2 (consumer): reads it back on a SEPARATE instance.
                - step_uri: space://steps/byoc
                  config:
                    poll_interval_seconds: 5
                    byoc_config:
                      image: "{image}"
                      repo: "https://github.com/octocat/Hello-World.git"
                      ref: ""
                      workdir: "code"
                      command: >-
                        {consumer_cmd}
                    compute_config:
                      num_gpus_per_node: 0
                      total_memory_per_node: 1Gi
        """)


# The buildtest.yaml expectations: both steps run (producer + consumer), each
# registers one env:// output; the seed env:// input is a no-op binding. The
# build only reaches SUCCESS if the consumer's `test ... = <value>` held, i.e.
# EFS carried step 1's sentinel to step 2's separate instance.
_BUILDTEST_YAML = textwrap.dedent("""\
    targets:
      - efs-producer-consumer
    target_expectations:
      - target_name: efs-producer-consumer
        step_count: 2
        input_artifact_count: 1
        output_artifact_count: 2
        expected_steps:
          - step_uri: space://steps/byoc
          - step_uri: space://steps/byoc
    space_uri: ../../space
    tests:
      - runner
    simulate_step_failure: false
    timeout_minutes: 40
    """)


def _write_spec_dir(tmp_path: Path, image: str) -> Path:
    """Materialize environment.yaml + build.yaml + buildtest.yaml into a spec dir.

    Returns the directory containing the three rendered spec files. The EFS
    coordinates are read from the fixture env vars (validated by the caller via
    :func:`_require_efs_fixture`).
    """
    fs_id, region = _require_efs_fixture()
    spec_dir = tmp_path / ("containerized" if image else "bare")
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "environment.yaml").write_text(_environment_yaml(fs_id, region))
    (spec_dir / "build.yaml").write_text(_build_yaml(image=image))
    (spec_dir / "buildtest.yaml").write_text(_BUILDTEST_YAML)
    return spec_dir


async def _submit_and_wait(spec_dir: Path) -> None:
    """Submit the materialized 2-step build to the real SkyPilot/aws harness.

    SCAFFOLD -- FINAL HARNESS WIRING TRACKED IN #393 (do NOT run for real until
    wired):

    The shipped skypilot-aws step tests submit via
    ``libgbtest.buildrunner.buildtest.AbstractYamlBuildRunnerTest`` +
    ``BuildTestSpecification.from_yaml(<spec_dir>/buildtest.yaml)``, resolving the
    ``build.yaml``, the environment, and ``space://steps/byoc`` through a RENDERED
    Space (buildtest.yaml ``space_uri``). That Space must expose our materialized
    ``environment.yaml`` as ``space://environments/skypilot/aws-shared-fs`` (the
    ``environment_uri`` in ``_build_yaml``). Wiring this fully requires:

      1. Rendering a Space that chains (base_uris) to configurations/assets AND
         carries the environment.yaml written into ``spec_dir`` as
         environments/skypilot/aws-shared-fs (mirror how
         steps/byoc/skypilot/Makefile's ``make space`` renders the byoc test
         Space, then drop our environment.yaml into it), pointing buildtest.yaml's
         ``space_uri`` at that rendered Space instead of the placeholder ``../../space``.
      2. Driving the build to a terminal state via the same code path
         ``AbstractYamlBuildRunnerTest._run_build_test`` uses (load the spec with
         ``BuildTestSpecification.from_yaml`` and run it), asserting SUCCESS and the
         buildtest.yaml ``target_expectations`` (step_count == 2, the two byoc
         steps, the two env:// outputs).

    This is deliberately NOT fabricated here: the exact Space-render helper for a
    runtime-materialized environment is not determinable from the codebase without
    running the byoc ``make space`` flow, and the brief forbids inventing a harness
    API. Everything the real run needs — the environment.yaml (efs block from the
    fixture env vars), the concrete 2-step producer->consumer build.yaml, the
    buildtest.yaml expectations, and the assertions below — is complete; only this
    submit call is a scaffold.
    """
    raise NotImplementedError(
        "shared-fs EFS E2E: final SkyPilot/aws build-submission wiring is a "
        "scaffold (tracked in #393). Render a Space exposing the materialized "
        "environment.yaml as space://environments/skypilot/aws-shared-fs and "
        "drive the build via AbstractYamlBuildRunnerTest / "
        "BuildTestSpecification.from_yaml"
        f"({spec_dir / 'buildtest.yaml'}), asserting SUCCESS + the "
        "target_expectations. See this function's docstring."
    )


async def test_efs_producer_consumer_bare(tmp_path):
    """Bare step 1 writes a sentinel under $GB_BUILD_WORKDIR; step 2 (separate EC2)
    reads it; teardown removes the per-run prefix.

    Bare node (byoc ``image: ""`` -> SkyPilot ``image_id: ""``): the commands run
    directly on the EC2 instance against the EFS mount at ``/mnt/gb-shared``.
    """
    _require_aws_credentials()
    fs_id, region = _require_efs_fixture()

    # Materialize the concrete spec (environment.yaml efs block from the fixture
    # coordinates + the 2-step producer->consumer build.yaml + buildtest.yaml).
    spec_dir = _write_spec_dir(tmp_path, image="")
    env_text = (spec_dir / "environment.yaml").read_text()
    assert fs_id in env_text and region in env_text
    assert _SENTINEL_NAME in (spec_dir / "build.yaml").read_text()

    # Submit to the real harness and drive to terminal state.
    # ASSERTIONS (post-submit, once _submit_and_wait is wired):
    #   * both steps reached SUCCESS (the consumer's `test ... = <value>` held,
    #     proving EFS carried step 1's sentinel to step 2's separate instance);
    #   * the per-run prefix builds/<build_id>/runs/<targetrun_id>/ under
    #     ${_MOUNT_POINT} is gone after teardown (WARN-on-failure cleanup).
    await _submit_and_wait(spec_dir)


async def test_efs_producer_consumer_containerized(tmp_path):
    """Same 2-step flow but each step sets image_id (docker:), exercising the
    in-container NFS mount (SkyPilot's default SYS_ADMIN/--net=host/fuse) and the
    1777/uid path.

    byoc ``image: "debian:12-slim"`` -> SkyPilot ``image_id: docker:debian:12-slim``,
    so the producer/consumer commands run INSIDE the container against the EFS mount
    that SkyPilot bind-mounts in (the nfs4 fallback DNS + NFS-client path from the
    EfsProvider).
    """
    _require_aws_credentials()
    fs_id, region = _require_efs_fixture()

    spec_dir = _write_spec_dir(tmp_path, image=_CONTAINER_IMAGE)
    env_text = (spec_dir / "environment.yaml").read_text()
    assert fs_id in env_text and region in env_text
    # build.yaml carries the byoc `image:` value; byoc's step-template renders it
    # to SkyPilot `image_id: docker:<image>` at launch (see byoc step-template.yaml).
    build_text = (spec_dir / "build.yaml").read_text()
    assert f'image: "{_CONTAINER_IMAGE}"' in build_text

    # ASSERTIONS (post-submit, once _submit_and_wait is wired): as the bare
    # variant, PLUS the read happened from INSIDE the container (image_id set), so
    # SUCCESS proves the in-container NFS mount + 1777/uid path.
    await _submit_and_wait(spec_dir)
