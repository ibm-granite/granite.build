# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""BYOC custom_code_skypilot target on BlueVela SLURM (via Skypilot) — MANUAL ONLY.

Exercises the bring-your-own-code (BYOC) step end-to-end against the BlueVela
SLURM environment (space://environments/skypilot/slurm/bluevela, provided by the
gb-test space): clone a GHE workload repo, materialize a hash-keyed conda/
micromamba environment from a conda lockfile, run the workload's start command
inside that env, capture a directory as the output artifact, and push it to
hf://. It also asserts the step recorded the resolved commit SHA as `commit_hash`
step metadata (build lineage). The committed build.yaml runs INSIDE a container
image (byoc_config.image = docker.io/library/buildpack-deps:bookworm-scm), the
SLURM containerized path that needs the Pyxis SPANK plugin BlueVela provides; set
byoc_config.image to "" to run on the bare launcher node instead (bootstrapping a
static micromamba binary, no Pyxis).

The fixture (build.yaml + buildtest.yaml) lives in the directory returned by
_get_yaml_spec_dir below. build.yaml declares an hf:// input (byoc_input) and an
hf:// output, so buildrunner brackets the step with an hfpull and an hfpush —
step_count is 3 (hfpull, custom_code_skypilot, hfpush). Its byoc_config points at
a concrete GHE repo/ref/lockfile with real commands; adjust those values for a
different workload.

Running it manually
-------------------
This test is SKIPPED unless GBTEST_ENABLE_MANUAL_TESTS=1, because it depends
on external, possibly-unpushed resources. To run it by hand:

1. Set GBTEST_ENABLE_MANUAL_TESTS=1 (and HF_TOKEN with write access to the
   output repo).
2. Point the sibling build.yaml's byoc_config at a workload YOU can reach (its
   committed github_url/github_ref/conda_lockfile_path/commands are a concrete
   example) and the hf:// output at a repo you can write to.
3. Ensure the gb-test space provides the 5 space secrets the step needs
   (GITHUB_IBM_PAT, CLEARML_API_HOST/ACCESS_KEY/SECRET_KEY, HF_TOKEN) and the
   BlueVela SSH key, and that you have BlueVela SSH access.

To run against the UNPUSHED assets step (before the assets branch is pushed):

4. Point buildtest.yaml `space_uri` at a LOCAL gb-test clone (uncomment one of the
   local alternatives there), and edit that clone's space.yaml so its `base_uris`
   resolves the step from the local assets clone:
   `base_uris: [file:///path/to/your/assets-clone]`.
5. The step.yaml's git+ssh `validator_uri` is cloned EAGERLY by the custom_code
   validator's constructor (Asset(...).sync during assimilate), and base_uris
   redirection does NOT apply to it — disabling validation would not avoid this
   clone. So either push the assets branch, OR temporarily set the local assets
   clone's step.yaml `validator_uri` to a local file:// path:
   `file:///path/to/your/assets-clone/steps/custom_code_skypilot/validators`.

All of steps 2, 4, 5 are manual edits to your own clones — never automated.

Note (SSH auth): to validate SSH against a freshly edited key/credential, set
GBTEST_SKY_SSH_RESET=true in gbserver's environment before running — SkyPilot
otherwise reuses a persisted SSH ControlMaster socket keyed on (host, port, user),
not the key, masking an edited cluster_ssh_config for the ControlPersist window.
Leave it unset for normal runs; the socket clear globs the whole per-user root and
could yank another parallel skypilot build's socket.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only, manual_testing_only

pytestmark = pytest.mark.ibm


@extended_testing_only
@manual_testing_only
@pytest.mark.xdist_group(name="buildtest_bv")
class TestSkypilotBlueVelaSlurmCustomCode(AbstractYamlBuildRunnerTest):
    """BYOC custom_code_skypilot step on BlueVela SLURM: clone → hash-keyed env →
    workload → hf:// artifact capture + commit_hash lineage."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "custom_code"
