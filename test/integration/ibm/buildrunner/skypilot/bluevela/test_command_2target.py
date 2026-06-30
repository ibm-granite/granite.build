# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Two command-step targets on BlueVela LSF (via Skypilot).

`first` runs the generic `command` step to echo an output file onto the shared
filesystem and register it as artifact `out1`; `second` binds `first.out1` as an
input, reads it, and registers its own output `out2`. This exercises cross-target
output -> input binding over the env_local (shared-FS) assetstore on BlueVela.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.ibm


@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_bv")
# For this test to run in IBM SPS build tests, it needs to
# 1) have a environments/skypilot/lsf/bluevela/environment.yaml referencing BV_SSH_PRIVATE_KEY secret
#     IdentityKey: BV_SSH_PRIVATE_KEY
# 2) Change the test to use the public IBM space, which uses the ibm secret manager
# Without these changes, the test uses the local space and expects a local ~/.ssh/ibm-bluevela.key
# This allows it to be run locally.
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until we have environments/skypilot/lsf/bluevela/environment.yaml with key reference in gb-test and other space repos",
)
class TestSkypilotBlueVelaCommand2Target(AbstractYamlBuildRunnerTest):
    """Two command-step targets on BlueVela LSF; target 2 binds target 1's output."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
