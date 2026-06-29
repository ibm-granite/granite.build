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
class TestSkypilotBlueVelaCommand2Target(AbstractYamlBuildRunnerTest):
    """Two command-step targets on BlueVela LSF; target 2 binds target 1's output."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
