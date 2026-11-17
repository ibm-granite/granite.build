# SPDX-License-Identifier: Apache-2.0

"""YAML-driven equivalent of TestBuildRunnerInvalidBuild.

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

pytestmark = pytest.mark.standalone


@extended_testing_only
@pytest.mark.xdist_group(name="invalidbuildtest")
class TestBuildRunnerInvalidBuild(AbstractYamlBuildRunnerTest):

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "invalid"
