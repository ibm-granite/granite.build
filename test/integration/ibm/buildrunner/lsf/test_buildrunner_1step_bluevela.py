# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""YAML-driven equivalent of TestBuildRunner1StepBlueVela.

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


# @pytest.mark.skipif(
#     os.environ.get("GBTEST_ENABLE_BLUEVELA_TESTS", "false").lower() == "false"
#     and os.environ.get("GBTEST_ENABLE_EXTENDED_TESTS", "false").lower() != "true",
#     reason="GBTEST_ENABLE_BLUEVELA_TESTS is set to false",
# )
@extended_testing_only
# @pytest.mark.skip
@pytest.mark.xdist_group(name="buildtest_bv")
class TestBuildRunner1StepBlueVela(AbstractYamlBuildRunnerTest):

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step/bluevela"
