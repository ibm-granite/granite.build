from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.ibm


@extended_testing_only
class TestDiGiT_SFTFull_FMEval(AbstractYamlBuildRunnerTest):

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "builds/DiGiT_SFTFull_FMEval"
