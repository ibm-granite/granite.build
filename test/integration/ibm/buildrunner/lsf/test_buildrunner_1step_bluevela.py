import os
from typing import Self

import pytest
from lib.buildwatcher.buildtest import (
    AbstractBuildRunnerTest,
    BuildTestSpecification,
    ClassTestedEnum,
    ExpectedTarget,
)
from lib.constants import extended_testing_only

pytestmark = pytest.mark.ibm

_src_file_dir = os.path.abspath(os.path.dirname(__file__))
_test_data_dir = _src_file_dir.replace("test", "test-data", 1)


OneStepBlueVelaDownloadTestConfig = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "1step/bluevela/build.yaml"),
    targets=["download_file"],
    target_expections=[
        # TODO: is this true for BlueVela?
        # CPU env has 5 steps (not 4 like GPU) since it does not mount the AFM and does an lhpull step instead.
        ExpectedTarget(
            target_name="download_file",
            step_count=5,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=3,
        ),
    ],
)


# @pytest.mark.skipif(
#     os.environ.get("GBTEST_ENABLE_BLUEVELA_TESTS", "false").lower() == "false"
#     and os.environ.get("GBTEST_ENABLE_EXTENDED_TESTS", "false").lower() != "true",
#     reason="GBTEST_ENABLE_BLUEVELA_TESTS is set to false",
# )
@extended_testing_only
# @pytest.mark.skip
@pytest.mark.xdist_group(name="buildtest_bv")
class TestBuildRunner1StepBlueVela(AbstractBuildRunnerTest):

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return OneStepBlueVelaDownloadTestConfig

    def test_runner_cancellation(self):
        """Test simple build cancelation on our 1step build.yaml.  We don't need to test this on every build we want to test, so just included it here"""
        test_spec = self._get_test_specification()
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=test_spec,
            test_cancel=True,
        )
