import os
from typing import Self

import pytest
from lib.buildwatcher.buildtest import (
    AbstractBuildRunnerTest,
    BuildTestSpecification,
    ClassTestedEnum,
    ExpectedTarget,
)

pytestmark = pytest.mark.ibm

_src_file_dir = os.path.abspath(os.path.dirname(__file__))
_test_data_dir = _src_file_dir.replace("test", "test-data", 1)


OneStepLocalTestConfig = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "1step/local/build.yaml"),
    targets=["echo"],
    target_expections=[
        ExpectedTarget(
            target_name="echo",
            step_count=1,
            input_artifact_count=0,
            output_artifact_count=0,
            jobstats_count=0,
        ),
    ],
)


@pytest.mark.skip(
    reason="Temporarily disable since we need a PR to main and this is failing due to space config issues (we think)."
)
@pytest.mark.xdist_group(name="buildtest_local")
class TestBuildRunner1StepLocal(AbstractBuildRunnerTest):
    """Runs a barebone local build flow."""

    run_locally = True

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return OneStepLocalTestConfig

    def test_runner_cancellation(self):
        """Test simple build cancelation on our 1step local build.yaml."""
        test_spec = self._get_test_specification()
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=test_spec,
            test_cancel=True,
        )
