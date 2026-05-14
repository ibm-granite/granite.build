import os
from typing import Self

import pytest
from lib.buildwatcher.buildtest import (
    AbstractBuildRunnerTest,
    BuildTestSpecification,
    ExpectedTarget,
)
from lib.constants import extended_testing_only

pytestmark = pytest.mark.ibm

_src_file_dir = os.path.abspath(os.path.dirname(__file__))
_test_data_dir = _src_file_dir.replace("test", "test-data", 1)


@extended_testing_only
class AbstractLongBuildTest(AbstractBuildRunnerTest):
    """Defined just so we can share the skip definition above."""

    pass


DiGiT_SFTFull_FMEval = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "builds/DiGiT_SFTFull_FMEval/build.yaml"),
    targets=None,
    target_expections=[
        ExpectedTarget(
            target_name="syntheticdatageneration",
            step_count=3,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=1,
        ),
        ExpectedTarget(
            target_name="tunedmodel",
            step_count=6,   # 2 inputs, 1 tuning, 2 outputs, one of which has 2 checkpoints = 6 
            input_artifact_count=2,
            output_artifact_count=2,
            jobstats_count=3,
        ),
        ExpectedTarget(
            target_name="evalresults",
            step_count=3,   # 1 input, 1 step, 1 output
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=1,
        ),
    ],
    timeout_minutes=120,
)


class TestDiGiT_SFTFull_FMEval(AbstractLongBuildTest):

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return DiGiT_SFTFull_FMEval
