import os
from typing import Self

import pytest
from lib.buildwatcher.buildtest import (
    AbstractBuildRunnerTest,
    BuildTestSpecification,
)
from lib.constants import extended_testing_only

from gbserver.types.status import Status

pytestmark = pytest.mark.ibm

_src_file_dir = os.path.abspath(os.path.dirname(__file__))
_test_data_dir = _src_file_dir.replace("test", "test-data", 1)


InvalidBuildTestConfig = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "invalid/build.yaml"),
    expected_status=Status.INVALID,
    targets=[],
    target_expections=[],
)


@extended_testing_only
@pytest.mark.xdist_group(name="invalidbuildtest")
class TestBuildRunnerInvalidBuild(AbstractBuildRunnerTest):

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return InvalidBuildTestConfig
