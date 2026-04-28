import os
from abc import abstractmethod

import pytest
from gbserver_test.buildwatcher.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
)
from gbserver_test.buildwatcher.test_buildrunner_1step import (
    InvalidBuildTestConfig,
    OneStepCPUDownloadTestConfig,
    OneStepGPUDownloadTestConfig,
)

from gbserver_test.constants import extended_testing_only

pytestmark = pytest.mark.ibm


@pytest.mark.skipif(
    os.environ.get("GBTEST_HAS_GB_CLUSTER_ACCESS", "True").lower() == "false"
    or os.environ.get("HAS_GB_CLUSTER_ACCESS", "True").lower() == "false",
    reason="Can't run this since it is configured as not having G.B cluster access",
)
class AbstractTestBuildWatcher(AbstractBuildTest):

    def _get_build_count(self) -> int:
        return 1

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        raise NotImplementedError("Must provide test config")

    def test_build_watcher_run(self):
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDWATCHER,
            test_spec=self._get_test_config(),
            test_cancel=False,
            build_count=self._get_build_count(),
        )

    def test_build_watcher_cancel(self):
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDWATCHER,
            test_spec=self._get_test_config(),
            test_cancel=True,
            build_count=self._get_build_count(),
        )



@pytest.mark.xdist_group(name="buildwatcher_cpu")
@extended_testing_only
class TestBuildWatcherCPU(AbstractTestBuildWatcher):
    def _get_build_count(self) -> int:
        return 1

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        return OneStepCPUDownloadTestConfig


@pytest.mark.xdist_group(name="buildwatcher_invalid_build")
class TestBuildWatcherInvalidBuild(AbstractTestBuildWatcher):
    def _get_build_count(self) -> int:
        return 1

    # No need to run this for an invalid build (and cancel test does not support invalid builds).
    @pytest.mark.skip
    def test_build_watcher_cancel(self):
        pass

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        return InvalidBuildTestConfig


@pytest.mark.xdist_group(name="buildwatcher_gpu")
class TestBuildWatcherGPU(AbstractTestBuildWatcher):

    def _get_build_count(self) -> int:
        return 1

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        return OneStepGPUDownloadTestConfig


# @pytest.mark.skip(
#     reason="K8s AppWrapper infrastructure flaky — disabled for v0.3.0 release"
# )
@pytest.mark.xdist_group(name="buildwatcher_multi_cpu")
@extended_testing_only
class TestBuildWatcherMultiCPU(AbstractTestBuildWatcher):
    """Provides a test of the build watcher to run and cancel simultaneous builds."""

    def _get_build_count(self) -> int:
        return 3

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        return OneStepCPUDownloadTestConfig
