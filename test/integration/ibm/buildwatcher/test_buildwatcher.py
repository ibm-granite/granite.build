import os
from abc import abstractmethod

import pytest
from lib.buildwatcher.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
    get_test_data_dir_for,
)
from lib.constants import extended_testing_only

pytestmark = pytest.mark.ibm

# K8s buildrunner fixtures live under test-data/integration/ibm/buildrunner/k8s/
# (sibling of this file's test-data dir test-data/integration/ibm/buildwatcher).
_K8S_FIXTURES = get_test_data_dir_for(__file__).parent / "buildrunner" / "k8s"
_CPU_YAML = _K8S_FIXTURES / "1step" / "cpu" / "buildtest.yaml"
_GPU_YAML = _K8S_FIXTURES / "1step" / "gpu" / "buildtest.yaml"
# The invalid-build fixture is environment-agnostic and lives in the standalone tree.
_INVALID_YAML = (
    get_test_data_dir_for(__file__).parent.parent
    / "standalone"
    / "buildrunner"
    / "invalid"
    / "buildtest.yaml"
)


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
class TestBuildWatcherCPU(AbstractTestBuildWatcher):
    def _get_build_count(self) -> int:
        return 1

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(_CPU_YAML)


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
        return BuildTestSpecification.from_yaml(_INVALID_YAML)


@pytest.mark.xdist_group(name="buildwatcher_gpu")
class TestBuildWatcherGPU(AbstractTestBuildWatcher):

    def _get_build_count(self) -> int:
        return 1

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(_GPU_YAML)


# @pytest.mark.skip(
#     reason="K8s AppWrapper infrastructure flaky — disabled for v0.3.0 release"
# )
@pytest.mark.xdist_group(name="buildwatcher_multi_cpu")
class TestBuildWatcherMultiCPU(AbstractTestBuildWatcher):
    """Provides a test of the build watcher to run and cancel simultaneous builds."""

    def _get_build_count(self) -> int:
        return 3

    @abstractmethod
    def _get_test_config(self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(_CPU_YAML)
