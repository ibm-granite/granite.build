import os

import pytest
from gbserver_test.buildwatcher.buildtest import AbstractBuildTest, ClassTestedEnum
from gbserver_test.buildwatcher.test_buildrunner_1step import (
    OneStepCPUDownloadTestConfig,
)
from gbserver_test.constants import extended_testing_only

pytestmark = pytest.mark.ibm

@extended_testing_only
class TestBuildRunnerJob(AbstractBuildTest):

    # We set HAS_GB_CLUSTER_ACCESS=False in the travis builds. HAS_VELA_ACCESS is deprecated.
    @pytest.mark.skipif(
        os.environ.get("GBTEST_HAS_GB_CLUSTER_ACCESS", "True").lower() == "false"
        or os.environ.get("HAS_GB_CLUSTER_ACCESS", "True").lower() == "false",
        reason="Can't run this since it is configured as not having G.B cluster access",
    )
    def test_runnerjob(self):
        # Note that this requires oc login to ris3
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNERJOB,
            test_spec=OneStepCPUDownloadTestConfig,
            test_cancel=False,
        )

    # We set HAS_GB_CLUSTER_ACCESS=False in the travis builds. HAS_VELA_ACCESS is deprecated.
    @pytest.mark.skipif(
        os.environ.get("GBTEST_HAS_GB_CLUSTER_ACCESS", "True").lower() == "false"
        or os.environ.get("HAS_GB_CLUSTER_ACCESS", "True").lower() == "false",
        reason="Can't run this since it is configured as not having G.B cluster access",
    )
    def test_runnerjob_cancellation(self):
        # Note that this requires oc login to ris3
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNERJOB,
            test_spec=OneStepCPUDownloadTestConfig,
            test_cancel=True,
        )
