import os

import pytest
from lib.buildwatcher.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
    get_test_data_dir_for,
)
from lib.constants import extended_testing_only

pytestmark = pytest.mark.ibm


@extended_testing_only
class TestBuildRunnerJob(AbstractBuildTest):

    def _get_test_specification(self) -> BuildTestSpecification:
        """Load the 1step/cpu fixture spec from buildtest.yaml.

        The fixture lives under the sibling buildrunner/k8s/ test-data tree —
        this test exercises the same build but via the K8s buildrunner job
        path rather than the local buildrunner — so reach across via .parent.
        """
        k8s_fixtures = (
            get_test_data_dir_for(__file__).parent.parent / "buildrunner" / "k8s"
        )
        return BuildTestSpecification.from_yaml(
            k8s_fixtures / "1step" / "cpu" / "buildtest.yaml"
        )

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
            test_spec=self._get_test_specification(),
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
            test_spec=self._get_test_specification(),
            test_cancel=True,
        )
