import os
from typing import Self
from gbserver_test.constants import extended_testing_only

import pytest
from gbserver_test.buildwatcher.buildtest import (
    AbstractBuildRunnerTest,
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
    ExpectedTarget,
)

from gbserver.types.status import Status

pytestmark = pytest.mark.ibm

_src_file_dir = os.path.abspath(os.path.dirname(__file__))
_test_data_dir = _src_file_dir.replace("test", "test-data", 1)

################################################################################
OneStepGPUDownloadTestConfig = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "1step/gpu/build.yaml"),
    # targets=None,
    targets=["download_file"],
    target_expections=[
        ExpectedTarget(
            target_name="download_file",
            step_count=5,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=3,
        ),
    ],
)
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_gpu")
class TestBuildRunner1StepGPU(AbstractBuildRunnerTest):

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return OneStepGPUDownloadTestConfig

    def test_runner_cancellation(self):
        """Test simple build cancelation on our 1step build.yaml.  We don't need to test this on every build we want to test, so just included it here"""
        test_spec = self._get_test_specification()
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=test_spec,
            test_cancel=True,
        )

################################################################################
OneStepCPUDownloadTestConfig = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "1step/cpu/build.yaml"),
    targets=["download_file"],
    target_expections=[
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
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_cpu")
class TestBuildRunner1StepCPU(AbstractBuildRunnerTest):

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return OneStepCPUDownloadTestConfig

    def test_runner_cancellation(self):
        """Test simple build cancelation on our 1step build.yaml.  We don't need to test this on every build we want to test, so just included it here"""
        test_spec = self._get_test_specification()
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=test_spec,
            test_cancel=True,
        )



################################################################################
InvalidBuildTestConfig = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "invalid/build.yaml"),
    expected_status=Status.INVALID,
    targets=[],  # ["download_file"],
    target_expections=[
        # # CPU env has 5 steps (not 4 like GPU) since it does not mount the AFM and does an lhpull step instead.
        # ExpectedTarget(
        #     target_name="download_file",
        #     step_count=5,
        #     input_artifact_count=1,
        #     output_artifact_count=1,
        #     jobstats_count=3,
        # ),
    ],
)
@extended_testing_only
@pytest.mark.xdist_group(name="invalidbuildtest")
class TestBuildRunnerInvalidBuild(AbstractBuildRunnerTest):

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return InvalidBuildTestConfig

################################################################################
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
@pytest.mark.skip
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

################################################################################
OneStepHFTestConfig = BuildTestSpecification(
    build_yaml=os.path.join(_test_data_dir, "1step/hf/build.yaml"),
    targets=["download_file"],
    target_expections=[
        ExpectedTarget(
            target_name="download_file",
            step_count=5,
            input_artifact_count=1,
            output_artifact_count=1,
            jobstats_count=3,
        ),
    ],
)
@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_hf")
class TestBuildRunner1StepHF(AbstractBuildRunnerTest):

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        return OneStepHFTestConfig



################################################################################
#### -- We disable this now that we only reuse targets within a retry chain and not
#### -- across unrelated builds.
# TargetHashSkipSecondRunTestConfig = BuildTestSpecification(
#     build_yaml=os.path.join(_test_data_dir, "1step/skip/build.yaml"),
#     input_artifact_uris=[
#         "lh://staging/granite_dot_build.public/tables/maximo_digit_input1",
#     ],
#     targets=None,  # run all targets in build.yaml
#     target_expections=[
#         ExpectedTarget(  # skipped — step/artifact counts not checked by verifier
#             target_name="download_file1",
#             step_count=0,
#             input_artifact_count=0,
#             output_artifact_count=0,
#             jobstats_count=3,  # derived from original target's 3 output artifacts
#         ),
#         ExpectedTarget(  # actually runs, receives download_file output via binding
#             target_name="download_file2",
#             step_count=5,
#             input_artifact_count=1,
#             output_artifact_count=1,
#             jobstats_count=3,
#         ),
#     ],
#     skip_target_names=["download_file1"],
#     simulate_failure=False,  # skip tests don't inject pod eviction failures
# )
# TargetHashSkipTestConfig = BuildTestSpecification(
#     build_yaml=os.path.join(_test_data_dir, "1step/skip/build.yaml"),
#     input_artifact_uris=[
#         "lh://staging/granite_dot_build.public/tables/maximo_digit_input1",
#     ],
#     targets=["download_file1"],
#     target_expections=[
#         ExpectedTarget(
#             target_name="download_file1",
#             step_count=5,
#             input_artifact_count=1,
#             output_artifact_count=1,
#             jobstats_count=3,
#         ),
#     ],
# )
# @pytest.mark.skipif(
#     os.environ.get("GBTEST_ENABLE_EXTENDED_TESTS", "true").lower() == "false",
#     reason="GBTEST_ENABLE_EXTENDED_TESTS is set to false",
# )
# @pytest.mark.xdist_group(name="buildtest_k8s")
# class TestBuildRunnerTargetSkip(AbstractBuildTest):
#     """
#     Verifies hash-based target skip across two builds:
#     1. First build runs build.yaml (single target: download_file) to completion.
#     2. Second build runs build2.yaml (two targets: download_file + process_file).
#        download_file is skipped because its target_hash matches the first build.
#        process_file receives download_file's output via binding and runs normally.
#     """

#     def test_runner_target_skip(self: Self):
#         self._run_two_builds_sequential(
#             tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
#             test_spec=TargetHashSkipTestConfig,
#             second_run_spec=TargetHashSkipSecondRunTestConfig,
#         )


# #@pytest.mark.skipif( os.environ.get("GBTEST_ENABLE_EXTENDED_TESTS", "true").lower() == "false", reason="GBTEST_ENABLE_EXTENDED_TESTS is set to false")
# @pytest.mark.skip(reason="Temporarily disable since we need a PR to main and this is failing due to space config issues (we think).")
# @pytest.mark.xdist_group(name="buildtest_k8s")
# =======

################################################################################
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
