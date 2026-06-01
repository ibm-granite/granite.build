#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
End-to-end build tests.
"""

import os
import re
from abc import abstractmethod
from datetime import timedelta
from enum import Enum
from pathlib import Path
from time import sleep, time
from typing import ClassVar, List, Optional, Self, Union

import pytest
import yaml

pytest.importorskip("kubernetes_asyncio")

from libgbtest.buildrunner.utils import (
    ExceptionRaisingThread,
    cluster_logout,
    delete_buildrunner_pod,
    gb_cluster_login,
    is_buildrunner_pod_finished,
)
from libgbtest.constants import (
    GBTEST_JOB_TERMINATION_TIMEOUT_SECONDS,
    GBTEST_SKIP_BUILD_TEARDOWN,
    GBTEST_SPACE_NAME,
    GBTEST_USER_NAME,
    failed_build_assert_message,
    is_extended_testing_enabled,
)
from libgbtest.utils import (
    AbstractSingletonStorageUsingPreloadedSpaceTest,
    check_env_var_set,
    is_pytest_running_parallel,
)
from pydantic import BaseModel, field_validator

from gbcommon.types.testing import (
    disable_failure_simulation,
    disable_hf_mocks,
    enable_failure_simulation,
    enable_hf_mocks,
)
from gbcommon.uri.uri import URI
from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.buildrunnerjob.buildrunnerjob import BuildRunnerJob
from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.github.myghapi import MyGHApi
from gbserver.lineage.jobstats import get_lineage_store, reset_lineage_store
from gbserver.storage.artifact_registration import (
    ArtifactRegistration,
    ArtifactRegistrationStatus,
)
from gbserver.storage.sql.space_storage import SQLSpaceStorage
from gbserver.storage.storage import IItemStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import (
    ENV_VAR_GBSERVER_IMAGE_TAG,
    ENV_VAR_SIDECAR_MONITORING_IMAGE_TAG,
    GB_ENVIRONMENT,
    GBSERVER_DEFAULT_BUILDRUNNER_TYPE,
    GBSERVER_GBSERVER_IMAGE_TAG,
    GBSERVER_GITHUB_TOKEN,
    GBSERVER_SIDECAR_MONITORING_IMAGE_TAG,
)
from gbserver.types.prwatcherconfig import get_uri_parts
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_time, get_uuid

logger = get_logger(__name__)


class ExpectedTarget(BaseModel):
    target_name: str
    """Name of the target as it appears in the build.yaml"""
    step_count: int
    """Number of expected steps records to be created for the target. -1 if not to be checked."""
    input_artifact_count: int
    """Number of input artifacts to be recorded in the recorded target record."""
    output_artifact_count: int
    """Number of output artifacts to be recorded in the recorded target record."""
    jobstats_count: int


class BuildTestSpecification(BaseModel):
    build_yaml: str
    """Path to build.yaml file to run"""
    expected_status: Status = Status.SUCCESS
    """The end-of-build build status, usually either SUCCESS or INVALID"""
    space_name: str = GBTEST_SPACE_NAME
    targets: Optional[list[str]] = None  # list of targets to run or None for all.
    """Optional list of targets within the build.yaml to be executed."""
    target_expectations: list[ExpectedTarget]
    """List of expected results for each target execute in the build.yaml."""
    timeout_minutes: int = 30
    """Number of minutes to wait for the build completion"""
    simulate_step_failure: bool = True
    """If True, signal the environment to inject a simulated failure once the workload starts.
    The specific failure event is chosen by the environment implementation (e.g. K8s injects
    an AppWrapper failure). The RetryHandler absorbs the failure and retries, so expected_status
    should still be SUCCESS unless all retries are exhausted."""
    space_uri: Optional[str] = None
    """If set, overrides the space's git_repo_uri so the BuildRunner resolves space:// URIs
    from this local path instead of cloning from GitHub.  PR creation and verification are
    automatically skipped when this is set (no GitHub repo available)."""
    skip_target_names: list[str] = []
    """Target names expected to be skipped on a second run (used by _run_two_builds_sequential)."""
    tests: list[str] = ["runner", "runner_cancellation"]
    """Which test methods on AbstractYamlBuildRunnerTest run for this spec.
    Each value <key> maps to a method test_<key> on the base class.  Default
    runs both the basic build and the cancellation variant — fixtures that
    want to skip the cancellation path can override with ['runner'].  Adding
    a new key requires both a new ``test_<key>`` method on
    AbstractYamlBuildRunnerTest AND adding the key to ``KNOWN_TEST_KEYS``
    below so YAML typos are caught at load time."""

    KNOWN_TEST_KEYS: ClassVar[set[str]] = {"runner", "runner_cancellation"}
    """Valid values for the ``tests`` field.  Kept in sync with the
    ``test_<key>`` methods on AbstractYamlBuildRunnerTest so a YAML typo like
    ``tests: [runer]`` fails at load time rather than silently skipping every
    test method."""

    @field_validator("expected_status", mode="before")
    @classmethod
    def _normalize_status(cls, v):
        """Allow YAML to write SUCCESS or success — Status (StrEnum) values are lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("tests")
    @classmethod
    def _validate_tests(cls, v: list[str]) -> list[str]:
        """Reject unknown test keys at YAML-load time.

        Without this, ``tests: [runer]`` would silently skip every test method
        with no error.  Catching it here means the failure surfaces during
        collection with a clear message naming the offending key.
        """
        unknown = sorted(set(v) - cls.KNOWN_TEST_KEYS)
        assert (
            not unknown
        ), f"unknown test keys {unknown}; valid keys: {sorted(cls.KNOWN_TEST_KEYS)}"
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "BuildTestSpecification":
        """Construct a BuildTestSpecification from a YAML file.

        Path-typed fields are resolved relative to the YAML file's directory so a
        fixture is self-contained (move the directory, the spec still resolves).
        Specifically:
          - ``build_yaml`` defaults to ``./build.yaml`` (sibling) when omitted, and
            relative paths resolve against the YAML's parent directory.
          - ``space_uri``: a relative ``file://`` URI or a bare relative filesystem
            path is resolved against the YAML's parent directory and returned as
            an absolute ``file://`` URI. Other URI schemes pass through unchanged.

        Args:
            path: Path to the buildtest.yaml file.

        Returns:
            A validated BuildTestSpecification.

        Raises:
            AssertionError: if the path does not exist or YAML is not a mapping.
            pydantic.ValidationError: if required fields are missing/invalid.
        """
        path = Path(path).resolve()
        assert path.is_file(), f"expected path '{path}' to be a file"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        assert isinstance(data, dict), f"expected a dict, actual: {data}"

        yaml_dir = path.parent
        data["build_yaml"] = _resolve_build_yaml(data.get("build_yaml"), yaml_dir)
        if data.get("space_uri") is not None:
            data["space_uri"] = _resolve_space_uri(data["space_uri"], yaml_dir)
        return cls.model_validate(data)


def _resolve_build_yaml(value: Optional[str], yaml_dir: Path) -> str:
    """Resolve a build_yaml value from a buildtest.yaml.

    Defaults to './build.yaml' when value is None.  Relative paths are resolved
    against ``yaml_dir`` and absolute paths are returned unchanged.
    """
    if value is None:
        value = "./build.yaml"
    p = Path(value)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    return str(p)


def _resolve_space_uri(value: str, yaml_dir: Path) -> str:
    """Resolve a space_uri value from a buildtest.yaml.

    Returns absolute URIs unchanged.  Relative ``file://`` URIs and bare relative
    filesystem paths are resolved against ``yaml_dir`` and returned as absolute
    ``file://`` URIs.
    """
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+\-.]*)://", value)
    if scheme_match:
        if scheme_match.group(1) != "file":
            return value
        path_part = value[len("file://") :]
        p = Path(path_part)
        if p.is_absolute():
            return value
        p = (yaml_dir / p).resolve()
        return f"file://{p}"
    p = Path(value)
    if not p.is_absolute():
        p = (yaml_dir / p).resolve()
    return f"file://{p}"


def get_test_data_dir_for(test_module_file: str) -> Path:
    """Return the test-data directory that mirrors the given test module file.

    Implements the project's ``test/<...>`` ↔ ``test-data/<...>`` parallel-tree
    convention so YAML-driven test classes can spell out the fixture directory
    in one readable line:

        return get_test_data_dir_for(__file__) / "1step/cpu"

    Args:
        test_module_file: Pass ``__file__`` from the test module.

    Returns:
        Absolute Path to the test-data directory parallel to the test module.
    """
    d = Path(test_module_file).resolve().parent
    return Path(str(d).replace("/test/", "/test-data/", 1))


def get_test_data_root() -> Path:
    """Return the absolute path of the integration test-data root.

    Use this for cross-tree fixture references (e.g. a buildwatcher test that
    consumes a buildrunner fixture) so callers don't have to count
    ``.parent.parent`` levels — those break the moment a test file moves up or
    down a directory.  Top-down reads like::

        get_test_data_root() / "ibm" / "buildrunner" / "k8s"

    are stable against test-tree reorganization.

    Computed from this module's own filesystem location
    (``<repo>/test/libgbtest/buildrunner/buildtest.py``) so the result is independent
    of the caller's location.

    Returns:
        Absolute Path of ``<repo>/test-data/integration``.
    """
    return Path(__file__).resolve().parents[3] / "test-data" / "integration"


class ClassTestedEnum(Enum):
    TEST_BUILDRUNNER = 1
    TEST_BUILDRUNNERJOB = 2
    TEST_BUILDWATCHER = 3


class AbstractBuildTest(AbstractSingletonStorageUsingPreloadedSpaceTest):

    def setup_method(self: Self, method):
        # breakpoint()
        # Only use real HF calls when extended testing is enabled.
        if not is_extended_testing_enabled:
            self._enable_artifact_mocks()

        self.class_tested = None
        run_locally = getattr(self, "run_locally", False)
        logger.info(f"Test to be run locally: {run_locally}")
        if run_locally:
            self.was_logged_in = True
        else:
            if GBSERVER_DEFAULT_BUILDRUNNER_TYPE == "job":
                logger.info("Begin cluster login. ")
                self.was_logged_in = gb_cluster_login()
                logger.info("End cluster login. ")
            else:
                self.was_logged_in = True  # Don't oc logout when done
        # These env vars do not apply to all build environments, but lets
        # generally require at least the sidecar image for K8s and maybe the other for BuildRunnerJob
        check_env_var_set(
            ENV_VAR_GBSERVER_IMAGE_TAG,
            f"Build tests must be configured with the gbserver image to use with the {ENV_VAR_GBSERVER_IMAGE_TAG} env var. Use 'make info' in the dev or main branch.",
        )
        check_env_var_set(
            ENV_VAR_SIDECAR_MONITORING_IMAGE_TAG,
            f"Build tests must be configured with the sidecar image to use with the {ENV_VAR_SIDECAR_MONITORING_IMAGE_TAG} env var. Use 'make info' in the dev or main branch.",
        )

        super().setup_method(method)

    def teardown_method(self: Self, method):
        if not is_extended_testing_enabled:
            self._disable_artifact_mocks()

        if GBTEST_SKIP_BUILD_TEARDOWN:
            logger.warning("skipping the teardown of the test build!")
            return
        # Remove all the output artifacts created by the builds.
        try:
            builds = self.storage.build_storage.get_by_uuid(None)
        except Exception as e:
            logger.warning(
                f"Could not get builds.  Skipping removal of their output artifacts. {e}"
            )
            builds = []

        # Cleanup output artifacts
        for build in builds:
            try:
                self._delete_output_artifacts(build.uuid)
            except Exception as e:
                logger.warning(
                    f"Ignoring failure to delete 1 or more output artifacts for build {build.uuid}: {e}"
                )

        # Clean up left over pods/jobs
        # breakpoint()
        if self.class_tested in [
            ClassTestedEnum.TEST_BUILDRUNNERJOB,
            ClassTestedEnum.TEST_BUILDWATCHER,
        ]:
            for build in builds:
                delete_buildrunner_pod(build.uuid)

        if not self.was_logged_in and not is_pytest_running_parallel():
            # When a developer is running locally with existing login, we don't want to log them out.
            # And when running in parallel, neverl logout so we don't disturb another test that may require login.
            cluster_logout()

        reset_lineage_store()

        # And now do the super cleanups
        return super().teardown_method(method)

    def _enable_artifact_mocks(self) -> None:
        """Enable push/pull/exists/delete mocking for this process and remote jobs/pods
        and the input/output URIs used in the build.  Currently onl hf://."""
        enable_hf_mocks()

    def _disable_artifact_mocks(self) -> None:
        """Disable artifact mocking."""
        disable_hf_mocks()

    def _failed_build_msg(self: Self, build_id: str, message: str) -> str:
        """Format an assertion message with the build ID for easier debugging."""
        return failed_build_assert_message(build_id, message)

    def _check_and_setup_space(
        self: Self, test_spec: BuildTestSpecification
    ) -> StoredSpace:
        """Fetch the space for the test, upserting git_repo_uri when space_uri is set.

        When test_spec.space_uri is provided (a local file:// path), the stored space
        record is updated so that validation.py's assertion — which compares
        get_gb_space_config_uri(git_repo_uri) against Space.uristr — passes without
        making any real GitHub calls.

        Args:
            test_spec (BuildTestSpecification): The test specification containing
                space_name and optional space_uri.

        Returns:
            StoredSpace: The resolved (and possibly updated) space record.

        Raises:
            AssertionError: If no space is found and space_uri is also not set.
        """
        logger.info(f"Getting the {test_spec.space_name} space")
        space = self.storage.space_storage.get_by_name(name=test_spec.space_name)
        logger.info(f"Got the {test_spec.space_name} space")
        if test_spec.space_uri is not None:
            # Upsert the space record so git_repo_uri matches the local file:// URI.
            # validation.py passes file:// URIs through get_gb_space_config_uri unchanged,
            # so storing space_uri as git_repo_uri makes the assertion at line 270 pass.
            if space is None:
                space = StoredSpace(
                    name=test_spec.space_name,
                    git_repo_uri=test_spec.space_uri,
                    lakehouse_namespace="",
                )
                self.storage.space_storage.add([space])
            else:
                space.git_repo_uri = test_spec.space_uri
                self.storage.space_storage.update(space)
        assert (
            space is not None
        ), f"failed to find a space with name {test_spec.space_name}"
        return space

    def _delete_output_artifacts(self: Self, build_id: str):
        """Delete all output artifacts created by a build from their remote stores.

        Each artifact's URI handles its own deletion via ``URI.delete()``.
        Unsupported URI types raise ``NotImplementedError``, which is caught and logged.

        Args:
            build_id: UUID of the build whose output artifacts should be removed.
        """
        artifacts = self.storage.artifact_registry.get_by_where(
            {"created_by_build_id": build_id}
        )
        for artifact in artifacts:
            uri = URI.get_uri(artifact.uri)
            try:
                if not uri.delete():
                    logger.warning(
                        f"Could not delete output artifact {URI.get_uristr(uri)}"
                    )
            except NotImplementedError:
                logger.warning(
                    f"Not deleting output artifact {URI.get_uristr(uri)}: delete not implemented for {type(uri).__name__}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not delete output artifact {URI.get_uristr(uri)}: {e}"
                )

    # def _get_test_config(self) -> BuildTestSpecification:
    #     raise ValueError(
    #         "Must be implemented by sub-class to define build file and other test config"
    #     )

    # def _run_build_tests(
    #     self: Self,
    #     tested_class: ClassTestedEnum,
    #     test_specs: list[BuildTestSpecification],
    #     test_cancel=False,
    #     build_count: int = 1,
    # ):
    #     """Convenience to run list of build tests calling _run_build()
    #     If any one of the tests fail, the whole pytest run fails.

    #     Args:
    #         tested_class (ClassTestedEnum): _description_
    #         test_specs (list[BuildTestSpecification]): _description_
    #         test_cancel (bool, optional): _description_. Defaults to False.
    #         build_count (int, optional): _description_. Defaults to 1.
    #     """
    #     for build_test_spec in test_specs:
    #         self._run_build_test(
    #             tested_class, build_test_spec, test_cancel, build_count
    #         )

    def _run_build_test(
        self: Self,
        tested_class: ClassTestedEnum,
        test_spec: BuildTestSpecification,
        test_cancel=False,
        build_count: int = 1,
    ):
        """Runs a single build test with controls over whether the BuildRunner, BuildRunnerJob or BuildWatcher is used to
        run test. Additional options control cancel/success testing and ability to run N simultaneous builds.
        All tested classes require a connection to a cluster (currently via oc login).  This can be done manually outside of the test
        or env vars can be used to set a an api key.  See below for api keys.

        Args:
            tested_class (ClassTestedEnum): specifies one of BuildRunner/Job/Watcher as the mechanism to run the build.
            For BuildRunner-based tests and not already logged into the compute cluster,
            one of GBTEST_COMPUTE_CLUSTER_API_KEY or GBTEST_COMPUTE_CLUSTER_TOKEN env vars must be set.
            For BuildRunnerJob/Watcher-base tests and if not already logged into the GB cluster,
            one of GBTEST_GB_CLUSTER_API_KEY or GBTEST_GB_CLUSTER_TOKEN env vars must be set, if not already loggedbbbb

            test_spec (BuildTestSpecification): specifies build.yaml, targets, and expected results.

            test_cancel (bool, optional): If true, then cancel the build in the middle of RUNNING status and confirm expected behavior.
            If False, the verify the SUCCESS status of build/target/steps and expected output artifacts. Defaults to False.

            build_count (int, optional): Allows for more than 1 build to be run at the same time.  Only available for BuildWatcher-based runs. Defaults to 1.
        """
        assert build_count > 0, "Test misconfigured. build_count must be larger than 0"
        self.class_tested = tested_class
        logger.info(f"Testing build {test_spec}")
        build_yaml = test_spec.build_yaml
        space = self._check_and_setup_space(test_spec)
        logger.info(f"Creating the build")
        stored_build = StoredBuild.create(
            name="test",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,  # No @, only characters allowed in openshift labels (for now)
            build_yaml_path=build_yaml,
            targets=test_spec.targets,
        )
        logger.info(f"Done creating the build")
        timeout_seconds = build_count * test_spec.timeout_minutes * 60

        if test_spec.simulate_step_failure:
            enable_failure_simulation()

        try:
            build_ids = []
            if (
                tested_class == ClassTestedEnum.TEST_BUILDRUNNER
                or tested_class == ClassTestedEnum.TEST_BUILDRUNNERJOB
            ):
                assert (
                    build_count == 1
                ), "Build runner/job tests do not support more than 1 build (yet)."
                stored_build.status = (
                    Status.PENDING
                )  #  BuildRunner handles, FAILED and PENDING.  Here we just do a normal PENDING.
                build_ids = self._run_build_test_build(
                    stored_build,
                    tested_class,
                    test_cancel,
                    test_spec.expected_status,
                    timeout_seconds,
                    space_uri=test_spec.space_uri,
                )
            elif tested_class == ClassTestedEnum.TEST_BUILDWATCHER:
                stored_build.status = (
                    Status.SUBMITTED
                )  #    Buildwatcher handles SUBMITTED builds.
                build_ids = self.__run_buildwatcher_test_build(
                    stored_build,
                    build_count,
                    test_cancel,
                    test_spec.expected_status,
                    timeout_seconds,
                )

            else:
                assert False, "Did not get a known test class type enum"

            self._verify_build_results(build_ids, test_spec, tested_class, test_cancel)

        finally:
            disable_failure_simulation()

    def _run_two_builds_sequential(
        self: Self,
        tested_class: ClassTestedEnum,
        test_spec: BuildTestSpecification,
        second_run_spec: BuildTestSpecification,
    ) -> None:
        """
        Runs the build twice to verify target-skip behaviour:
        1. First run: all targets execute normally and register their output artifacts.
        2. Second run: uses second_run_spec's build_yaml and targets; targets in
           skip_target_names are expected to be skipped because their target_hash already
           exists in gb_targets from the first run. Exercises downstream binding
           propagation from a skipped target.

        The artifact registry is NOT cleared between runs so first-run output artifacts
        remain visible to the second run's skip check.
        """
        assert (
            tested_class == ClassTestedEnum.TEST_BUILDRUNNER
        ), "_run_two_builds_sequential only supports TEST_BUILDRUNNER"

        # First run: run normally, verify expectations.
        self._run_build_test(tested_class=tested_class, test_spec=test_spec)

        # Second run: the first run stored a target_hash in gb_targets; the BuildRunner's
        # target_already_run_fn will find it and skip matching targets.
        # Output artifacts from the first run remain in the registry for the second run's skip check.
        self._run_build_test(
            tested_class=tested_class,
            test_spec=second_run_spec,
        )

    def _verify_build_cancellations(self: Self, build_ids: list[str]) -> None:
        for build_id in build_ids:
            self._verify_build_cancellation(build_id)

    def _verify_build_cancellation(self: Self, build_id: str) -> None:
        self._verify_build_status(build_id, [Status.CANCELLED])
        # Verify each target and step for those targets
        targets = self.storage.target_storage.get_by_where({"build_id": build_id})
        for target in targets:
            self._verify_target_status(
                build_id, target, [Status.SUCCESS, Status.CANCELLED]
            )

        # Make sure the artifacts got cancelled or were finished
        self._verify_build_artifact_status(
            build_id,
            [ArtifactRegistrationStatus.SUCCESS, ArtifactRegistrationStatus.CANCELLED],
        )

    def _verify_build_results(
        self: Self,
        build_ids: list[str],
        test_spec: BuildTestSpecification,
        tested_class: ClassTestedEnum,
        test_cancel: bool,
    ) -> None:
        if test_cancel:
            self._verify_build_cancellations(build_ids)
        else:
            self._verify_finished_builds_expectations(build_ids, test_spec)

        if (
            tested_class == ClassTestedEnum.TEST_BUILDWATCHER
            and GBSERVER_DEFAULT_BUILDRUNNER_TYPE == "job"
        ) or tested_class == ClassTestedEnum.TEST_BUILDRUNNERJOB:
            self._verify_pods_finished(build_ids)
        if test_spec.simulate_step_failure:
            self.__verify_simulated_step_retry_event(build_ids)

        if test_spec.space_uri is None:
            logger.info("Verifying build watcher pr creation. ")
            self._verify_prs(build_ids, test_spec.space_name)

    def __verify_simulated_step_retry_event(self: Self, build_ids: list[str]) -> None:
        for build_id in build_ids:
            simulate_events = self.storage.event_storage.get_sorted_build_events(
                build_id=build_id, where={"source": "simulate"}
            )
            if not simulate_events:
                # For now we only warn since not all environments may support this
                # TODO: We could make this somehow configured in the test config
                logger.warning(
                    "[simulate] Failure simulation was requested but no simulated event was "
                    "recorded in storage for build %s. "
                    "Check that retry is enabled and a RetryHandler was created.",
                    build_id,
                )

    def _run_build_test_build(
        self: Self,
        stored_build: StoredBuild,
        tested_class: ClassTestedEnum,
        test_cancel: bool,
        expected_status: Status,
        timeout_seconds: float,
        space_uri: Optional[str] = None,
    ) -> list[str]:
        # BuildRunner is only expected to handle builds that have these initial status values.
        assert stored_build.status in (
            Status.PENDING,
            Status.FAILED,
        ), "Unexpected build status"
        # Run the build directly via a BuildRunner or BuildRunnerJob
        build_ids: list[str] = [stored_build.uuid]
        if tested_class == ClassTestedEnum.TEST_BUILDRUNNER:
            runner = BuildRunner(
                build=stored_build, space_uri=space_uri, create_pr=space_uri is None
            )
        else:
            runner = BuildRunnerJob(build=stored_build)
        logger.info("Starting the Build object. ")
        self._wait_for_build_status_threaded(
            runner, build_ids, test_cancel, expected_status, timeout_seconds
        )
        return build_ids

    def _wait_for_build_status_threaded(
        self: Self,
        runner: BuildRunner | BuildRunnerJob,
        build_ids: list[str],
        test_cancel: bool,
        expected_status: Status,
        timeout_seconds: float,
    ) -> None:
        if test_cancel:
            thread = ExceptionRaisingThread(
                name="Wait and cancel",
                target=self._wait_for_cancelled,
                args=(runner, build_ids, timeout_seconds),
            )
        else:
            thread = ExceptionRaisingThread(
                name="Wait success",
                target=self._wait_for_builds_with_status,
                args=(runner, build_ids, expected_status, timeout_seconds),
            )
        logger.info("Starting the build waiting thread. ")
        thread.start()
        runner.start_and_wait()
        thread.join()  # This will raise the assert exceptions from the thread, if needed

    def __run_buildwatcher_test_build(
        self: Self,
        stored_build: StoredBuild,
        build_count: int,
        test_cancel: bool,
        expected_status: Status,
        timeout_seconds: float,
    ) -> list[str]:
        # BuildWatcher is only expected to handle builds that have this initial status values.
        assert stored_build.status in (Status.SUBMITTED), "Unexpected build status"
        # Store the build(s) in storage and expect the BuildWatcher to pick it up and run it using a BuildRunner.
        watcher = BuildWatcher()
        build_ids: list[str] = []
        for i in range(0, build_count):
            # Store a copy of the build but using a different uuid
            stored_build.uuid = get_uuid()
            # Bump the time of the build so we can (internally/manually) check FIFO processing of builds.
            stored_build.created_time = stored_build.created_time + timedelta(seconds=1)
            self.storage.build_storage.add(stored_build)
            build_ids.append(stored_build.uuid)
        logger.info("Done creating BuildWatcher. ")
        if test_cancel:
            thread = ExceptionRaisingThread(
                name="Wait and cancel",
                target=self._wait_for_cancelled,
                args=(watcher, build_ids, timeout_seconds),
            )
        else:
            thread = ExceptionRaisingThread(
                name="Wait success",
                target=self._wait_for_builds_with_status,
                args=(watcher, build_ids, expected_status, timeout_seconds),
            )
        logger.info("Starting the build watcher test thread. ")
        thread.start()
        logger.info("Starting the build watcher test thread. ")
        watcher.start_and_wait()  # This blocks until watcher.stop() is called in the thread
        thread.join()  # This will raise the assert exceptions from the thread, if needed
        return build_ids

    def _verify_pods_finished(self: Self, build_ids: list[str]) -> None:
        unfinished = []
        start_time = time()
        sleep_time = GBTEST_JOB_TERMINATION_TIMEOUT_SECONDS / 10
        while True:
            unfinished = []
            logger.info(
                "Waiting on the following jobs/pods to finish running: %s", build_ids
            )
            for build_id in build_ids:
                if not is_buildrunner_pod_finished(build_id):
                    unfinished.append(build_id)
            waited_time = time() - start_time
            if (
                len(unfinished) == 0
                or waited_time > GBTEST_JOB_TERMINATION_TIMEOUT_SECONDS
            ):
                break
            logger.info("sleeping for %s seconds", sleep_time)
            sleep(sleep_time)
        assert (
            len(unfinished) == 0
        ), f"Pods for the following builds are still running: {unfinished}"

    def _wait_for_running_then_cancel_one_build(self, build_id: str, timeout: float):
        """Wait for the build to be RUNNING then, immedidately request cancellation."""
        self._wait_for_build_status(build_id, [Status.RUNNING], timeout)
        # sleep(10)    # Wait for build to progress some - yes, we seem to have some bugs if we cancel right away.
        self._request_build_cancellation(build_id)

    def _wait_for_cancelled(
        self: Self,
        watcher_or_runner: Union[BuildWatcher, BuildRunner],
        build_ids: list[str],
        timeout: float,
    ):
        success = False
        try:
            # Wait for all to be RUNNING
            threads = []
            for build_id in build_ids:
                # Have each check in a separate thread to try and avoid serializing the cancellations
                thread = ExceptionRaisingThread(
                    name="Cancel 1 build",
                    target=self._wait_for_running_then_cancel_one_build,
                    args=(build_id, timeout),
                )
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # Now wait for them to become canceled.
            self._wait_for_canceled_builds(build_ids, timeout)

            success = True
        except Exception as e:
            assert False, f"Got exception waiting for job status, {e}"
        finally:  # In case the wait issues an assert failure
            if not success or isinstance(watcher_or_runner, BuildWatcher):
                # BuildWatcher does not return from stop_and_wait() when jobs are cancelled.
                watcher_or_runner.stop()
            # else the cancellation request above should cause the runner/runnerjob to return from start_and_wait().

    def _request_build_cancellation(self, build_id: str):
        build: StoredBuild = self.storage.build_storage.get_by_uuid(build_id)
        assert build is not None, f"failed to find a build with id {build_id}"
        assert (
            build.status == Status.RUNNING
        ), "Build was not running when cancellation is being requested"
        build.status = Status.CANCEL_REQUESTED
        self.storage.build_storage.update(build)

    def _wait_for_builds_with_status(
        self: Self,
        watcher_or_runner: Union[BuildWatcher, BuildRunner],
        build_ids: list[str],
        status: Status,
        timeout: float,
    ):
        try:
            for build_id in build_ids:
                self._wait_for_build_status(build_id, [status], timeout)
        finally:  # In case the wait issues an assert failure
            logger.info("Making sure the build is stopped (manually stopping)")
            watcher_or_runner.stop()

    def _wait_for_build_status(
        self: Self,
        build_id: str,
        statuses: list[Status],
        timeout: float,
        failed_statuses: Optional[list[Status]] = [],
    ) -> None:
        return self._wait_for_status(
            build_id=build_id,
            item_name="build",
            item_id=build_id,
            item_storage=self.storage.build_storage,
            statuses=statuses,
            timeout=timeout,
            failed_statuses=failed_statuses,
        )

    def _wait_for_target_status(
        self: Self,
        build_id: str,
        target_id: str,
        statuses: list[Status],
        timeout: float,
    ) -> None:
        return self._wait_for_status(
            build_id=build_id,
            item_name="target",
            item_id=target_id,
            item_storage=self.storage.target_storage,
            statuses=statuses,
            timeout=timeout,
        )

    def _wait_for_step_status(
        self: Self, build_id: str, step_id: str, statuses: list[Status], timeout: float
    ) -> None:
        return self._wait_for_status(
            build_id=build_id,
            item_name="step",
            item_id=step_id,
            item_storage=self.storage.step_storage,
            statuses=statuses,
            timeout=timeout,
        )

    def _get_item_statuses(self: Self, item_storage: IItemStorage) -> dict[str, Status]:
        statuses = {}
        items = item_storage.get_by_uuid(None)
        for item in items:
            statuses[item.uuid] = item.status.name
        return statuses

    def _wait_for_status(
        self: Self,
        build_id: str,
        item_name: str,
        item_id: str,
        item_storage: IItemStorage,
        statuses: list[Status],
        timeout: float,
        failed_statuses: Optional[list[Status]] = [],
    ) -> None:
        """Wait for one of the status values in the item in the given storage.


        Args:
            build_id: Build ID to include in assertion messages for debugging.
            item_id (_type_): _description_
            statuses (list[Status]): _description_
            timeout (float): _description_

        Returns:
            _type_: a bool that indicates if one of the statuses was found and either the matched status (if found),
            or the last status (if not found).
        """
        is_success = False
        last_status = None
        sleep_time = 5
        if timeout < sleep_time:
            sleep_time = timeout / 10
        time_waited = 0
        item = None
        start_time = time()
        while time_waited <= timeout:
            item = item_storage.get_by_uuid(item_id)
            status = self._get_item_statuses(item_storage)
            if (
                item is not None
            ):  # When using a BuildRunner, the build is stored by its start() method. This thread is started before that call.
                logger.info(
                    f"Looping on {item_name}:{item_id} status.  Currently {item.status}.  All others: {status.values()}"
                )
                last_status = item.status
                if last_status in statuses:
                    is_success = True
                    break
                if last_status in failed_statuses:
                    is_success = False
                    break
                match last_status:
                    case Status.FAILED | Status.CANCELLED | Status.INVALID:
                        # And fail the is_success assert below
                        break
                    case Status.CANCEL_REQUESTED if Status.CANCELLED not in statuses:
                        # Unexpected cancellation — fail fast instead of looping until timeout
                        break
                    case _:
                        pass
            else:
                logger.info(f"Looping on {item_name}:{item_id} status.  Not found yet.")
            sleep(sleep_time)
            now = time()
            time_waited = now - start_time
        assert time_waited <= timeout, self._failed_build_msg(
            build_id,
            f"we exceeded the timeout of {timeout} seconds looking for item {item_name}:{item_id} in {item_storage}",
        )
        assert item is not None, self._failed_build_msg(
            build_id,
            f"{item_name}:{item_id} was not found. It likely did not generate an event in the expected {timeout} seconds.",
        )
        assert is_success, self._failed_build_msg(
            build_id,
            f"{item_name}:{item_id} did not achieve any of the success {statuses} status(es). Last status was {last_status}",
        )

    def _wait_for_canceled_builds(self: Self, build_ids: list[str], timeout: float):
        assert len(build_ids) > 0, "No build ids to verify"
        for build_id in build_ids:
            self._wait_for_build_status(
                build_id=build_id,
                statuses=[Status.CANCELLED],
                timeout=timeout,
                failed_statuses=[Status.SUCCESS],
            )
            target_list = self.storage.target_storage.get_by_where(
                {"build_id": build_id}
            )
            for target in target_list:
                self._wait_for_canceled_target_and_steps(target, timeout)

    def _wait_for_canceled_target_and_steps(
        self: Self, target: StoredTargetRun, timeout
    ):
        build_id = target.build_id
        self._wait_for_target_status(
            build_id, target.uuid, [Status.CANCELLED, Status.SUCCESS], timeout
        )
        # assert target.status == Status.CANCELLED or target.status == Status.SUCCESS, f"Target was not canceled, but was {target.status}"
        step_list = self.storage.step_storage.get_by_where({"target_id": target.uuid})
        for step in step_list:
            self._wait_for_step_status(
                build_id, step.uuid, [Status.CANCELLED, Status.SUCCESS], timeout
            )
            # assert isinstance(step,StoredStepRun)
            # assert step.status == Status.CANCELLED or step.status == Status.SUCCESS, f"Step was not canceled, but was {step.status}"

    def _verify_finished_builds_expectations(
        self: Self, build_ids: list[str], test_spec: BuildTestSpecification
    ) -> None:
        assert len(build_ids) > 0, "No build ids to verify"
        for build_id in build_ids:
            self._verify_finished_build_expectations(build_id, test_spec)

    def _verify_build_status(self: Self, build_id: str, status_list: list[Status]):
        build = self.storage.build_storage.get_by_uuid(build_id)
        assert build, self._failed_build_msg(
            build_id, f"Could not find build with id {build_id}"
        )
        assert build.status in status_list, self._failed_build_msg(
            build_id,
            f"Build has status {build.status}, but expected one of {status_list}",
        )

    def _verify_finished_build_expectations(
        self: Self, build_id: str, test_spec: BuildTestSpecification
    ) -> None:
        """Verify a build that finished w/o cancellation.

        Args:
            self (Self): _description_
            build_id (str): _description_
            expected (list[ExpectedTarget]): _description_
        """
        # Make sure the build has the expected status
        self._verify_build_status(build_id, [test_spec.expected_status])

        if test_spec.expected_status == Status.INVALID:
            return  # TODO what can we verify here.

        # target_list = self._verify_target_status(build_id, [Status.SUCCESS])
        target_list = self.storage.target_storage.get_by_where({"build_id": build_id})
        # This list comes back unordered, so make accessible by name
        target_dict = {}
        for target in target_list:
            assert isinstance(target, StoredTargetRun)
            target_dict[target.name] = target

        skip_set = set(test_spec.skip_target_names)
        expected_targets = test_spec.target_expectations
        for index in range(0, len(expected_targets)):
            expected_target = expected_targets[index]
            target = target_dict.get(expected_target.target_name)
            assert target is not None, self._failed_build_msg(
                build_id,
                f"Did not find expected target named {expected_target.target_name}",
            )
            if expected_target.target_name in skip_set:
                self._verify_skipped_target_and_steps(
                    build_id, target, [Status.SUCCESS], expected_target
                )
            else:
                self._verify_unskipped_target_and_steps(
                    build_id, target, [Status.SUCCESS], expected_target
                )

        # Do this after in case there are extra targets
        assert len(target_list) == len(expected_targets), self._failed_build_msg(
            build_id,
            f"Number of built targets does not match the expected number, actual: {len(target_list)} expected: {len(expected_targets)}",
        )

        # Verify no artifacts created by this build have PENDING status
        self._verify_build_artifact_status(
            build_id, [ArtifactRegistrationStatus.SUCCESS]
        )

    def _verify_build_artifact_status(
        self: Self, build_id: str, status_list: list[ArtifactRegistrationStatus]
    ) -> None:
        """Verify that no artifacts created by this build have PENDING status."""
        artifacts = self.storage.artifact_registry.get_by_where(
            {"created_by_build_id": build_id}
        )
        for artifact in artifacts:
            assert artifact.status in status_list, self._failed_build_msg(
                build_id,
                f"Artifact status is {artifact.status}, but expected one of {status_list}",
            )
            try:
                uri = URI.get_uri(artifact.uri)
            except Exception as e:
                raise AssertionError(
                    f"Could not resolve artifact uri {artifact.uri} to an object: {e}"
                ) from e
            assert (
                uri.exists()
            ), f"URI {artifact.uri} does not exist in artifact storage"

    def _verify_skipped_target_and_steps(
        self: Self,
        build_id: str,
        built_target: StoredTargetRun,
        status_list: list[Status],
        expected: ExpectedTarget,
    ):
        self._verify_target_status(build_id, built_target, status_list)
        assert len(built_target.input_artifacts) == 0, self._failed_build_msg(
            build_id,
            f"Skipped target '{built_target.name}' should have 0 input artifacts but has {len(built_target.input_artifacts)}",
        )
        assert len(built_target.output_artifacts) == 0, self._failed_build_msg(
            build_id,
            f"Skipped target '{built_target.name}' should have 0 output artifacts but has {len(built_target.output_artifacts)}",
        )
        step_list = self.storage.step_storage.get_by_where(
            {"target_id": built_target.uuid}
        )
        assert len(step_list) == 0, self._failed_build_msg(
            build_id,
            f"Skipped target '{built_target.name}' should have 0 steps but has {len(step_list)}",
        )
        count = get_lineage_store().count_release_ids(
            release_id=built_target.build_id,
            target_id=built_target.uuid,
        )
        assert count == expected.jobstats_count, self._failed_build_msg(
            build_id,
            f"Skipped target {built_target.name} created {count} jobstats, but expected  {expected.jobstats_count}",
        )

    def _verify_unskipped_target_and_steps(
        self: Self,
        build_id: str,
        built_target: StoredTargetRun,
        status_list: list[Status],
        expected: ExpectedTarget,
    ):
        self._verify_target_status(build_id, built_target, status_list)

        # Check the number of input and output artifacts
        assert (
            len(built_target.input_artifacts) == expected.input_artifact_count
        ), self._failed_build_msg(
            build_id,
            f"actual: {built_target.input_artifacts} expected: {expected.input_artifact_count}",
        )
        assert (
            len(built_target.output_artifacts) == expected.output_artifact_count
        ), self._failed_build_msg(
            build_id,
            f"actual: {len(built_target.output_artifacts)} expected: {expected.output_artifact_count}",
        )
        # Verify each artifact is present and has the expected status
        for name, uuids in built_target.output_artifacts.items():
            for uuid in uuids:
                artifact = self.storage.artifact_registry.get_by_uuid(uuid)
                assert artifact is not None, self._failed_build_msg(
                    build_id,
                    f"Did not find artifact from {built_target.name}.{name} for uuid {uuid}",
                )
                assert isinstance(artifact, ArtifactRegistration)
                assert (
                    artifact.status == ArtifactRegistrationStatus.SUCCESS
                ), self._failed_build_msg(
                    build_id,
                    f"stored artifact {uuid} is not marked as success, status is {artifact.status}",
                )
                assert artifact.type != ArtifactType.UNDEFINED, self._failed_build_msg(
                    build_id, f"stored artifact {uuid} has an undefined type"
                )

        # Verify the number of expected steps and their status values
        step_list = self.storage.step_storage.get_by_where(
            {"target_id": built_target.uuid}
        )
        if expected.step_count <= 0:
            logger.warning(
                f"Not verifying step count for target {built_target.name} because verified step count is <=0"
            )
        else:
            assert len(step_list) == expected.step_count, self._failed_build_msg(
                build_id, f"actual: {len(step_list)} expected: {expected.step_count}"
            )
        self._verify_steplist_status(build_id, step_list, status_list)

        # Verify the number of jobstats/lineage records matches the expected count.
        # Retry a few times because Lakehouse writes from the K8s build runner pod
        # may not be immediately visible to a query in this process.
        count = 0
        for attempt in range(5):
            count = get_lineage_store().count_release_ids(
                release_id=built_target.build_id,
                target_id=built_target.uuid,
            )
            if count == expected.jobstats_count:
                break
            if attempt < 4:
                logger.info(
                    "JobStats count %d != expected %d for target %s, retrying (%d/5)...",
                    count,
                    expected.jobstats_count,
                    built_target.name,
                    attempt + 1,
                )
                sleep(3)
        assert count == expected.jobstats_count, self._failed_build_msg(
            build_id=built_target.build_id,
            message=f"Target {built_target.name} created {count} JobStats, expected {expected.jobstats_count} ",
        )

    def _verify_target_status(
        self, build_id: str, target: StoredTargetRun, status_list: list[Status]
    ) -> None:
        assert isinstance(target, StoredTargetRun)
        assert target.status in status_list, self._failed_build_msg(
            build_id,
            f"Status of target {target.name} is {target.status} but one of {status_list} was expected",
        )
        self._verify_target_steps(build_id, target.uuid, status_list)

    def _verify_target_steps(
        self: Self, build_id: str, target_id, status_list: list[Status]
    ):
        steps = self.storage.step_storage.get_by_where({"target_id": target_id})
        self._verify_steplist_status(build_id, steps, status_list)

    def _verify_steplist_status(
        self: Self,
        build_id: str,
        step_list: list[StoredStepRun],
        status_list: list[Status],
    ) -> None:
        for step in step_list:
            assert isinstance(step, StoredStepRun)
            assert step.status in status_list, self._failed_build_msg(
                build_id,
                f"Status of step {step.definition_uri} is {step.status} but one of {status_list} was expected",
            )

    def _verify_prs(self: Self, build_ids: List[str], space_name: str) -> None:
        gh_token = GBSERVER_GITHUB_TOKEN
        assert gh_token is not None, "Can't pass this test w/o a github_token"
        space_storage: SQLSpaceStorage = self.storage.space_storage
        stored_space = space_storage.get_by_name(name=space_name)
        assert stored_space is not None
        assert isinstance(stored_space, StoredSpace)
        git_uri = stored_space.git_repo_uri
        git_uri_parts = get_uri_parts(git_uri)
        # (scheme, domain, owner, repo, sub_directory)
        owner = git_uri_parts[2]
        repo = git_uri_parts[3]
        myghapi = MyGHApi(token=gh_token, owner=owner, repo=repo)

        verify_pr_timeout = 10 * 60  # 10 minutes
        sleep_time = 5  # 5 seconds
        start_time = get_time()
        curr_time = get_time()
        time_elapsed = curr_time - start_time
        to_check = set(build_ids)
        checked = set()

        while len(to_check) > 0 and time_elapsed.total_seconds() < verify_pr_timeout:
            for build_id in to_check:
                logger.info("checking PR for build %s", build_id)
                stored_build = self.storage.build_storage.get_by_uuid(build_id)
                assert (
                    stored_build is not None
                ), f"Did not find expected build under id {build_id}"
                assert isinstance(stored_build, StoredBuild)
                pr_url = stored_build.source_uri
                if pr_url == "":
                    logger.info(
                        "Build %s was not assigned a PR uri, skip and check later",
                        build_id,
                    )
                    continue
                pr_num = -1
                try:
                    pr_num = int(pr_url.split("/")[-1], base=10)
                    assert (
                        pr_num > 0
                    ), f"Could not get a valid PR number from PR URL provided by the stored build: '{pr_url}'"
                except Exception as e:
                    raise ValueError(
                        f"build_id {build_id} invalid pr_url: {pr_url}"
                    ) from e
                pr_id = str(pr_num)
                pr = myghapi.get_pr(pr_id=pr_id)
                build_user = stored_build.username or "<unknown>"
                expected_title = f"run: user `{build_user}` build `{build_id}`"
                if stored_build.name:
                    expected_title += f" `{stored_build.name}`"
                assert (
                    pr.title == expected_title
                ), f"PR title is incorrect, expected: {expected_title} actual: {pr.title}"
                checked.add(build_id)
                logger.info("build %s passed PR checks", build_id)
            to_check = to_check - checked
            logger.info("sleeping for %d seconds...", sleep_time)
            sleep(sleep_time)
            curr_time = get_time()
            time_elapsed = curr_time - start_time

        if len(to_check) > 0:
            raise ValueError(
                f"Timeout! The following build ids didn't get a PR assigned: {to_check}"
            )


@pytest.mark.skipif(
    os.environ.get("GBTEST_HAS_COMPUTE_CLUSTER_ACCESS", "True").lower() == "false"
    or os.environ.get("HAS_COMPUTE_CLUSTER_ACCESS", "True").lower() == "false",
    reason="Can't run this since it is configured as not having compute cluster access",
)
class AbstractBuildRunnerTest(AbstractBuildTest):

    def setup_method(self, method):
        """Always runs locally via BuildRunner (TEST_BUILDRUNNER), never needs cluster login."""
        self.run_locally = True
        super().setup_method(method)

    @abstractmethod
    def _get_test_specification(self: Self) -> BuildTestSpecification:
        raise ValueError("Sub-class must implement this method")

    def test_runner(self):
        test_spec = self._get_test_specification()
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=test_spec,
            test_cancel=False,
        )


class AbstractYamlBuildRunnerTest(AbstractBuildRunnerTest):
    """YAML-driven variant of AbstractBuildRunnerTest.

    Concrete subclasses override exactly one method, ``_get_yaml_spec_dir``,
    which returns the directory containing this fixture's ``build.yaml`` and
    ``buildtest.yaml``.  This base class implements ``_get_test_specification``
    (loads the YAML) and provides ``test_runner`` + ``test_runner_cancellation``
    methods that consult the spec's ``tests:`` list and skip themselves when
    the YAML doesn't opt in.

    Example:
        @extended_testing_only
        @pytest.mark.xdist_group(name="buildtest_cpu")
        class TestBuildRunner1StepCPU(AbstractYamlBuildRunnerTest):
            def _get_yaml_spec_dir(self) -> Path:
                return get_test_data_dir_for(__file__) / "1step/cpu"
    """

    @abstractmethod
    def _get_yaml_spec_dir(self: Self) -> Path:
        """Return the directory containing this fixture's build.yaml and buildtest.yaml."""
        raise NotImplementedError

    def _get_test_specification(self: Self) -> BuildTestSpecification:
        """Load the spec from <spec_dir>/buildtest.yaml.

        Implemented here so subclasses don't need to repeat the from_yaml call —
        the only contract for a concrete class is to override
        ``_get_yaml_spec_dir``.
        """
        return BuildTestSpecification.from_yaml(
            self._get_yaml_spec_dir() / "buildtest.yaml"
        )

    def _run_yaml_spec(self: Self, test_key: str, test_cancel: bool) -> None:
        """Shared body for the YAML-driven test_<key> methods.

        Loads the spec, skips if ``test_key`` is not in the spec's ``tests``
        list, and otherwise dispatches to ``_run_build_test``.
        """
        spec = self._get_test_specification()
        if test_key not in spec.tests:
            pytest.skip(f"YAML did not list '{test_key}' in tests: {spec.tests}")
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=spec,
            test_cancel=test_cancel,
        )

    def test_runner(self):
        """Run the basic build (test_cancel=False).

        Skipped if the YAML's ``tests:`` list does not include ``'runner'``.
        Overrides the inherited AbstractBuildRunnerTest.test_runner so the
        spec's tests list controls both methods symmetrically.
        """
        self._run_yaml_spec("runner", test_cancel=False)

    def test_runner_cancellation(self):
        """Run with the cancellation path (test_cancel=True).

        Skipped if the YAML's ``tests:`` list does not include
        ``'runner_cancellation'``.
        """
        self._run_yaml_spec("runner_cancellation", test_cancel=True)
