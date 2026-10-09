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

"""Standalone BuildWatcher tests on the in-process Bash environment.

The standalone counterpart of test/integration/ibm/buildwatcher/test_buildwatcher.py:
the same run/cancel/invalid/multi-build cases, but the builds run in the local
Bash environment through a thread BuildRunner, against the repo's local space,
with no GitHub token (so no PRs). No IBM cloud, cluster, or image-tag
dependencies. There is no GPU variant (Bash has no GPU).
"""

from abc import abstractmethod
from typing import Self

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractBuildTest,
    BuildTestSpecification,
    ClassTestedEnum,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from gbserver.buildwatcher.buildwatcher import BuildWatcher

# These run the Bash environment in-process, which refuses to start outside
# STANDALONE; the test suite runs as GB_ENVIRONMENT=DEV.
pytestmark = [
    pytest.mark.standalone,
    pytest.mark.usefixtures("allow_bash_environment"),
]

_FIXTURES = get_test_data_dir_for(__file__)
_ONE_STEP_YAML = _FIXTURES / "1step" / "buildtest.yaml"
_INVALID_YAML = _FIXTURES / "invalid" / "buildtest.yaml"


# Real BuildWatcher runs (~90s total) — only run in the extended suite
# (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
class AbstractTestStandaloneBuildWatcher(AbstractBuildTest):
    """Run and cancel builds through a thread-runner BuildWatcher (no cluster/GitHub)."""

    def setup_method(self: Self, method):
        # Always run locally via the thread BuildRunner — no cluster login.
        self.run_locally = True
        super().setup_method(method)

    def _create_build_watcher(
        self: Self, test_spec: BuildTestSpecification
    ) -> BuildWatcher:
        """Create an in-process BuildWatcher that needs no GitHub or cluster.

        Args:
            test_spec: The test specification; its space_uri is the space every
                build the watcher runs resolves space:// URIs from.

        Returns:
            A BuildWatcher using the thread runner, fast polling, and no PRs.
        """
        watcher = BuildWatcher(gh_token="", all_build_space_uri=test_spec.space_uri)
        watcher.config.buildrunner_type = "thread"
        # 1s is the minimum interval; fast polling keeps the tests quick.
        watcher.config.monitoring_interval = 1
        return watcher

    def _get_build_count(self: Self) -> int:
        return 1

    @abstractmethod
    def _get_test_config(self: Self) -> BuildTestSpecification:
        raise NotImplementedError("Must provide test config")

    def test_build_watcher_run(self: Self):
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDWATCHER,
            test_spec=self._get_test_config(),
            test_cancel=False,
            build_count=self._get_build_count(),
        )

    def test_build_watcher_cancel(self: Self):
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDWATCHER,
            test_spec=self._get_test_config(),
            test_cancel=True,
            build_count=self._get_build_count(),
        )


@pytest.mark.xdist_group(name="standalone_buildwatcher_1step")
class TestStandaloneBuildWatcher1Step(AbstractTestStandaloneBuildWatcher):
    def _get_test_config(self: Self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(_ONE_STEP_YAML)


@pytest.mark.xdist_group(name="standalone_buildwatcher_invalid_build")
class TestStandaloneBuildWatcherInvalidBuild(AbstractTestStandaloneBuildWatcher):
    @pytest.mark.skip(
        reason="No need to run this for an invalid build (and cancel test does not support invalid builds)."
    )
    def test_build_watcher_cancel(self: Self):
        pass

    def _get_test_config(self: Self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(_INVALID_YAML)


@pytest.mark.xdist_group(name="standalone_buildwatcher_multi")
class TestStandaloneBuildWatcherMulti(AbstractTestStandaloneBuildWatcher):
    """Run and cancel simultaneous builds through one BuildWatcher."""

    def _get_build_count(self: Self) -> int:
        return 3

    def _get_test_config(self: Self) -> BuildTestSpecification:
        return BuildTestSpecification.from_yaml(_ONE_STEP_YAML)
