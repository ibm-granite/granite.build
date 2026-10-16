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

"""API tests for the build status endpoint under in-place retry.

With in-place retry a build keeps a single build id across attempts, so
``GET /{build_id}/status`` returns the one build with every target run recorded
against it — a target that failed and was re-run has both a FAILED and a SUCCESS
run, the latter linked to the former via ``retry_of_target_id``. There is no
retry chain to follow. ``/status2`` is retained as a backward-compatible alias.
"""

from types import SimpleNamespace
from typing import Self

import pytest
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.api.builds import get_build_status, get_build_status2
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone


def _owner_request() -> SimpleNamespace:
    """A fake Request whose caller is 'tester', the owner of the build in this
    test, so authorize_build_read_access lets the calls through."""
    return SimpleNamespace(
        state=SimpleNamespace(
            data={"user": SimpleNamespace(login="tester", email="tester@example.com")}
        )
    )


class TestBuildStatusInPlaceRetry(AbstractSingletonStorageUsingTest):
    """get_build_status reports one build with its FAILED/SUCCESS target runs."""

    def _add_build(self: Self, status: Status, retry_count: int) -> StoredBuild:
        build = StoredBuild(
            name="test",
            space_name="testspace",
            source_uri="",
            username="tester",
            status=status,
            retry_count=retry_count,
        )
        self.storage.build_storage.add(build)
        return build

    def _add_target(
        self: Self,
        build_id: str,
        name: str,
        status: Status,
        started_at,
        retry_of_target_id: str = "",
    ) -> StoredTargetRun:
        target = StoredTargetRun(
            name=name,
            build_id=build_id,
            environment_uri="space://environments/bash",
            status=status,
            started_at=started_at,
            retry_of_target_id=retry_of_target_id,
        )
        self.storage.target_storage.add(target)
        return target

    def _make_retried_build(self: Self):
        """One build (retry_count 1): targetA succeeded once; targetB failed then
        re-ran to success, the SUCCESS run linking back to the FAILED run."""
        build = self._add_build(Status.SUCCESS, 1)
        self._add_target(
            build.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00.000Z"
        )
        b_failed = self._add_target(
            build.uuid, "targetB", Status.FAILED, "2020-01-01T00:01:00.000Z"
        )
        b_success = self._add_target(
            build.uuid,
            "targetB",
            Status.SUCCESS,
            "2020-01-01T00:02:00.000Z",
            retry_of_target_id=b_failed.uuid,
        )
        return build, b_failed, b_success

    def test_status_returns_single_build_with_all_runs(self: Self):
        build, b_failed, b_success = self._make_retried_build()

        resp = get_build_status(_owner_request(), build.uuid)

        assert resp.status.build.uuid == build.uuid
        # Every run lives on the one build id — there is no chain.
        assert {tr.target.build_id for tr in resp.status.target_runs} == {build.uuid}
        assert len(resp.status.target_runs) == 3

    def test_success_run_links_back_to_failed_run(self: Self):
        build, b_failed, b_success = self._make_retried_build()

        resp = get_build_status(_owner_request(), build.uuid)

        by_uuid = {tr.target.uuid: tr.target for tr in resp.status.target_runs}
        # The re-run SUCCESS run points at the prior FAILED run of the same target.
        assert by_uuid[b_success.uuid].retry_of_target_id == b_failed.uuid
        # The original FAILED run carries no linkage.
        assert by_uuid[b_failed.uuid].retry_of_target_id == ""

    def test_status2_alias_matches_status(self: Self):
        build, _b_failed, _b_success = self._make_retried_build()

        primary = get_build_status(_owner_request(), build.uuid)
        alias = get_build_status2(_owner_request(), build.uuid)

        assert alias.status.build.uuid == primary.status.build.uuid
        assert {tr.target.uuid for tr in alias.status.target_runs} == {
            tr.target.uuid for tr in primary.status.target_runs
        }
