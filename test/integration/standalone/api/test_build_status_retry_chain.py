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

"""API tests for the build status endpoint's retry-chain following.

``GET /{build_id}/status`` returns just the queried build by default. With
``follow_retries=true`` it additionally returns ``retry_chain`` — every build in
the chain, root-first, each with its target runs. ``/status2`` is retained as a
backward-compatible alias.
"""

from typing import Self

import pytest
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.api.builds import get_build_status, get_build_status2
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone


class TestBuildStatusRetryChain(AbstractSingletonStorageUsingTest):
    """get_build_status follows the retry chain only when asked to."""

    def _add_build(
        self: Self, status: Status, retry_count: int, retry_of_build_id=None
    ) -> StoredBuild:
        build = StoredBuild(
            name="test",
            space_name="testspace",
            source_uri="",
            username="tester",
            status=status,
            retry_count=retry_count,
            retry_of_build_id=retry_of_build_id,
        )
        self.storage.build_storage.add(build)
        return build

    def _add_target(
        self: Self,
        build_id: str,
        name: str,
        status: Status,
        started_at,
        skipped_for_prerun_target_id: str = "",
    ) -> StoredTargetRun:
        target = StoredTargetRun(
            name=name,
            build_id=build_id,
            environment_uri="space://environments/bash",
            status=status,
            started_at=started_at,
            skipped_for_prerun_target_id=skipped_for_prerun_target_id,
        )
        self.storage.target_storage.add(target)
        return target

    def _make_chain(self: Self):
        """root (succeeded targetA, failed targetB) -> retry (skipped targetA, ok targetB)."""
        root = self._add_build(Status.FAILED, 0)
        retry = self._add_build(Status.SUCCESS, 1, retry_of_build_id=root.uuid)
        root.retry_build_id = retry.uuid
        self.storage.build_storage.update(root)

        root_a = self._add_target(
            root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00.000Z"
        )
        self._add_target(
            root.uuid, "targetB", Status.FAILED, "2020-01-01T00:01:00.000Z"
        )
        # targetA is reused (skipped) in the retry; skipped runs have no start time.
        self._add_target(
            retry.uuid,
            "targetA",
            Status.SUCCESS,
            None,
            skipped_for_prerun_target_id=root_a.uuid,
        )
        self._add_target(
            retry.uuid, "targetB", Status.SUCCESS, "2020-01-01T00:02:00.000Z"
        )
        return root, retry, root_a

    def test_no_follow_returns_only_queried_build(self: Self):
        _root, retry, _root_a = self._make_chain()

        resp = get_build_status(retry.uuid)

        assert resp.retry_chain is None
        assert resp.status.build.uuid == retry.uuid
        assert {tr.target.build_id for tr in resp.status.target_runs} == {retry.uuid}

    def test_follow_returns_chain_root_first(self: Self):
        root, retry, root_a = self._make_chain()

        resp = get_build_status(retry.uuid, follow_retries=True)

        assert resp.retry_chain is not None
        # Root first, then the retry — regardless of which member was queried.
        assert [m.build.uuid for m in resp.retry_chain] == [root.uuid, retry.uuid]

        # The skipped target in the retry points back at the original run.
        retry_member = resp.retry_chain[1]
        skipped = [
            tr
            for tr in retry_member.target_runs
            if tr.target.skipped_for_prerun_target_id
        ]
        assert len(skipped) == 1
        assert skipped[0].target.skipped_for_prerun_target_id == root_a.uuid

    def test_status2_alias_matches_status(self: Self):
        _root, retry, _root_a = self._make_chain()

        primary = get_build_status(retry.uuid, follow_retries=True)
        alias = get_build_status2(retry.uuid, follow_retries=True)

        assert [m.build.uuid for m in alias.retry_chain] == [
            m.build.uuid for m in primary.retry_chain
        ]
