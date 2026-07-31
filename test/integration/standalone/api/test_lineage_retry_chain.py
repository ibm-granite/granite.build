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

"""API tests for build lineage across a retry chain.

GET /lineage/build/{id} reports only the queried build by default, so the root's
graph is full of failures and a retry's graph has holes where targets were
skipped for reuse. With follow_retries=true it reports one entry per spec target,
taking the run that actually produced the artifacts; failed and never-dispatched
targets are omitted.

In standalone mode the lineage store is a no-op that returns an empty jobstats
dict per target, so these tests assert on the number of emitted entries and on
job_id, not on backend payload internals.
"""

from types import SimpleNamespace
from typing import Self

import pytest
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.api.lineage import get_build_jobstats
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

pytestmark = pytest.mark.standalone


def _owner_request() -> SimpleNamespace:
    """A fake Request whose caller is 'tester', the owner of every build in
    this test's chains, so authorize_build_read_access lets the calls through."""
    return SimpleNamespace(
        state=SimpleNamespace(
            data={"user": SimpleNamespace(login="tester", email="tester@example.com")}
        )
    )


class TestLineageRetryChain(AbstractSingletonStorageUsingTest):
    """get_build_jobstats follows the retry chain only when asked to."""

    def _add_build(
        self: Self,
        status: Status,
        retry_count: int,
        retry_of_build_id=None,
        targets=None,
    ) -> StoredBuild:
        build = StoredBuild(
            name="test",
            space_name="testspace",
            source_uri="",
            username="tester",
            status=status,
            retry_count=retry_count,
            retry_of_build_id=retry_of_build_id,
            targets=targets,
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
        root = self._add_build(Status.FAILED, 0, targets=["targetA", "targetB"])
        retry = self._add_build(
            Status.SUCCESS,
            1,
            retry_of_build_id=root.uuid,
            targets=["targetA", "targetB"],
        )
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

    def test_no_follow_reports_only_the_queried_build(self: Self):
        _root, retry, _root_a = self._make_chain()

        resp = get_build_jobstats(_owner_request(), retry.uuid)

        assert resp.job_id is None
        # Unchanged behaviour: both of the retry's own runs, including the
        # skipped one that produced nothing.
        assert len(resp.targets) == 2

    def test_follow_reports_one_entry_per_spec_target(self: Self):
        root, retry, _root_a = self._make_chain()

        resp = get_build_jobstats(_owner_request(), retry.uuid, follow_retries=True)

        assert resp.job_id == root.uuid
        assert resp.build_id == retry.uuid
        # One entry per spec target, no duplicates across the two attempts.
        assert len(resp.targets) == 2

    def test_follow_reports_the_same_graph_regardless_of_which_member_queried(
        self: Self,
    ):
        root, retry, _root_a = self._make_chain()

        by_root = get_build_jobstats(_owner_request(), root.uuid, follow_retries=True)
        by_retry = get_build_jobstats(_owner_request(), retry.uuid, follow_retries=True)

        assert by_root.job_id == by_retry.job_id == root.uuid
        assert len(by_root.targets) == len(by_retry.targets) == 2

    def test_follow_omits_targets_that_produced_nothing(self: Self):
        root = self._add_build(
            Status.FAILED, 0, targets=["targetA", "targetB", "targetC"]
        )
        self._add_target(
            root.uuid, "targetA", Status.SUCCESS, "2020-01-01T00:00:00.000Z"
        )
        self._add_target(
            root.uuid, "targetB", Status.FAILED, "2020-01-01T00:01:00.000Z"
        )

        resp = get_build_jobstats(_owner_request(), root.uuid, follow_retries=True)

        # Only targetA produced artifacts; the failed and never-run targets have
        # no provenance to report.
        assert resp.job_id == root.uuid
        assert len(resp.targets) == 1
