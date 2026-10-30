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

"""``mem://`` artifacts are kept out of recorded lineage (#561).

A ``mem://`` artifact counts as absent: a target whose only outputs are mem:// is
recorded as a no-output target when it has real inputs, and skipped as
artifact-less when it has none. The selector, ``expected_run_count`` and the
emitter must all agree on that, so each case checks all three together.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.indexer import TargetLineageIndexer
from gbserver.lineage.lineage_reconciler import (
    expected_run_count,
    reconcile_build,
    select_recordable_views,
)
from gbserver.lineage.mem_artifacts import (
    is_mem_resource,
    is_mem_uri,
    lineage_view,
)
from gbserver.lineage.wandb_jobstats import WandBLineageStore
from gbserver.lineage.wandb_service import WandBLineageService
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.artifact import ArtifactType
from gbserver.types.status import Status

_FINISHED = datetime(2026, 1, 1, tzinfo=timezone.utc)

_URIS = {
    "mem-1": "mem://mem_output1",
    "mem-2": "mem://mem_output2",
    "real-in": "hf://datasets/org/data",
    "real-out": "s3://bucket/model",
    "real-out-2": "s3://bucket/model-2",
}


def _artifact(uuid: str) -> ArtifactRegistration:
    artifact = ArtifactRegistration(
        type=ArtifactType.DATASET,
        uri=_URIS[uuid],
        space_name="sp",
        username="u",
        name=uuid,
    )
    artifact.uuid = uuid
    return artifact


def _storage(target: StoredTargetRun) -> MagicMock:
    """Admin storage holding one build with ``target`` and the known artifacts."""
    storage = MagicMock()
    storage.artifact_registry.get_by_uuid.side_effect = lambda u: (
        _artifact(u) if u in _URIS else None
    )
    storage.target_storage.get_by_where.side_effect = (
        lambda where, query_control=None: (
            [target] if query_control.pagination.index == 0 else []
        )
    )
    storage.step_storage.get_by_where.return_value = []
    return storage


def _target(inputs=None, outputs=None) -> StoredTargetRun:
    return StoredTargetRun(
        uuid="t1",
        build_id="b1",
        name="t1",
        environment_uri="env://test",
        status=Status.SUCCESS,
        finished_at=_FINISHED,
        input_artifacts=inputs or {},
        output_artifacts=outputs or {},
    )


def _build() -> StoredBuild:
    build = StoredBuild(name="b", space_name="sp", source_uri="https://x", username="u")
    build.uuid = "b1"
    return build


def _emit(storage, target):
    store = WandBLineageStore.__new__(WandBLineageStore)
    events, _ = store._build_events_for_target(storage, _build(), target)
    return events


def _uris(resources) -> list[str]:
    return [r["uri"] for r in resources]


class TestUriChecks:
    def test_mem_uri(self):
        assert is_mem_uri("mem://mem_output1")
        assert is_mem_uri("MEM://mem_output1")
        assert not is_mem_uri("s3://bucket/mem://x")
        assert not is_mem_uri(None)

    @pytest.mark.parametrize(
        "resource",
        [
            {"uri": "mem://x"},
            {"facets": {"artifact_uri": "mem://x"}},
            {"facets": {"gb-artifact-uri": "mem://x"}},
        ],
    )
    def test_mem_resource_reads_uri_and_facets(self, resource):
        assert is_mem_resource(resource)

    def test_real_resource(self):
        assert not is_mem_resource(
            {"uri": "s3://b/k", "facets": {"artifact_uri": "s3://b/k"}}
        )


class TestTargetShapes:
    """Each row of the #561 table: selector, count and emitter agree."""

    def test_only_mem_outputs_no_inputs_is_skipped(self):
        target = _target(outputs={"svc": ["mem-1"]})
        storage = _storage(target)

        assert select_recordable_views(storage, "b1") == []
        assert _emit(storage, target) == []

    def test_only_mem_outputs_with_inputs_is_a_no_output_target(self):
        target = _target(
            inputs={"data": "real-in"}, outputs={"svc": ["mem-1", "mem-2"]}
        )
        storage = _storage(target)

        [(selected, _)] = select_recordable_views(storage, "b1")
        events = _emit(storage, target)

        assert expected_run_count(storage, selected) == 1 == len(events)
        assert _uris(events[0]["inputs"]) == ["hf://datasets/org/data"]
        assert events[0]["outputs"] == []

    def test_only_mem_inputs_is_skipped(self):
        target = _target(inputs={"svc": "mem-1"})
        storage = _storage(target)

        assert select_recordable_views(storage, "b1") == []
        assert _emit(storage, target) == []

    def test_mixed_records_only_real_artifacts(self):
        target = _target(
            inputs={"data": "real-in", "svc": "mem-1"},
            outputs={"model": ["real-out"], "url": ["mem-2"]},
        )
        storage = _storage(target)

        [(selected, _)] = select_recordable_views(storage, "b1")
        events = _emit(storage, target)

        assert expected_run_count(storage, selected) == 1 == len(events)
        assert _uris(events[0]["inputs"]) == ["hf://datasets/org/data"]
        assert _uris(events[0]["outputs"]) == ["s3://bucket/model"]

    def test_unresolvable_uuid_is_kept(self):
        """An uuid missing from the registry still counts, as before #561."""
        target = _target(outputs={"model": ["gone"]})
        storage = _storage(target)

        [(selected, _)] = select_recordable_views(storage, "b1")

        assert expected_run_count(storage, selected) == 1 == len(_emit(storage, target))

    def test_filter_does_not_mutate_the_stored_row(self):
        target = _target(inputs={"data": "real-in"}, outputs={"svc": ["mem-1"]})

        lineage_view(_storage(target), target)

        assert target.output_artifacts == {"svc": ["mem-1"]}

    def test_count_on_a_raw_target_excludes_mem(self):
        """expected_run_count filters itself; a raw target cannot over-count."""
        target = _target(
            inputs={"data": "real-in"}, outputs={"svc": ["mem-1", "mem-2"]}
        )

        assert expected_run_count(_storage(target), target) == 1


class TestOutputIndex:
    """Dropping a mem:// sibling must not renumber or un-index a real output."""

    @staticmethod
    def _refs(events) -> list[str]:
        return [e["outputs"][0]["facets"]["target_artifact_reference"] for e in events]

    def test_real_output_keeps_its_original_index(self):
        target = _target(outputs={"out": ["mem-1", "real-out"]})

        [ref] = self._refs(_emit(_storage(target), target))

        assert ref.endswith("out[1]")

    def test_indices_are_not_renumbered(self):
        target = _target(outputs={"out": ["real-out", "mem-1", "real-out-2"]})

        refs = self._refs(_emit(_storage(target), target))

        assert [r[-6:] for r in refs] == ["out[0]", "out[2]"]

    def test_single_output_stays_unindexed(self):
        target = _target(outputs={"out": ["real-out"]})

        [ref] = self._refs(_emit(_storage(target), target))

        assert not ref.endswith("]")


class TestRegistryReads:
    def test_reconcile_reads_each_artifact_once(self):
        """The real reconcile path shares one resolved view end to end."""
        target = _target(inputs={"data": "real-in"}, outputs={"out": ["real-out"]})
        storage = _storage(target)
        storage.build_storage.get_by_uuid.return_value = _build()
        storage.target_storage.get_by_where.side_effect = (
            lambda where, query_control=None: (
                [target]
                if query_control is None or query_control.pagination.index == 0
                else []
            )
        )
        store = WandBLineageStore.__new__(WandBLineageStore)
        store._service = MagicMock()
        store.filter_unrecorded = lambda c, expected, on_query_error: set(c)

        result = reconcile_build(store, storage, "b1")

        assert result.newly_recorded == 1
        store._service.emit_event.assert_called_once()
        assert storage.artifact_registry.get_by_uuid.call_count == 2


class TestSelectionOrder:
    def test_mem_only_retry_does_not_shadow_an_older_real_run(self):
        """A newer mem-only SUCCESS run is skipped before the dedup, like an
        artifact-less one, so the older run's real lineage is still recorded."""
        old = _target(inputs={"data": "real-in"}, outputs={"out": ["real-out"]})
        new = StoredTargetRun(
            uuid="t1-retry",
            build_id="b1",
            name="t1",
            environment_uri="env://test",
            status=Status.SUCCESS,
            finished_at=_FINISHED + timedelta(minutes=5),
            output_artifacts={"svc": ["mem-1"]},
        )
        storage = _storage(old)
        storage.target_storage.get_by_where.side_effect = (
            lambda where, query_control=None: (
                [new, old] if query_control.pagination.index == 0 else []
            )
        )

        assert [t.uuid for t, _ in select_recordable_views(storage, "b1")] == ["t1"]


class TestRegistryErrors:
    def test_a_failed_registry_read_is_contained_to_its_target(self):
        target = _target(inputs={"data": "real-in"}, outputs={"out": ["real-out"]})
        storage = _storage(target)
        storage.artifact_registry.get_by_uuid.side_effect = RuntimeError("locked")
        store = MagicMock()
        errors = []

        result = reconcile_build(
            store, storage, "b1", on_error=lambda b, t, e: errors.append((b, t))
        )

        assert not result.all_confirmed
        assert errors == [("b1", "t1")]
        store.add_jobstats_for_build_target.assert_not_called()


class TestIndexer:
    def test_mem_only_target_is_not_built(self):
        """The indexer's gate agrees: no events are built for a mem-only target."""
        target = _target(outputs={"svc": ["mem-1"]})
        sink = MagicMock()
        indexer = TargetLineageIndexer.__new__(TargetLineageIndexer)
        indexer._sink = sink

        assert indexer._index(_storage(target), target) is False
        sink.create_jobstats_for_target.assert_not_called()


class TestAlreadyRecorded:
    def test_target_with_runs_from_before_the_fix_writes_nothing(self):
        """N runs recorded pre-fix still satisfy the lower count: no rewrite."""
        target = _target(
            inputs={"data": "real-in"}, outputs={"svc": ["mem-1", "mem-2"]}
        )
        storage = _storage(target)
        store = MagicMock()
        # The sink holds 2 runs; filter_unrecorded's >= check passes for 1.
        store.filter_unrecorded.side_effect = lambda c, expected, on_query_error: {
            t for t in c if 2 < expected[t]
        }

        result = reconcile_build(store, storage, "b1")

        assert result.all_confirmed
        store.filter_unrecorded.assert_called_once()
        assert store.filter_unrecorded.call_args.args[1] == {"t1": 1}
        store.add_jobstats_for_build_target.assert_not_called()


class TestWandBRegistration:
    @pytest.mark.parametrize(
        "resource",
        [
            {"uri": "mem://x", "name": "x"},
            {"name": "x", "facets": {"artifact_uri": "mem://x"}},
        ],
    )
    def test_mem_resource_is_not_registered(self, resource):
        service = WandBLineageService.__new__(WandBLineageService)
        run = MagicMock()
        event = {"inputs": [resource], "outputs": [resource]}

        with patch("gbserver.lineage.wandb_service.wandb") as wandb:
            service._register_artifacts(run, event)

        wandb.Artifact.assert_not_called()
        run.log_artifact.assert_not_called()
        run.use_artifact.assert_not_called()
