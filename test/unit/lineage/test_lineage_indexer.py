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

"""Unit tests for the lineage indexer (gbserver.lineage.indexer)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage import indexer as idx
from gbserver.storage.stored_target_run import StoredTargetRun


class _KV:
    def __init__(self):
        self.values = {}

    def get_value(self, key):
        return self.values.get(key)

    def set_value(self, key, value):
        self.values[key] = value


def _storage():
    return SimpleNamespace(kv_pair_storage=_KV())


def _artifact(uri):
    return SimpleNamespace(
        metadata={"uri": uri, "namespace": "ns", "name": uri}, name=uri
    )


def _run(run_id, created_at, state="finished", job_id="t1", outputs=("out",)):
    config = {"job_name": "train", "job_namespace": "space/build"}
    if job_id:
        config.update({"job_id": job_id, "release_id": "b1", "owner": "u"})
    return SimpleNamespace(
        id=run_id,
        name="train",
        created_at=created_at,
        state=state,
        config=config,
        tags=["build_id=b1", f"target_id={job_id}"],
        used_artifacts=lambda: [_artifact("in")],
        logged_artifacts=lambda: [_artifact(o) for o in outputs],
    )


def _indexer(runs):
    api = MagicMock()
    api.runs.return_value = runs
    sink = MagicMock()
    ix = idx.WandBLineageIndexer(sink=sink, api=api)
    return ix, api, sink


def _checkpoint(storage):
    return storage.kv_pair_storage.values.get(idx.INDEXER_CHECKPOINT_KEY)


@pytest.fixture(autouse=True)
def _project(monkeypatch):
    monkeypatch.setattr(
        idx.WandBLineageIndexer, "_project_path", staticmethod(lambda: "e/p")
    )


# -- source resolution ------------------------------------------------------


def test_source_follows_startup_mode():
    with patch("gbcommon.types.gbenvconfig.is_standalone", return_value=True):
        assert idx.resolve_indexer_source() == idx.INDEXER_SOURCE_ADMIN_DB
    with patch("gbcommon.types.gbenvconfig.is_standalone", return_value=False):
        assert idx.resolve_indexer_source() == idx.INDEXER_SOURCE_LINEAGE_STORE


# -- create_indexer ---------------------------------------------------------


def test_admin_db_is_target_indexer_not_the_watcher():
    sink = MagicMock()
    ix = idx.create_indexer(idx.INDEXER_SOURCE_ADMIN_DB, sink=sink)
    assert isinstance(ix, idx.TargetLineageIndexer)
    assert ix._sink is sink


@pytest.mark.parametrize("provider", ["db", "none"])
def test_lineage_store_that_is_index_or_none_is_noop(provider):
    with patch.object(idx, "_resolve_lineage_provider", return_value=provider):
        assert (
            idx.create_indexer(idx.INDEXER_SOURCE_LINEAGE_STORE, sink=MagicMock())
            is None
        )


def test_lineage_store_wandb_builds_wandb_indexer():
    with patch.object(idx, "_resolve_lineage_provider", return_value="wandb"):
        ix = idx.create_indexer(idx.INDEXER_SOURCE_LINEAGE_STORE, sink=MagicMock())
    assert isinstance(ix, idx.WandBLineageIndexer)


# -- run conversion ---------------------------------------------------------


def test_run_to_job_uses_config_namespace_not_entity_project():
    job = idx.wandb_run_to_job(_run("r1", "2026-01-01T00:00:00"))
    assert job["job"]["namespace"] == "space/build"
    assert job["job_details"]["job_id"] == "t1"
    assert [s["uri"] for s in job["sources"]] == ["in"]
    assert [t["uri"] for t in job["targets"]] == ["out"]


def test_run_without_job_id_is_not_lineage():
    assert idx.wandb_run_to_job(_run("r1", "t", job_id=None)) is None


# -- W&B scanning -----------------------------------------------------------

D1, D2, D3 = "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"


def test_scan_indexes_and_checkpoints_only_a_timestamp():
    storage = _storage()
    ix, api, sink = _indexer([_run("r1", D1, job_id="j1"), _run("r2", D2, job_id="j2")])

    assert ix.scan_once(storage) == 2
    assert sink.write_job.call_count == 2
    assert _checkpoint(storage) == {
        "timestamp": D2,
        "version": idx.INDEXER_CHECKPOINT_VERSION,
    }

    api.runs.return_value = []
    ix.scan_once(storage)
    assert api.runs.call_args.kwargs["filters"] == {"createdAt": {"$gte": D2}}


def test_first_scan_reads_from_the_beginning():
    ix, api, _ = _indexer([])
    ix.scan_once(_storage())
    assert api.runs.call_args.kwargs["filters"] == {}


def test_running_run_stops_scan_without_advancing():
    storage = _storage()
    ix, _, _ = _indexer(
        [
            _run("r1", D1, job_id="j1"),
            _run("r2", D2, job_id="j2", state="running"),
            _run("r3", D3, job_id="j3"),
        ]
    )
    assert ix.scan_once(storage) == 1
    assert _checkpoint(storage)["timestamp"] == D1


def test_failing_run_retries_per_run_then_is_skipped():
    storage = _storage()
    # Same job_id on both: the attempt count must still be per run.
    ix, _, sink = _indexer([_run("r1", D1), _run("r2", D2)])
    sink.write_job.side_effect = lambda job, **kw: (
        (_ for _ in ()).throw(RuntimeError("boom"))
        if job["run"]["runId"] == "r1"
        else None
    )
    for _ in range(idx._MAX_JOB_ATTEMPTS - 1):
        assert ix.scan_once(storage) == 0
        assert _checkpoint(storage) is None
    assert ix.scan_once(storage) == 1
    assert _checkpoint(storage)["timestamp"] == D2


def test_non_lineage_run_advances_checkpoint_without_writing():
    storage = _storage()
    ix, _, sink = _indexer([_run("r1", D1, job_id=None)])
    assert ix.scan_once(storage) == 0
    sink.write_job.assert_not_called()
    assert _checkpoint(storage)["timestamp"] == D1


# -- W&B seeding ------------------------------------------------------------


def test_wandb_seed_by_job_id_checkpoints_at_its_first_run():
    ix, api, _ = _indexer([_run("r1", D1, job_id="j1")])
    storage = _storage()
    assert ix.seed_if_absent(storage, "j1") is True
    assert api.runs.call_args.kwargs["filters"] == {"config.job_id": "j1"}
    assert _checkpoint(storage)["timestamp"] == D1


def test_seed_never_overwrites_and_all_writes_nothing():
    ix, api, _ = _indexer([])
    storage = _storage()
    assert ix.seed_if_absent(storage, "all") is False
    assert storage.kv_pair_storage.values == {}
    storage.kv_pair_storage.set_value(
        idx.INDEXER_CHECKPOINT_KEY, {"timestamp": D1}
    )
    assert ix.seed_if_absent(storage, "j1") is False
    api.runs.assert_not_called()


def test_wandb_seed_unknown_job_id_raises():
    ix, _, _ = _indexer([])
    with pytest.raises(idx.LineageSeedError):
        ix.seed_if_absent(_storage(), "nope")


# -- gb_targets scanning (standalone) ---------------------------------------

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ts(minutes):
    return (T0 + timedelta(minutes=minutes)).isoformat()


def _target(uuid, minutes, build_id="b1", artifacts=True):
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env",
        finished_at=T0 + timedelta(minutes=minutes),
        input_artifacts={"in": "a-in"} if artifacts else {},
        output_artifacts={"out": ["a-out"]} if artifacts else {},
    )


def _target_indexer(targets, indexed=()):
    """Targets are served newest-finished first, as the DB page would."""
    sink = MagicMock()
    sink.row_storage.has_rows_for_job.side_effect = lambda job_id: job_id in indexed
    ix = idx.TargetLineageIndexer(sink=sink)
    ordered = sorted(targets, key=lambda t: t.finished_at, reverse=True)
    page = patch.object(
        idx,
        "_successful_targets_page",
        side_effect=lambda storage, i: ordered if i == 0 else [],
    )
    return ix, sink, page


def test_targets_are_indexed_oldest_first_across_builds():
    storage = _storage()
    ix, sink, page = _target_indexer(
        [_target("t2", 2, build_id="b2"), _target("t1", 1, build_id="b1")]
    )
    with page:
        assert ix.scan_once(storage) == 2
    assert [c.kwargs["target_id"] for c in sink.add_jobstats_for_build_target.call_args_list] == ["t1", "t2"]
    assert _checkpoint(storage)["timestamp"] == _ts(2)


def test_targets_before_checkpoint_minus_lookback_are_not_read():
    storage = _storage()
    storage.kv_pair_storage.set_value(
        idx.INDEXER_CHECKPOINT_KEY,
        {"timestamp": _ts(60)},
    )
    ix, sink, page = _target_indexer(
        [_target("old", 0), _target("late", 58), _target("new", 61)]
    )
    with page:
        assert ix.scan_once(storage) == 2
    ids = [c.kwargs["target_id"] for c in sink.add_jobstats_for_build_target.call_args_list]
    # "late" landed behind the mark but inside the lookback: still indexed,
    # and it does not rewind the checkpoint.
    assert ids == ["late", "new"]
    assert _checkpoint(storage)["timestamp"] == _ts(61)


def test_overlap_does_not_move_checkpoint_back():
    storage = _storage()
    mark = _ts(60)
    storage.kv_pair_storage.set_value(
        idx.INDEXER_CHECKPOINT_KEY, {"timestamp": mark}
    )
    ix, _, page = _target_indexer([_target("late", 58)])
    with page:
        ix.scan_once(storage)
    assert _checkpoint(storage) == {"timestamp": mark}


def test_already_indexed_and_artifactless_targets_are_not_rewritten():
    storage = _storage()
    ix, sink, page = _target_indexer(
        [_target("done", 1), _target("empty", 2, artifacts=False), _target("t3", 3)],
        indexed={"done"},
    )
    with page:
        assert ix.scan_once(storage) == 1
    sink.add_jobstats_for_build_target.assert_called_once()
    assert _checkpoint(storage)["timestamp"] == _ts(3)


# -- gb_targets seeding -----------------------------------------------------


def test_target_seed_by_job_id_anchors_at_its_finished_at():
    storage = _storage()
    storage.target_storage = MagicMock()
    storage.target_storage.get_by_uuid.return_value = _target("t1", 5)
    ix, _, _ = _target_indexer([])
    assert ix.seed_if_absent(storage, "t1") is True
    assert _checkpoint(storage)["timestamp"] == _ts(5)


def test_target_seed_from_latest_takes_newest_finished():
    storage = _storage()
    ix, _, page = _target_indexer([_target("t1", 1), _target("t2", 2)])
    with page:
        assert ix.seed_if_absent(storage, "from-latest") is True
    assert _checkpoint(storage)["timestamp"] == _ts(2)


def test_target_seed_unknown_job_id_raises():
    storage = _storage()
    storage.target_storage = MagicMock()
    storage.target_storage.get_by_uuid.return_value = None
    ix, _, _ = _target_indexer([])
    with pytest.raises(idx.LineageSeedError):
        ix.seed_if_absent(storage, "nope")


# -- DBLineageStore.write_job ----------------------------------------------


def test_db_store_write_job_delegates():
    from gbserver.lineage.db_jobstats import DBLineageStore

    store = DBLineageStore(storage=MagicMock())
    with patch.object(store, "_write_job") as inner:
        store.write_job({"x": 1}, build_id="b", target_run_uuid="t")
    inner.assert_called_once_with(
        {"x": 1}, build_id="b", target_run_uuid="t", extra_tags=None
    )
