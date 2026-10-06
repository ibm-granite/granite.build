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


def _artifact(uri, type="dataset"):
    return SimpleNamespace(
        metadata={"uri": uri, "namespace": "ns", "name": uri}, name=uri, type=type
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


def _checkpoint(storage, provider="wandb"):
    """Get the checkpoint for a provider. Defaults to wandb for W&B tests."""
    key = idx._checkpoint_key_for_provider(provider)
    return storage.kv_pair_storage.values.get(key)


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


def test_run_to_job_drops_wandb_system_artifacts():
    # wandb attaches e.g. run-<id>-history once someone opens the run's charts;
    # it must not become a dataset in the index. Filtering is by type, so an
    # artifact whose *name* starts with "run-" is still kept.
    run = _run("r1", "2026-01-01T00:00:00")
    run.logged_artifacts = lambda: [
        _artifact("out"),
        _artifact("run-r1-history", type="wandb-history"),
        _artifact("run-r1-events", type="wandb-events"),
        _artifact("run-r1-samples", type="run_table"),
        _artifact("run-model", type="model"),
    ]
    job = idx.wandb_run_to_job(run)
    assert [t["uri"] for t in job["targets"]] == ["out", "run-model"]


def test_run_without_job_id_is_not_lineage():
    assert idx.wandb_run_to_job(_run("r1", "t", job_id=None)) is None


# -- W&B scanning -----------------------------------------------------------

D1, D2, D3 = "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"


def test_scan_checkpoints_the_timestamp_and_the_ids_done_on_it():
    storage = _storage()
    ix, api, sink = _indexer([_run("r1", D1, job_id="j1"), _run("r2", D2, job_id="j2")])

    assert ix.scan_once(storage) == 2
    assert sink.write_job.call_count == 2
    assert _checkpoint(storage) == {
        "timestamp": D2,
        "item_ids": ["r2"],
        "version": 1,
    }

    api.runs.return_value = []
    ix.scan_once(storage)
    assert api.runs.call_args.kwargs["filters"] == {"createdAt": {"$gte": D2}}


def test_first_scan_reads_from_the_beginning():
    ix, api, _ = _indexer([])
    ix.scan_once(_storage())
    assert api.runs.call_args.kwargs["filters"] == {}


def test_running_run_is_indexed_and_does_not_stop_scan():
    storage = _storage()
    ix, _, _ = _indexer(
        [
            _run("r1", D1, job_id="j1"),
            _run("r2", D2, job_id="j2", state="running"),
            _run("r3", D3, job_id="j3"),
        ]
    )
    assert ix.scan_once(storage) == 3
    assert _checkpoint(storage)["timestamp"] == D3


def test_runs_on_the_same_instant_are_all_indexed_across_a_crash():
    """Two runs on one second: the first is done, the next scan picks up the second."""
    storage = _storage()
    ix, api, sink = _indexer([_run("r1", D1, job_id="j1"), _run("r2", D1, job_id="j2")])
    sink.write_job.side_effect = lambda job, **kw: (
        (_ for _ in ()).throw(RuntimeError("crash"))
        if job["run"]["runId"] == "r2"
        else None
    )
    assert ix.scan_once(storage) == 1
    assert _checkpoint(storage)["item_ids"] == ["r1"]

    sink.write_job.reset_mock(side_effect=True)
    assert ix.scan_once(storage) == 1
    assert [c.args[0]["run"]["runId"] for c in sink.write_job.call_args_list] == ["r2"]
    assert _checkpoint(storage) == {
        "timestamp": D1,
        "item_ids": ["r1", "r2"],
        "version": 1,
    }


def test_caught_up_scan_neither_reindexes_nor_writes_the_checkpoint():
    storage = _storage()
    ix, _, sink = _indexer([_run("r1", D1, job_id="j1"), _run("r2", D2, job_id="j2")])
    ix.scan_once(storage)
    sink.write_job.reset_mock()
    with patch.object(storage.kv_pair_storage, "set_value") as set_value:
        assert ix.scan_once(storage) == 0
    sink.write_job.assert_not_called()
    set_value.assert_not_called()


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


def test_wandb_seed_by_timestamp_is_spelled_like_created_at():
    """A seed reads like a checkpoint a scan wrote: UTC with a trailing Z."""
    ix, api, _ = _indexer([])
    storage = _storage()
    assert ix.seed_if_absent(storage, "2026-01-01T03:00:00+03:00") is True
    assert _checkpoint(storage)["timestamp"] == D1
    api.runs.assert_not_called()


def test_wandb_seeded_timestamp_is_the_scan_filter():
    ix, api, _ = _indexer([])
    storage = _storage()
    ix.seed_if_absent(storage, D2)
    ix.scan_once(storage)
    assert api.runs.call_args.kwargs["filters"] == {"createdAt": {"$gte": D2}}


def test_seed_never_overwrites_and_all_writes_nothing():
    ix, api, _ = _indexer([])
    storage = _storage()
    assert ix.seed_if_absent(storage, "all") is False
    assert storage.kv_pair_storage.values == {}
    key = idx._checkpoint_key_for_provider("wandb")
    storage.kv_pair_storage.set_value(key, {"timestamp": D1})
    assert ix.seed_if_absent(storage, D2) is False
    assert _checkpoint(storage)["timestamp"] == D1
    api.runs.assert_not_called()


def test_seed_that_is_not_a_timestamp_raises():
    """A job_id (the old flag's value) is refused, not silently read as a date."""
    ix, _, _ = _indexer([])
    storage = _storage()
    with pytest.raises(idx.LineageSeedError):
        ix.seed_if_absent(storage, "3cd41742-fb87-4ead-9fce-84dba81f3edf")
    assert storage.kv_pair_storage.values == {}


def test_wandb_seed_from_latest_with_no_runs_raises():
    ix, _, _ = _indexer([])
    with pytest.raises(idx.LineageSeedError):
        ix.seed_if_absent(_storage(), "from-latest")


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
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.scan_once(storage) == 2
    assert [
        c.kwargs["target_id"] for c in sink.add_jobstats_for_build_target.call_args_list
    ] == ["t1", "t2"]
    assert _checkpoint(storage, provider="none")["timestamp"] == _ts(2)


def test_targets_behind_the_checkpoint_are_not_read():
    storage = _storage()
    mark = {"timestamp": _ts(60), "item_ids": ["at"], "version": 1}
    key = idx._checkpoint_key_for_provider("none")
    storage.kv_pair_storage.set_value(key, dict(mark))
    ix, sink, page = _target_indexer(
        [_target("old", 58), _target("at", 60), _target("tie", 60), _target("new", 61)]
    )
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.scan_once(storage) == 2
    ids = [
        c.kwargs["target_id"] for c in sink.add_jobstats_for_build_target.call_args_list
    ]
    # "at" is listed as done on the mark; "tie" shares its instant and is not.
    assert ids == ["tie", "new"]
    assert _checkpoint(storage, provider="none")["timestamp"] == _ts(61)
    assert _checkpoint(storage, provider="none")["item_ids"] == ["new"]


def test_target_scan_reads_past_a_page_that_reaches_the_checkpoint():
    """A newer row on a later page (SQLite text order) must still be read."""
    storage = _storage()
    key = idx._checkpoint_key_for_provider("none")
    storage.kv_pair_storage.set_value(
        key, {"timestamp": _ts(60), "item_ids": []}
    )
    ix = idx.TargetLineageIndexer(sink=MagicMock())
    ix._sink.row_storage.has_rows_for_job.return_value = False
    first = [_target(f"old{i}", 0) for i in range(idx._SCAN_PAGE_SIZE)]
    pages = [first, [_target("missorted", 61)]]
    with patch.object(
        idx,
        "_successful_targets_page",
        side_effect=lambda storage, i: pages[i] if i < len(pages) else [],
    ):
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.scan_once(storage) == 1
    assert _checkpoint(storage, provider="none")["item_ids"] == ["missorted"]


def test_already_indexed_and_artifactless_targets_are_not_rewritten():
    storage = _storage()
    ix, sink, page = _target_indexer(
        [_target("done", 1), _target("empty", 2, artifacts=False), _target("t3", 3)],
        indexed={"done"},
    )
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.scan_once(storage) == 1
    sink.add_jobstats_for_build_target.assert_called_once()
    assert _checkpoint(storage, provider="none")["timestamp"] == _ts(3)


# -- gb_targets seeding -----------------------------------------------------


def test_target_seed_by_timestamp_keeps_its_offset():
    """gb_targets' form: the aware isoformat, never rewritten to UTC."""
    storage = _storage()
    ix, _, _ = _target_indexer([])
    with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
        assert ix.seed_if_absent(storage, "2026-01-01T02:05:00+02:00") is True
    assert _checkpoint(storage, provider="none")["timestamp"] == "2026-01-01T02:05:00+02:00"


def test_target_seed_indexes_from_that_instant_inclusive():
    storage = _storage()
    ix, sink, page = _target_indexer([_target("t1", 1), _target("t2", 5)])
    with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
        ix.seed_if_absent(storage, _ts(5))
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            ix.scan_once(storage)
    indexed = [
        c.kwargs["target_id"] for c in sink.add_jobstats_for_build_target.call_args_list
    ]
    assert "t2" in indexed


def test_target_seed_from_latest_takes_newest_finished():
    storage = _storage()
    ix, _, page = _target_indexer([_target("t1", 1), _target("t2", 2)])
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.seed_if_absent(storage, "from-latest") is True
    assert _checkpoint(storage, provider="none")["timestamp"] == _ts(2)


def test_target_seed_naive_timestamp_is_read_as_local():
    storage = _storage()
    ix, _, _ = _target_indexer([])
    with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
        ix.seed_if_absent(storage, "2026-01-01T00:00:00")
    stored = datetime.fromisoformat(_checkpoint(storage, provider="none")["timestamp"])
    assert stored.tzinfo is not None
    assert stored.replace(tzinfo=None) == datetime(2026, 1, 1)


# -- DBLineageStore.write_job ----------------------------------------------


def test_db_store_write_job_delegates():
    from gbserver.lineage.db_jobstats import DBLineageStore

    store = DBLineageStore(storage=MagicMock())
    with patch.object(store, "_write_job") as inner:
        store.write_job({"x": 1}, build_id="b", target_run_uuid="t")
    inner.assert_called_once_with(
        {"x": 1}, build_id="b", target_run_uuid="t", extra_tags=None
    )
