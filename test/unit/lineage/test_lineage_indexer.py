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
from gbserver.storage.stored_lineage_row import JobStore
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


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("none", idx.INDEXER_SOURCE_ADMIN_DB),
        ("db", idx.INDEXER_SOURCE_LINEAGE_JOB),
        ("wandb", idx.INDEXER_SOURCE_LINEAGE_STORE),
    ],
)
@pytest.mark.parametrize("standalone", [True, False])
def test_source_follows_provider_in_every_mode(standalone, provider, expected):
    with (
        patch("gbcommon.types.gbenvconfig.is_standalone", return_value=standalone),
        patch.object(idx, "_resolve_lineage_provider", return_value=provider),
    ):
        assert idx.resolve_indexer_source() == expected


# -- create_indexer ---------------------------------------------------------


@pytest.mark.parametrize(
    "source,cls",
    [
        (idx.INDEXER_SOURCE_ADMIN_DB, idx.TargetLineageIndexer),
        (idx.INDEXER_SOURCE_LINEAGE_JOB, idx.LineageJobIndexer),
        (idx.INDEXER_SOURCE_LINEAGE_STORE, idx.WandBLineageIndexer),
    ],
)
def test_create_indexer_per_source(source, cls):
    sink = MagicMock()
    ix = idx.create_indexer(source, sink=sink)
    assert isinstance(ix, cls)
    assert ix._sink is sink


# -- lineage_job source -----------------------------------------------------


def _job_record(job_id, recorded_at, entry=True):
    from gbserver.storage.stored_lineage_job import StoredLineageJob

    attributes = {}
    if entry:
        attributes["entry"] = {
            "event": {
                "job": {"namespace": "space/build", "name": "t"},
                "job_details": {"job_id": job_id},
                "sources": [{"uri": f"s3://bucket/in-{job_id}"}],
                "targets": [{"uri": f"s3://bucket/out-{job_id}"}],
            },
            "build_id": "b1",
            "target_run_uuid": job_id,
        }
    return StoredLineageJob(
        job_id=job_id, recorded_at=recorded_at, attributes=attributes
    )


def _admin_with_jobs(jobs):
    kv = {}
    storage = MagicMock()
    storage.lineage_job_storage.get_paged.side_effect = lambda **_: iter([list(jobs)])
    storage.kv_pair_storage.get_value.side_effect = kv.get
    storage.kv_pair_storage.set_value.side_effect = kv.__setitem__
    return storage, kv


def test_lineage_job_indexer_reads_in_order_and_checkpoints():
    jobs = [
        _job_record("j2", "2026-01-02T00:00:00.000000+00:00"),
        _job_record("j1", "2026-01-01T00:00:00.000000+00:00"),
    ]
    storage, kv = _admin_with_jobs(jobs)
    # The rows collaborator indexes, not the sink: the index is the indexer's.
    rows = MagicMock()
    rows.index_job_records.side_effect = len
    ix = idx.LineageJobIndexer(sink=MagicMock(), rows=rows)
    with patch.object(idx, "_resolve_lineage_provider", return_value="db"):
        assert ix.scan_once(storage) == 2
        (batch,) = [c.args[0] for c in rows.index_job_records.call_args_list]
        assert [j.job_id for j in batch] == ["j1", "j2"]
        cp = kv["lineage_index_checkpoint:db"]
        assert cp["timestamp"] == "2026-01-02T00:00:00.000000+00:00"
        # Nothing new: nothing re-read.
        assert ix.scan_once(storage) == 0


def test_db_sink_record_only_then_index_dedups():
    """The store records; only the indexer writes rows. Both halves, in order."""
    from gbserver.lineage.db_jobstats import DBLineageStore
    from gbserver.lineage.row_indexing import LineageRowIndexer

    rows, jobs = MagicMock(), MagicMock()
    jobs.get_job.return_value = None
    store = DBLineageStore(job_storage=jobs)
    event = _job_record("j1", "x").attributes["entry"]["event"]
    with patch("gbserver.lineage.row_indexing.upsert_row") as upsert_row:
        store.write_job(event, build_id="b1", target_run_uuid="j1")
        # The store cannot write a row -- it has no row storage at all now.
        upsert_row.assert_not_called()
        record = jobs.add.call_args.args[0]
        assert record.attributes["entry"]["events"] == [event]

        reader = LineageRowIndexer(storage=rows)
        assert reader.index_job_record(record) is True
        assert reader.index_job_record(record) is True
        # Every pass goes through upsert_row, which merges into an existing row.
        assert upsert_row.call_count == 2
        jobs.add.assert_called_once()


def test_index_job_records_bulk_adds_new_and_merges_existing():
    from gbserver.lineage.row_indexing import LineageRowIndexer
    from gbserver.storage.stored_lineage_row import StoredLineageRow

    def row(job_id, inp, out):
        return StoredLineageRow(job_id=job_id, input=inp, output=out).model_dump(
            mode="json", exclude={"uuid"}
        )

    new = _job_record("new", "x", entry=False)
    new.attributes["entry"] = {
        "rows": [row("new", "a", "b"), row("new", "a", "b"), row("new", "", "b")]
    }
    old = _job_record("old", "x", entry=False)
    old.attributes["entry"] = {"rows": [row("old", "c", "d")]}

    rows = MagicMock()
    rows.get_rows_by_jobs.return_value = [MagicMock(job_id="old")]
    store = LineageRowIndexer(storage=rows)
    with patch("gbserver.lineage.row_indexing.upsert_row") as upsert_row:
        assert store.index_job_records([new, old]) == 2
    # One bulk add: the duplicate collapsed, the superseded terminal dropped.
    (added,) = rows.add.call_args.args
    assert [(r.input, r.output) for r in added] == [("a", "b")]
    # The job that already had rows is merged row by row (dedup).
    assert [c.args[1].job_id for c in upsert_row.call_args_list] == ["old"]


def test_index_job_record_without_entry_is_skipped():
    from gbserver.lineage.row_indexing import LineageRowIndexer

    store = LineageRowIndexer(storage=MagicMock())
    assert store.index_job_record(_job_record("j1", "x", entry=False)) is False


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
    # The builder is pure; one event per target is enough to drive the loop.
    sink.create_jobstats_for_target.side_effect = lambda storage, target, build=None: (
        [{"job_id": target.uuid}],
        {},
    )
    # "Already indexed" is a question about the index, which the rows collaborator
    # owns. Under this source gb_targets IS the job store, so there is no job
    # record to ask the sink about.
    rows = MagicMock()
    rows.has_rows_for_job.side_effect = lambda job_id: job_id in indexed
    rows.index_job_entry.return_value = True
    ix = idx.TargetLineageIndexer(sink=sink, rows=rows)
    ordered = sorted(targets, key=lambda t: t.finished_at, reverse=True)
    page = patch.object(
        idx,
        "_successful_targets_page",
        side_effect=lambda storage, i: ordered if i == 0 else [],
    )
    return ix, rows, page


def test_targets_are_indexed_oldest_first_across_builds():
    storage = _storage()
    ix, rows, page = _target_indexer(
        [_target("t2", 2, build_id="b2"), _target("t1", 1, build_id="b1")]
    )
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.scan_once(storage) == 2
    assert [
        c.kwargs["target_run_uuid"] for c in rows.index_job_entry.call_args_list
    ] == ["t1", "t2"]
    assert _checkpoint(storage, provider="none")["timestamp"] == _ts(2)


def test_targets_behind_the_checkpoint_are_not_read():
    storage = _storage()
    mark = {"timestamp": _ts(60), "item_ids": ["at"], "version": 1}
    key = idx._checkpoint_key_for_provider("none")
    storage.kv_pair_storage.set_value(key, dict(mark))
    ix, rows, page = _target_indexer(
        [_target("old", 58), _target("at", 60), _target("tie", 60), _target("new", 61)]
    )
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.scan_once(storage) == 2
    ids = [c.kwargs["target_run_uuid"] for c in rows.index_job_entry.call_args_list]
    # "at" is listed as done on the mark; "tie" shares its instant and is not.
    assert ids == ["tie", "new"]
    assert _checkpoint(storage, provider="none")["timestamp"] == _ts(61)
    assert _checkpoint(storage, provider="none")["item_ids"] == ["new"]


def test_target_scan_reads_past_a_page_that_reaches_the_checkpoint():
    """A newer row on a later page (SQLite text order) must still be read."""
    storage = _storage()
    key = idx._checkpoint_key_for_provider("none")
    storage.kv_pair_storage.set_value(key, {"timestamp": _ts(60), "item_ids": []})
    ix, rows, _ = _target_indexer([])
    rows.has_rows_for_job.return_value = False
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
    ix, rows, page = _target_indexer(
        [_target("done", 1), _target("empty", 2, artifacts=False), _target("t3", 3)],
        indexed={"done"},
    )
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            assert ix.scan_once(storage) == 1
    rows.index_job_entry.assert_called_once()
    assert _checkpoint(storage, provider="none")["timestamp"] == _ts(3)


# -- gb_targets seeding -----------------------------------------------------


def test_target_seed_by_timestamp_keeps_its_offset():
    """gb_targets' form: the aware isoformat, never rewritten to UTC."""
    storage = _storage()
    ix, _, _ = _target_indexer([])
    with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
        assert ix.seed_if_absent(storage, "2026-01-01T02:05:00+02:00") is True
    assert (
        _checkpoint(storage, provider="none")["timestamp"]
        == "2026-01-01T02:05:00+02:00"
    )


def test_target_seed_indexes_from_that_instant_inclusive():
    storage = _storage()
    ix, rows, page = _target_indexer([_target("t1", 1), _target("t2", 5)])
    with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
        ix.seed_if_absent(storage, _ts(5))
    with page:
        with patch.object(idx, "_resolve_lineage_provider", return_value="none"):
            ix.scan_once(storage)
    indexed = [c.kwargs["target_run_uuid"] for c in rows.index_job_entry.call_args_list]
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

    store = DBLineageStore(job_storage=MagicMock())
    with patch.object(store, "_write_job") as inner:
        store.write_job({"x": 1}, build_id="b", target_run_uuid="t")
    inner.assert_called_once_with(
        {"x": 1},
        build_id="b",
        target_run_uuid="t",
        extra_tags=None,
        job_store=JobStore.OTHER,
    )


def test_each_source_stamps_its_job_store():
    """job_store names the store to follow for the job's full content.

    Stamped by the indexer, which writes the rows -- WANDB when it read the run
    from W&B, LINEAGE_JOB when it derived the rows from a job record.
    """
    from gbserver.lineage.row_indexing import LineageRowIndexer

    rows = MagicMock()
    rows.get_rows_by_job.return_value = []
    rows.get_rows_by_jobs.return_value = []
    indexer = LineageRowIndexer(storage=rows)

    event = _job_record("j1", "x").attributes["entry"]["event"]
    indexer.index_job_entry(
        event, build_id="b", target_run_uuid="t", job_store=JobStore.WANDB
    )
    assert {r.args[0].job_store for r in rows.add.call_args_list} == {JobStore.WANDB}

    rows.add.reset_mock()
    indexer.index_job_record(_job_record("j2", "x"))
    assert {r.args[0].job_store for r in rows.add.call_args_list} == {
        JobStore.LINEAGE_JOB
    }

    rows.add.reset_mock()
    prebuilt = _job_record("j3", "x", entry=False)
    prebuilt.attributes["entry"] = {
        "rows": [{"job_id": "j3", "input": "a", "output": "b", "job_store": "other"}]
    }
    indexer.index_job_records([prebuilt])
    (added,) = rows.add.call_args.args
    assert [r.job_store for r in added] == [JobStore.LINEAGE_JOB]
