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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage import indexer as idx
from gbserver.lineage.lineage_watcher import LineageWatcher


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


@pytest.fixture(autouse=True)
def _project(monkeypatch):
    monkeypatch.setattr(
        idx.WandBLineageIndexer, "_project_path", staticmethod(lambda: "e/p")
    )


# -- source resolution ------------------------------------------------------


def test_resolve_source_defaults_by_mode(monkeypatch):
    monkeypatch.delenv("GBSERVER_LINEAGE_INDEXER_SOURCE", raising=False)
    with patch("gbcommon.types.gbenvconfig.is_standalone", return_value=True):
        assert idx.resolve_indexer_source() == idx.INDEXER_SOURCE_ADMIN_DB
    with patch("gbcommon.types.gbenvconfig.is_standalone", return_value=False):
        assert idx.resolve_indexer_source() == idx.INDEXER_SOURCE_LINEAGE_STORE


def test_resolve_source_rejects_unknown(monkeypatch):
    monkeypatch.setenv("GBSERVER_LINEAGE_INDEXER_SOURCE", "bogus")
    with pytest.raises(idx.UnknownIndexerSource):
        idx.resolve_indexer_source()


def test_override_wins_over_env(monkeypatch):
    monkeypatch.setenv("GBSERVER_LINEAGE_INDEXER_SOURCE", "lineage_store")
    assert idx.resolve_indexer_source("admin_db") == "admin_db"


# -- create_indexer ---------------------------------------------------------


def test_admin_db_is_watcher_on_index_with_own_keys():
    sink = MagicMock()
    w = idx.create_indexer(idx.INDEXER_SOURCE_ADMIN_DB, sink=sink)
    assert isinstance(w, LineageWatcher)
    assert w._configured_store is sink
    assert w.checkpoint_key == idx.INDEXER_CHECKPOINT_KEY
    assert w.dropped_key == idx.INDEXER_DROPPED_KEY


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


# -- scanning ---------------------------------------------------------------


def test_scan_indexes_and_checkpoints_incrementally():
    storage = _storage()
    ix, api, sink = _indexer([_run("r1", "2026-01-01"), _run("r2", "2026-01-02")])

    assert ix.scan_once(storage) == 2
    assert sink.write_job.call_count == 2
    _, kwargs = sink.write_job.call_args
    assert kwargs == {"build_id": "b1", "target_run_uuid": "t1"}
    assert (
        storage.kv_pair_storage.values[idx.INDEXER_WANDB_CHECKPOINT_KEY]["run_id"]
        == "r2"
    )

    api.runs.return_value = []
    ix.scan_once(storage)
    assert api.runs.call_args.kwargs["filters"] == {"createdAt": {"$gte": "2026-01-02"}}


def test_first_scan_reads_from_the_beginning():
    ix, api, _ = _indexer([])
    ix.scan_once(_storage())
    assert api.runs.call_args.kwargs["filters"] == {}


def test_running_run_stops_scan_without_advancing():
    storage = _storage()
    ix, _, sink = _indexer(
        [_run("r1", "d1"), _run("r2", "d2", state="running"), _run("r3", "d3")]
    )
    assert ix.scan_once(storage) == 1
    assert (
        storage.kv_pair_storage.values[idx.INDEXER_WANDB_CHECKPOINT_KEY]["run_id"]
        == "r1"
    )


def test_failing_run_retries_then_is_skipped():
    storage = _storage()
    ix, _, sink = _indexer([_run("r1", "d1"), _run("r2", "d2")])
    sink.write_job.side_effect = lambda job, **kw: (
        (_ for _ in ()).throw(RuntimeError("boom"))
        if job["run"]["runId"] == "r1"
        else None
    )
    for _ in range(idx._MAX_RUN_ATTEMPTS - 1):
        assert ix.scan_once(storage) == 0
        assert idx.INDEXER_WANDB_CHECKPOINT_KEY not in storage.kv_pair_storage.values
    assert ix.scan_once(storage) == 1
    assert (
        storage.kv_pair_storage.values[idx.INDEXER_WANDB_CHECKPOINT_KEY]["run_id"]
        == "r2"
    )


def test_non_lineage_run_advances_checkpoint_without_writing():
    storage = _storage()
    ix, _, sink = _indexer([_run("r1", "d1", job_id=None)])
    assert ix.scan_once(storage) == 0
    sink.write_job.assert_not_called()
    assert (
        storage.kv_pair_storage.values[idx.INDEXER_WANDB_CHECKPOINT_KEY]["run_id"]
        == "r1"
    )


# -- DBLineageStore.write_job ----------------------------------------------


def test_db_store_write_job_delegates():
    from gbserver.lineage.db_jobstats import DBLineageStore

    store = DBLineageStore(storage=MagicMock())
    with patch.object(store, "_write_job") as inner:
        store.write_job({"x": 1}, build_id="b", target_run_uuid="t")
    inner.assert_called_once_with(
        {"x": 1}, build_id="b", target_run_uuid="t", extra_tags=None
    )
