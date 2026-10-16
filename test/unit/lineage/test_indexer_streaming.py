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

"""gb_targets source streams in batches; W&B warns about a pinned scan."""

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage import indexer as idx
from gbserver.storage.stored_target_run import StoredTargetRun

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
KEY = idx._checkpoint_key_for_provider("none")


@pytest.fixture(autouse=True)
def _standalone():
    with (
        patch.object(idx, "is_standalone", return_value=True),
        patch.object(idx, "_resolve_lineage_provider", return_value="none"),
    ):
        yield


def _no_keyset_read(storage, cutoff, after):
    raise NotImplementedError("these tests drive the listed-keys read")


@pytest.fixture(autouse=True)
def _listed_keys_read():
    """Every test here drives the listed-keys read; the keyset read has its own."""
    with patch.object(idx, "_finished_after_page", side_effect=_no_keyset_read):
        yield


class _KV:
    def __init__(self):
        self.values = {}
        self.writes = []

    def get_value(self, key):
        return self.values.get(key)

    def set_value(self, key, value):
        self.values[key] = value
        self.writes.append(dict(value))

    def compare_and_set(self, key, expected, value):
        if self.values.get(key) != expected:
            return False
        self.set_value(key, value)
        return True


def _target(uuid, minutes):
    return StoredTargetRun(
        uuid=uuid,
        build_id="b1",
        environment_uri="env",
        finished_at=T0 + timedelta(minutes=minutes),
        input_artifacts={"in": "a-in"},
        output_artifacts={"out": ["a-out"]},
    )


class _TargetStore:
    """Newest-first pages and by-uuid loads, counting what is loaded."""

    def __init__(self, targets):
        self.targets = {t.uuid: t for t in targets}
        self.loads = []

    def page(self, storage, page_index, cutoff):
        ordered = sorted(
            self.targets.values(), key=lambda t: t.finished_at, reverse=True
        )
        size = idx._SCAN_PAGE_SIZE
        return ordered[page_index * size : (page_index + 1) * size]

    def by_uuid(self, storage, uuids):
        self.loads.append(list(uuids))
        return [self.targets.get(u) for u in uuids]


def _run_scan(targets, checkpoint=None, index_side_effect=None, every=3):
    store = _TargetStore(targets)
    storage = SimpleNamespace(kv_pair_storage=_KV())
    if checkpoint:
        storage.kv_pair_storage.values[KEY] = dict(checkpoint)
    sink = MagicMock()
    sink.create_jobstats_for_target.side_effect = lambda s, t, build=None: (
        [{"job_id": t.uuid}],
        {},
    )
    rows = MagicMock()
    rows.index_job_entry.return_value = True
    if index_side_effect:
        rows.index_job_entry.side_effect = index_side_effect
    ix = idx.TargetLineageIndexer(sink=sink, rows=rows)
    ix.settle_grace_s = 0
    with (
        patch.object(idx, "_targets_page", side_effect=store.page),
        patch.object(idx, "_targets_by_uuid", side_effect=store.by_uuid),
        patch.object(idx, "_CHECKPOINT_EVERY", every),
    ):
        count = ix.scan_once(storage)
    ids = [c.kwargs["target_run_uuid"] for c in rows.index_job_entry.call_args_list]
    return count, ids, store, storage.kv_pair_storage, ix


def test_targets_are_loaded_a_batch_at_a_time_and_checkpointed_per_batch():
    targets = [_target(f"t{i}", i) for i in range(7)]
    count, ids, store, kv, _ = _run_scan(targets)
    assert count == 7
    assert ids == [f"t{i}" for i in range(7)]
    assert [len(batch) for batch in store.loads] == [3, 3, 1]
    assert [w["timestamp"] for w in kv.writes] == [
        (T0 + timedelta(minutes=m)).isoformat() for m in (2, 5, 6)
    ]


def test_a_tie_on_the_checkpoint_is_skipped_by_id_and_the_read_is_inclusive():
    targets = [_target("old", 1), _target("at", 2), _target("tie", 2), _target("n", 3)]
    mark = {"timestamp": (T0 + timedelta(minutes=2)).isoformat(), "item_ids": ["at"]}
    _, ids, _, kv, _ = _run_scan(targets, checkpoint=mark)
    assert ids == ["tie", "n"]
    assert kv.values[KEY]["item_ids"] == ["n"]


def test_unsettled_target_stops_the_scan_and_later_batches_are_never_loaded():
    now = datetime.now(timezone.utc)
    targets = [_target(f"t{i}", i) for i in range(3)]
    fresh = _target("fresh", 0)
    fresh.finished_at = now
    later = [_target(f"late{i}", 0) for i in range(4)]
    for i, t in enumerate(later):
        t.finished_at = now + timedelta(minutes=i + 1)
    store = _TargetStore(targets + [fresh] + later)
    storage = SimpleNamespace(kv_pair_storage=_KV())
    sink = MagicMock()
    sink.create_jobstats_for_target.side_effect = lambda s, t, build=None: (
        [{"job_id": t.uuid}],
        {},
    )
    rows = MagicMock()
    rows.index_job_entry.return_value = True
    ix = idx.TargetLineageIndexer(sink=sink, rows=rows)
    ix.settle_grace_s = 3600
    with (
        patch.object(idx, "_targets_page", side_effect=store.page),
        patch.object(idx, "_targets_by_uuid", side_effect=store.by_uuid),
        patch.object(idx, "_CHECKPOINT_EVERY", 3),
    ):
        assert ix.scan_once(storage) == 3
    assert len(store.loads) == 2  # the batch with "fresh", nothing after it
    assert storage.kv_pair_storage.values[KEY]["item_ids"] == ["t2"]


def test_a_failing_target_stops_the_scan_until_its_attempts_run_out():
    targets = [_target(f"t{i}", i) for i in range(5)]

    def fail_t2(*args, **kwargs):
        if kwargs["target_run_uuid"] == "t2":
            raise ValueError("bad")
        return True

    count, ids, _, kv, ix = _run_scan(targets, index_side_effect=fail_t2)
    assert count == 2
    assert ids == ["t0", "t1", "t2"]
    assert kv.values[KEY]["item_ids"] == ["t1"]
    assert ix._failed_attempts == {"t2": 1}


def test_a_transient_load_failure_keeps_the_checkpoint_where_indexing_reached():
    class OperationalError(Exception):
        pass

    targets = [_target(f"t{i}", i) for i in range(5)]
    store = _TargetStore(targets)
    calls = []

    def flaky(storage, uuids):
        calls.append(uuids)
        if len(calls) == 2:
            raise OperationalError("database is locked")
        return store.by_uuid(storage, uuids)

    storage = SimpleNamespace(kv_pair_storage=_KV())
    sink = MagicMock()
    sink.create_jobstats_for_target.side_effect = lambda s, t, build=None: (
        [{"job_id": t.uuid}],
        {},
    )
    rows = MagicMock()
    rows.index_job_entry.return_value = True
    ix = idx.TargetLineageIndexer(sink=sink, rows=rows)
    ix.settle_grace_s = 0
    with (
        patch.object(idx, "_targets_page", side_effect=store.page),
        patch.object(idx, "_targets_by_uuid", side_effect=flaky),
        patch.object(idx, "_CHECKPOINT_EVERY", 3),
    ):
        with pytest.raises(OperationalError):
            ix.scan_once(storage)
    # The first batch is checkpointed; nothing past it is skipped.
    assert storage.kv_pair_storage.values[KEY]["item_ids"] == ["t2"]
    assert ix._failed_attempts == {}


def test_a_target_deleted_after_listing_is_passed_over():
    targets = [_target(f"t{i}", i) for i in range(3)]
    store = _TargetStore(targets)
    listed = store.page(None, 0, None)
    del store.targets["t1"]
    storage = SimpleNamespace(kv_pair_storage=_KV())
    rows = MagicMock()
    rows.index_job_entry.return_value = True
    sink = MagicMock()
    sink.create_jobstats_for_target.side_effect = lambda s, t, build=None: (
        [{"job_id": t.uuid}],
        {},
    )
    ix = idx.TargetLineageIndexer(sink=sink, rows=rows)
    ix.settle_grace_s = 0
    with (
        patch.object(
            idx, "_targets_page", side_effect=lambda s, i, c: listed if i == 0 else []
        ),
        patch.object(idx, "_targets_by_uuid", side_effect=store.by_uuid),
    ):
        assert ix.scan_once(storage) == 2


def test_targets_by_uuid_maps_missing_rows_to_none():
    storage = SimpleNamespace(target_storage=MagicMock())
    t = _target("a", 1)
    storage.target_storage.get_by_uuid.return_value = [t, None]
    assert idx._targets_by_uuid(storage, ["a", "b"]) == [t, None]
    storage.target_storage.get_by_uuid.assert_called_once_with(["a", "b"])


# -- W&B: a run pinning the scan is warned about once ------------------------


def _wandb_run(run_id, age):
    created = (datetime.now(timezone.utc) - age).replace(tzinfo=None).isoformat()
    return SimpleNamespace(
        id=run_id,
        created_at=created,
        state="running",
        job_type="gb-lineage-event",
        config={"job_id": "j"},
    )


def _pin_warnings(caplog):
    return [
        r
        for r in caplog.records
        if "holds the lineage index checkpoint" in r.getMessage()
    ]


def test_a_long_running_lineage_run_is_warned_about_once(caplog):
    ix = idx.WandBLineageIndexer(sink=MagicMock(), api=MagicMock())
    run = _wandb_run("r1", timedelta(minutes=45))
    with caplog.at_level(logging.WARNING, logger=idx.logger.name):
        assert ix._is_pending(run)
        assert ix._is_pending(run)
    (warning,) = _pin_warnings(caplog)
    assert "r1" in warning.getMessage()
    assert "315 more minutes" in warning.getMessage()


def test_a_recent_running_run_is_not_warned_about(caplog):
    ix = idx.WandBLineageIndexer(sink=MagicMock(), api=MagicMock())
    with caplog.at_level(logging.WARNING, logger=idx.logger.name):
        assert ix._is_pending(_wandb_run("r1", timedelta(minutes=5)))
    assert not _pin_warnings(caplog)


def test_each_pinning_run_gets_its_own_warning(caplog):
    ix = idx.WandBLineageIndexer(sink=MagicMock(), api=MagicMock())
    with caplog.at_level(logging.WARNING, logger=idx.logger.name):
        ix._is_pending(_wandb_run("r1", timedelta(minutes=40)))
        ix._is_pending(_wandb_run("r2", timedelta(minutes=40)))
    assert len(_pin_warnings(caplog)) == 2


def test_keyset_read_releases_one_key_in_exact_instant_order():
    """SQLite ties instants within a millisecond; the scan skips any job behind
    the last one it took, so ties must come out in exact (instant, uuid) order,
    also when the tie spans a page boundary."""
    base = T0 + timedelta(minutes=1)
    # Listed in the storage's (key, uuid) order: one millisecond, uuids against
    # the instant order, then a later key.
    listed = [
        (1, _target("a", 0)),
        (1, _target("b", 0)),
        (1, _target("c", 0)),
        (2, _target("d", 0)),
    ]
    for (_, t), micros in zip(listed, (900, 500, 100, 0)):
        t.finished_at = base + timedelta(microseconds=micros)
    listed[3][1].finished_at = base + timedelta(milliseconds=1)
    pages = {None: listed[:2], (1, "b"): listed[2:]}

    def page(storage, cutoff, after):
        return pages.get(after, [])

    indexer = idx.TargetLineageIndexer(sink=MagicMock(), rows=MagicMock())
    with (
        patch.object(idx, "_SCAN_PAGE_SIZE", 2),
        patch.object(idx, "_finished_after_page", side_effect=page),
    ):
        order = [t.uuid for t in indexer._jobs_since(MagicMock(), None)]
    assert order == ["c", "b", "a", "d"]
