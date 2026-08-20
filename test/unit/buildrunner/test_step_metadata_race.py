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

"""Regression tests for the step-metadata event-ordering race.

A STEP_METADATA_UPDATE_EVENT (parsed from step stdout) and the STATUS_EVENT that
creates the StoredStepRun row come from different producers with no ordering
guarantee. If the metadata event is processed first, its value must not be lost:
the buildrunner buffers it in ``_pending_step_metadata`` and flushes it onto the
row once the row exists (from both the metadata handler and the step status
handler). These tests pin that buffer/flush behavior directly on
``BuildRunner._apply_pending_step_metadata`` and the metadata handler, using a
stub ``self`` so no full BuildRunner has to be constructed.
"""

from types import SimpleNamespace

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.types.buildevent import StepMetadataUpdateEventPayload

_SID = "step-uuid-1"

# The metadata handler is name-mangled (private), so reach it via its mangled name.
_process_metadata = BuildRunner._BuildRunner__process_step_metadata_update_event


class _FakeStepStorage:
    """Minimal step_storage backed by a dict: supports get_by_uuid / update."""

    def __init__(self):
        self._rows: dict = {}

    def add(self, item: StoredStepRun) -> None:
        self._rows[item.uuid] = item

    def get_by_uuid(self, uuid):
        return self._rows.get(uuid)

    def update(self, item: StoredStepRun) -> None:
        self._rows[item.uuid] = item


def _stub_runner(storage: _FakeStepStorage) -> SimpleNamespace:
    """A stand-in ``self`` carrying only what the metadata path touches.

    The metadata handler calls ``self._apply_pending_step_metadata`` internally, so
    the real (unbound) method is wired onto the namespace bound back to it.
    """
    runner = SimpleNamespace(
        _pending_step_metadata={},
        storage=SimpleNamespace(step_storage=storage),
    )
    runner._apply_pending_step_metadata = (
        lambda tid, _r=runner: BuildRunner._apply_pending_step_metadata(_r, tid)
    )
    return runner


def _row() -> StoredStepRun:
    """A minimal StoredStepRun with the fixed test uuid."""
    return StoredStepRun(
        uuid=_SID, build_id="b1", target_id="t1", definition_uri="step://x"
    )


def _meta_event(key: str, value: str) -> SimpleNamespace:
    """A STEP_METADATA_UPDATE_EVENT stand-in with payload + run_metadata."""
    return SimpleNamespace(
        payload=StepMetadataUpdateEventPayload(metadata_key=key, metadata_value=value),
        run_metadata=SimpleNamespace(targetsteprun_id=_SID),
    )


def _meta_event_no_id(targetsteprun_id) -> SimpleNamespace:
    """A metadata event whose run_metadata carries no usable targetsteprun_id."""
    return SimpleNamespace(
        payload=StepMetadataUpdateEventPayload(metadata_key="k", metadata_value="v"),
        run_metadata=SimpleNamespace(targetsteprun_id=targetsteprun_id),
    )


def test_metadata_event_without_targetsteprun_id_is_dropped():
    """A stray/uncorrelated marker (no targetsteprun_id) is a no-op, not a raise.

    Guards against an AssertionError here failing the whole build under
    GBSERVER_RAISE_BUILD_EXCEPTIONS. Both ``None`` and ``""`` are treated as absent.
    """
    for bad_id in (None, ""):
        runner = _stub_runner(_FakeStepStorage())
        _process_metadata(runner, _meta_event_no_id(bad_id))
        # Nothing buffered, nothing raised.
        assert runner._pending_step_metadata == {}


def test_flush_noop_when_row_absent():
    """Buffered metadata is retained (not dropped) while the row does not exist."""
    runner = _stub_runner(_FakeStepStorage())
    runner._pending_step_metadata[_SID] = {"commit_hash": "abc"}
    BuildRunner._apply_pending_step_metadata(runner, _SID)
    assert runner._pending_step_metadata[_SID] == {"commit_hash": "abc"}


def test_race_metadata_before_row_is_not_lost():
    """The reviewer's scenario: metadata processed before the row is created.

    The handler runs first (row absent) and buffers; once the row is created and
    the flush runs again (as the step status handler does), the value lands on the
    row and the buffer is cleared -- nothing is lost.
    """
    storage = _FakeStepStorage()
    runner = _stub_runner(storage)
    # Metadata event arrives BEFORE any row exists.
    _process_metadata(runner, _meta_event("commit_hash", "deadbeef"))
    assert storage.get_by_uuid(_SID) is None  # no row yet
    assert runner._pending_step_metadata[_SID] == {"commit_hash": "deadbeef"}
    # Status event later creates the row and triggers the flush.
    storage.add(_row())
    BuildRunner._apply_pending_step_metadata(runner, _SID)
    assert storage.get_by_uuid(_SID).metadata == {"commit_hash": "deadbeef"}
    assert _SID not in runner._pending_step_metadata  # buffer cleared


def test_metadata_applied_immediately_when_row_present():
    """Common case: row already exists, so the handler applies in one pass."""
    storage = _FakeStepStorage()
    storage.add(_row())
    runner = _stub_runner(storage)
    _process_metadata(runner, _meta_event("commit_hash", "cafe"))
    assert storage.get_by_uuid(_SID).metadata == {"commit_hash": "cafe"}
    assert _SID not in runner._pending_step_metadata


def test_flush_preserves_existing_metadata():
    """Flushing merges into (does not replace) metadata already on the row."""
    storage = _FakeStepStorage()
    row = _row()
    row.metadata = {"existing": "1"}
    storage.add(row)
    runner = _stub_runner(storage)
    runner._pending_step_metadata[_SID] = {"commit_hash": "abc"}
    BuildRunner._apply_pending_step_metadata(runner, _SID)
    assert storage.get_by_uuid(_SID).metadata == {"existing": "1", "commit_hash": "abc"}
