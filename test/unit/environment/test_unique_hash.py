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

"""Tests for the ``{{ unique_hash }}`` output-URI template variable."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gbcommon.uri.uri import URI
from gbserver.build.build import Build
from gbserver.build.buildrun import BuildRun
from gbserver.environment.environment import Environment
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_event import StoredEvent
from gbserver.types.buildconfig import (
    BuildConfig,
    BuildTargetConfig,
    BuildTargetOutputConfig,
)
from gbserver.types.buildevent import (
    ArtifactEventPayload,
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.utils.template import fill_template
from gbserver.utils.utils import get_uuid, short_alphanumeric_lower_hash

URI_TEMPLATE = "file:///tmp/out/eval_{{ unique_hash }}"
TS = datetime(2026, 10, 7, 12, 0, 0, 123456, tzinfo=timezone.utc)


def _artifact_event(timestamp: datetime = TS, binding_id: str = "out") -> BuildEvent:
    return BuildEvent(
        run_metadata=EntityRunMetadata(
            build_id="b1",
            target_name="t",
            targetrun_id="tr1",
            targetsteprun_id="tsr1",
        ),
        type=BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT,
        payload=ArtifactEventPayload(binding_id=binding_id, binding={"path": "/x"}),
        timestamp=timestamp,
    )


def test_unique_hash_definition():
    epoch_us = int(TS.timestamp()) * 1_000_000 + TS.microsecond
    assert _artifact_event().unique_hash("out") == short_alphanumeric_lower_hash(
        f"tsr1:out:{epoch_us}"
    )


def test_unique_hash_changes_per_event_and_output():
    h = _artifact_event().unique_hash("out")
    assert _artifact_event(TS + timedelta(microseconds=1)).unique_hash("out") != h
    assert _artifact_event().unique_hash("other") != h


def test_unique_hash_is_timezone_independent():
    h = _artifact_event().unique_hash("out")
    pst = TS.astimezone(timezone(timedelta(hours=-7)))
    assert _artifact_event(pst).unique_hash("out") == h
    assert _artifact_event(TS.replace(tzinfo=None)).unique_hash("out") == h


@pytest.mark.parametrize("ts", [TS, TS.replace(microsecond=0)])
def test_unique_hash_recomputable_from_gb_events(ts):
    """The value recomputed from the stored row matches the in-memory one."""
    storage = SqliteStorageFactory().create_event_storage(
        table_name=f"test_unique_hash_{get_uuid().replace('-', '_')}"
    )
    try:
        event = _artifact_event(ts.astimezone())
        event.run_metadata.build_id = get_uuid()
        storage.add(StoredEvent(build_event=event))
        (stored,) = storage.get_sorted_build_events(event.run_metadata.build_id)
        assert stored.build_event.unique_hash("out") == event.unique_hash("out")
    finally:
        storage.delete_table()


def test_non_strict_render_preserves_unique_hash():
    assert fill_template(URI_TEMPLATE, {"run_metadata": {}}) == URI_TEMPLATE


@pytest.mark.asyncio
async def test_buildrun_passes_source_event_hash():
    """BuildRun hashes the source artifact event, not its multi-artifact wrapper."""
    pushasset = MagicMock(side_effect=RuntimeError("stop after pushasset"))
    target = SimpleNamespace(
        config=BuildTargetConfig(
            environment_uri="space://environments/x",
            outputs={"out": BuildTargetOutputConfig(uri=URI_TEMPLATE)},
            steps=[],
        ),
        environment=SimpleNamespace(pushasset=pushasset),
    )
    entity = MagicMock(spec=Build)
    entity.config = MagicMock(spec=BuildConfig)
    entity.targets = {"t": target}
    stub = SimpleNamespace(
        build_id="b1",
        dispatch_event=MagicMock(),
        entity=entity,
        targetruns={"tr1": SimpleNamespace(bindings={})},
        targetrun_additionaljobs_queue={"tr1": asyncio.Queue()},
    )
    event = _artifact_event()
    with pytest.raises(RuntimeError, match="stop after pushasset"):
        await BuildRun._process_event(stub, event, tg=None)  # type: ignore[arg-type]
    assert pushasset.call_args.kwargs["unique_hash"] == event.unique_hash("out")


@pytest.mark.asyncio
async def test_pushasset_renders_unique_hash_once():
    async def _handler(self, **kwargs):
        return None

    env = SimpleNamespace(
        _get_storeconfig=lambda uri, **kw: (
            SimpleNamespace(type="file", get_secrets=lambda: {}),
            SimpleNamespace(push=None),
        ),
        pushasset_types={"file": _handler},
        event_q=asyncio.Queue(),
        asset_bindings={},
    )
    Environment._thread_local.asset_events = {}
    uri = await Environment.pushasset(
        env,  # type: ignore[arg-type]
        task_group=None,  # type: ignore[arg-type]
        binding={"path": "/tmp/out"},
        uristr=URI_TEMPLATE,
        binding_id="out",
        run_metadata=EntityRunMetadata(targetsteprun_id="tsr1"),
        unique_hash="abc12345",
    )
    uristr = URI.get_uristr(uri)
    assert uristr == "file:///tmp/out/eval_abc12345"
    events = []
    while not env.event_q.empty():
        events.append(env.event_q.get_nowait())
    # Registration and push events carry the same rendered URI.
    assert [e.type for e in events] == [
        BuildEventType.ARTIFACT_EVENT,
        BuildEventType.ARTIFACT_PUSHED_EVENT,
    ]
    assert {e.payload.uri for e in events} == {uristr}
