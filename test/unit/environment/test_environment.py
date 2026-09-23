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

import asyncio
from pathlib import Path
from typing import List, Self, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from libgbtest.buildrunner.buildtest import get_test_data_dir_for

from gbcommon.uri.uri import URI
from gbserver.environment.bash import Bash
from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.environment.io.descriptors import InlineDeferredPush
from gbserver.types.buildevent import (
    ArtifactPushedEventPayload,
    ArtifactType,
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
)


@pytest.fixture
def test_data_dir() -> Path:
    test_data_dir = get_test_data_dir_for(__file__)
    assert test_data_dir.is_dir()
    return test_data_dir


def get_test_env(
    test_data_dir: Path,
    test_data_filename: str,
) -> Tuple[asyncio.Queue, List[EventLogLineParserConfig], EntityRunMetadata]:
    yaml_path = test_data_dir / test_data_filename
    assert yaml_path.is_file()
    with open(yaml_path, "r", encoding="utf-8") as f:
        _event_configs = yaml.safe_load(f)["event_configs"]
    event_configs = [
        EventLogLineParserConfig.model_validate(event_config)
        for event_config in _event_configs
    ]
    event_q = asyncio.Queue()
    launch_id = "7c930009-e59d-4bd9-befc-b5bf80f1330d"
    entityrun_metadata = EntityRunMetadata(build_id=launch_id)
    return (event_q, event_configs, entityrun_metadata)


class TestEnvironment:
    @pytest.mark.asyncio
    async def test_get_events_from_log_line(self: Self, test_data_dir: Path) -> None:
        log_line = (
            "Pushed URI: lh://prod/granite_dot_build.public/tables/gb_tuning_data"
        )
        event_q, event_configs, entityrun_metadata = get_test_env(
            test_data_dir, "lhpush_events.yaml"
        )
        events = await Environment.get_events_from_log_line(
            log_line=log_line,
            event_configs=event_configs,
            event_q=event_q,
            entityrun_metadata=entityrun_metadata,
        )
        assert len(events) == 1
        event = events[0]
        expected_event = BuildEvent(
            run_metadata=EntityRunMetadata(
                build_id="7c930009-e59d-4bd9-befc-b5bf80f1330d",
                username="",
                type="",
                target_name="",
                targetrun_id="",
                targetsteprun_id="",
                targetstep_uri="",
            ),
            type=BuildEventType.ARTIFACT_PUSHED_EVENT,
            payload=ArtifactPushedEventPayload(
                uri="lh://prod/granite_dot_build.public/tables/gb_tuning_data",
                binding_id="",
                type=ArtifactType.UNDEFINED,
            ),
        )
        expected_event.timestamp = event.timestamp
        assert event == expected_event

    @pytest.mark.asyncio
    async def test_get_events_from_log_line_event_field_json_template(
        self: Self, test_data_dir: Path
    ) -> None:
        event_q = asyncio.Queue()
        log_line = 'Pushed URI: {"uri": "lh://prod/granite_dot_build.public/tables/gb_tuning_data"}'
        event_q, event_configs, entityrun_metadata = get_test_env(
            test_data_dir, "event_field_json_and_template.yaml"
        )
        events = await Environment.get_events_from_log_line(
            log_line=log_line,
            event_configs=event_configs,
            event_q=event_q,
            entityrun_metadata=entityrun_metadata,
        )
        assert len(events) == 1
        event = events[0]
        expected_event = BuildEvent(
            run_metadata=EntityRunMetadata(
                build_id="7c930009-e59d-4bd9-befc-b5bf80f1330d",
                username="",
                type="",
                target_name="",
                targetrun_id="",
                targetsteprun_id="",
                targetstep_uri="",
            ),
            type=BuildEventType.ARTIFACT_PUSHED_EVENT,
            payload=ArtifactPushedEventPayload(
                uri="lh://prod/granite_dot_build.public/tables/gb_tuning_data",
                binding_id="",
                type=ArtifactType.UNDEFINED,
            ),
        )
        expected_event.timestamp = event.timestamp
        assert event == expected_event


@pytest.fixture
def pushasset_test_env():
    """A real Bash environment with a stubbed hf:// store resolution.

    ``Environment.pushasset`` resolves the URI to a declared assetstore via
    ``_get_storeconfig``; a bare Bash env has none, so we stub the resolution to
    a MagicMock store of type ``hfstore``. The event queue, ``asset_bindings``
    and ``Environment._thread_local.asset_events`` are the environment's real
    objects so the test asserts on genuinely emitted events.
    """
    env = Bash(event_q=asyncio.Queue())
    # asset_events is lazily created in get_or_create_environment; this test
    # constructs the env directly, so seed the per-thread map pushasset writes to.
    if not hasattr(Environment._thread_local, "asset_events"):
        Environment._thread_local.asset_events = {}
    assetstore = MagicMock()
    assetstore.type = "Hfstore"
    assetstore.get_secrets.return_value = {}
    assetstoreenv_config = MagicMock()
    assetstoreenv_config.push = None
    env._get_storeconfig = MagicMock(  # type: ignore[method-assign]
        return_value=(assetstore, assetstoreenv_config)
    )
    return env


def drain_event_queue(event_q: asyncio.Queue) -> List[BuildEvent]:
    """Pop every currently-queued BuildEvent without blocking."""
    events: List[BuildEvent] = []
    while not event_q.empty():
        events.append(event_q.get_nowait())
    return events


@pytest.mark.asyncio
async def test_pushasset_sentinel_emits_created_only(pushasset_test_env):
    """An InlineDeferredPush result emits CREATED and registers the binding but
    suppresses the immediate PUSHED and does not signal asset_events (spec §6)."""
    env = pushasset_test_env
    input_uri = "hf:///ns/out"
    # pushasset canonicalizes the URI, so the binding/asset_events keys use the
    # normalized form, not the input string.
    uristr = URI.get_uristr(URI.get_uri(input_uri))
    with patch.dict(
        env.pushasset_types,
        {"hfstore": AsyncMock(return_value=InlineDeferredPush())},
    ):
        task = env.pushasset(
            task_group=None,
            binding={"path": "/out"},
            uristr=input_uri,
            binding_id="out",
            run_metadata=EntityRunMetadata(build_id="build-sentinel"),
            output_config=None,
        )
        await task

    events = drain_event_queue(env.event_q)
    types = [e.type for e in events]
    # CREATED emitted, immediate PUSHED suppressed (deferred to step monitor).
    assert BuildEventType.ARTIFACT_EVENT in types
    assert BuildEventType.ARTIFACT_PUSHED_EVENT not in types
    # Binding still registered for the produced artifact.
    assert uristr in env.asset_bindings
    # asset_events NOT signalled: the producing step's monitor sets PUSHED later.
    assert not Environment._thread_local.asset_events[uristr].is_set()


@pytest.mark.asyncio
async def test_pushasset_inline_emits_created_and_pushed(pushasset_test_env):
    """A non-sentinel inline push (memstore/envstore-style) keeps today's
    behavior: CREATED + immediate PUSHED and asset_events signalled."""
    env = pushasset_test_env
    input_uri = "hf:///ns/out"
    uristr = URI.get_uristr(URI.get_uri(input_uri))
    with patch.dict(
        env.pushasset_types,
        {"hfstore": AsyncMock(return_value=None)},
    ):
        task = env.pushasset(
            task_group=None,
            binding={"path": "/out"},
            uristr=input_uri,
            binding_id="out",
            run_metadata=EntityRunMetadata(build_id="build-inline"),
            output_config=None,
        )
        await task

    events = drain_event_queue(env.event_q)
    types = [e.type for e in events]
    assert BuildEventType.ARTIFACT_EVENT in types
    assert BuildEventType.ARTIFACT_PUSHED_EVENT in types
    assert Environment._thread_local.asset_events[uristr].is_set()
