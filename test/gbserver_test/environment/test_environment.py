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

import pytest
import yaml

from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.types.buildevent import (
    ArtifactPushedEventPayload,
    ArtifactType,
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
)


@pytest.fixture
def test_data_dir() -> Path:
    src_file_dir = Path(__file__).resolve().parent
    assert src_file_dir.is_dir()
    # print("src_file_dir.parts", src_file_dir.parts)
    path_paths: List[str] = []
    test_done = False
    # start from the end and replace
    for x in src_file_dir.parts[::-1]:
        if not test_done and x == "test":
            test_done = True
            path_paths.append("test-data")
            continue
        path_paths.append(x)
    test_data_dir = Path(*path_paths[::-1])
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
        EventLogLineParserConfig.model_validate(event_config) for event_config in _event_configs
    ]
    event_q = asyncio.Queue()
    launch_id = "7c930009-e59d-4bd9-befc-b5bf80f1330d"
    entityrun_metadata = EntityRunMetadata(build_id=launch_id)
    return (event_q, event_configs, entityrun_metadata)


class TestEnvironment:
    @pytest.mark.asyncio
    async def test_get_events_from_log_line(self: Self, test_data_dir: Path) -> None:
        log_line = "Pushed URI: lh://prod/granite_dot_build.public/tables/gb_tuning_data"
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
