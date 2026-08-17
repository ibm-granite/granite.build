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

"""Unit tests for the LLMB_STEP_METADATA_KEY/VALUE step-metadata stdout hook.

Covers the event/payload plumbing, the StoredStepRun.metadata field, and that all
three builtin monitor configs (bash/docker/skypilot) parse the marker — including
the anchoring difference (bash anchors, skypilot/docker do not).
"""

from pathlib import Path

import pytest
import yaml

from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.types.buildevent import (
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
    StepMetadataUpdateEventPayload,
)

_MONITORS = Path(__file__).resolve().parents[3] / "src/gbserver/builtins/monitors"
_SHA = "deadbeefcafe0123456789abcdef012345678901"
_MARKER = f"LLMB_STEP_METADATA_KEY:commit_hash LLMB_STEP_METADATA_VALUE:{_SHA}"


def _configs_for(monitor: str) -> list[EventLogLineParserConfig]:
    """Load a builtin monitor's event_configs as compiled parser configs.

    :param monitor: monitor subdir name under builtins/monitors (bash/docker/skypilot).
    :returns: the monitor's event_configs, compiled.
    """
    raw = yaml.safe_load((_MONITORS / monitor / "monitor.yaml").read_text())
    return [EventLogLineParserConfig(**ec) for ec in raw["config"]["event_configs"]]


async def _parse(monitor: str, line: str) -> list:
    """Run a log line through a monitor's configs, returning STEP_METADATA events."""
    events = await Environment.get_events_from_log_line(
        line, _configs_for(monitor), entityrun_metadata=EntityRunMetadata()
    )
    return [e for e in events if e.type is BuildEventType.STEP_METADATA_UPDATE_EVENT]


@pytest.mark.standalone
def test_payload_parser_builds_step_metadata_payload():
    """payload_parser maps the event type to StepMetadataUpdateEventPayload."""
    payload = EventPayload.payload_parser(
        BuildEventType.STEP_METADATA_UPDATE_EVENT,
        {"metadata_key": "commit_hash", "metadata_value": _SHA},
    )
    assert isinstance(payload, StepMetadataUpdateEventPayload)
    assert payload.metadata_key == "commit_hash"
    assert payload.metadata_value == _SHA


@pytest.mark.standalone
def test_event_type_is_not_internal():
    """The metadata event is history-worthy, not an internal sentinel."""
    assert not BuildEventType.STEP_METADATA_UPDATE_EVENT.is_internal_event()


@pytest.mark.standalone
def test_stored_step_run_metadata_defaults_and_old_row_compat():
    """metadata defaults to {} and old rows without the key deserialize to {}."""
    from gbserver.storage.stored_step_run import StoredStepRun

    fresh = StoredStepRun(build_id="b", target_id="t", definition_uri="d")
    assert fresh.metadata == {}
    old_row = StoredStepRun.model_validate(
        {"uuid": "u", "build_id": "b", "target_id": "t", "definition_uri": "d"}
    )
    assert old_row.metadata == {}


@pytest.mark.standalone
@pytest.mark.asyncio
@pytest.mark.parametrize("monitor", ["bash", "docker", "skypilot"])
async def test_all_monitors_parse_marker(monitor):
    """Every builtin monitor parses the marker into the correct payload."""
    events = await _parse(monitor, _MARKER)
    assert len(events) == 1, f"{monitor}: expected 1 event, got {len(events)}"
    payload = events[0].payload
    assert isinstance(payload, StepMetadataUpdateEventPayload)
    assert payload.metadata_key == "commit_hash"
    assert payload.metadata_value == _SHA


@pytest.mark.standalone
@pytest.mark.asyncio
@pytest.mark.parametrize("monitor", ["bash", "docker", "skypilot"])
@pytest.mark.parametrize("trailer", ["\r", "  ", "\t", " \r"])
async def test_trailing_whitespace_trimmed_from_value(monitor, trailer):
    """Trailing whitespace/CR (e.g. CRLF logs) is not folded into the value.

    Guards the value regex against capturing to end-of-line: a trailing ``\\r`` would
    otherwise corrupt the recorded SHA and fail a ``^[0-9a-f]{40}$`` assertion.
    """
    events = await _parse(monitor, _MARKER + trailer)
    assert len(events) == 1
    assert events[0].payload.metadata_value == _SHA


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_bash_anchor_rejects_prefixed_line():
    """Bash anchors the marker, so an echoed/prefixed line does not match."""
    assert await _parse("bash", "byoc: echoing " + _MARKER) == []


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_skypilot_unanchored_matches_prefixed_line():
    """SkyPilot's retrieved logs are prefixed, so the marker must match unanchored."""
    events = await _parse("skypilot", "(job, pid=123) " + _MARKER)
    assert len(events) == 1
    assert events[0].payload.metadata_value == _SHA
