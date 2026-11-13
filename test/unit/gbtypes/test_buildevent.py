#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

"""Tests for buildevent types."""

import json
from unittest.mock import MagicMock, patch

import pytest

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    BuildEventWorkloadStatusPayload,
    EntityRunMetadata,
    EventPayload,
)
from gbserver.types.status import Status

_STEP_LINE = "Step: 3/10 loss=0.7012"


class TestEntityRunMetadata:
    def test_from_dict_reads_build_config_name(self):
        meta = EntityRunMetadata.from_dict(
            {"build_id": "abc", "build_config_name": "foo"}
        )
        assert meta.build_config_name == "foo"

    def test_from_dict_defaults_build_config_name_empty(self):
        meta = EntityRunMetadata.from_dict({"build_id": "abc"})
        assert meta.build_config_name == ""

    def test_to_dict_includes_build_config_name(self):
        meta = EntityRunMetadata(build_id="abc", build_config_name="foo")
        d = meta.to_dict()
        assert d["build_config_name"] == "foo"

    def test_round_trip_preserves_build_config_name(self):
        meta = EntityRunMetadata(build_id="abc", build_config_name="foo")
        assert EntityRunMetadata.from_dict(meta.to_dict()).build_config_name == "foo"


def _step_progress_rule(line_regex: str = "Step:\\s\\d+") -> EventLogLineParserConfig:
    """The ``Step:`` progress rule several step monitors declare (e.g. hf-peft-sft)."""
    return EventLogLineParserConfig(
        event_type="WORKLOAD_STATUS_EVENT",
        line_regex=line_regex,
        event_fields=[
            {"field_name": "status", "field_value_template": "RUNNING"},
            {"field_name": "message", "field_regex": "Step:.+"},
        ],
    )


@pytest.mark.standalone
class TestWorkloadStatusPayloadMessage:
    """The workload-status payload's ``message`` field (#598).

    Step monitor rules send ``message`` with a WORKLOAD_STATUS_EVENT (e.g. the
    ``Step:`` progress rule). Before #598 the payload had no such field, so
    ``payload_parser`` raised a TypeError and the event was dropped.
    """

    def test_payload_parser_accepts_message(self):
        """payload_parser builds the payload with the message a monitor rule sends."""
        payload = EventPayload.payload_parser(
            BuildEventType.WORKLOAD_STATUS_EVENT,
            {"status": "RUNNING", "message": "Step: 3/10 loss=0.7"},
        )
        assert isinstance(payload, BuildEventWorkloadStatusPayload)
        assert payload.message == "Step: 3/10 loss=0.7"
        # Pins the parser's current behaviour (no coercion to Status), not desired behaviour.
        assert payload.status == "RUNNING"

    def test_payload_parser_without_message_defaults_empty(self):
        """A payload with no message still parses, with an empty message."""
        payload = EventPayload.payload_parser(
            BuildEventType.WORKLOAD_STATUS_EVENT, {"status": "RUNNING"}
        )
        assert isinstance(payload, BuildEventWorkloadStatusPayload)
        assert payload.message == ""

    def test_existing_constructor_call_sites_still_work(self):
        """``(status=...)`` and ``(status=..., data=...)`` callers keep working."""
        assert BuildEventWorkloadStatusPayload(status=Status.FAILED).message == ""
        payload = BuildEventWorkloadStatusPayload(status=Status.FAILED, data={"k": "v"})
        assert payload.data == {"k": "v"}
        assert payload.message == ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "line_regex, expected_message",
        [
            # The narrow rule the legacy hand-authored openinstruct-sft assets
            # (and others) still ship.
            # get_events_from_log_line narrows the line to the line_regex match
            # before field regexes run, so the message is only "Step: <n>".
            ("Step:\\s\\d+", "Step: 3"),
            # The wide rule hf-peft-sft ships: it spans the line, so the message
            # carries the whole progress text.
            ("Step:\\s\\d+.*", _STEP_LINE),
        ],
    )
    async def test_step_progress_line_yields_workload_status_with_message(
        self, line_regex, expected_message
    ):
        """A ``Step:`` line through the monitor rule yields one event carrying the message."""
        with patch("gbserver.environment.environment.logger") as mock_logger:
            events = await Environment.get_events_from_log_line(
                _STEP_LINE,
                [_step_progress_rule(line_regex)],
                entityrun_metadata=EntityRunMetadata(),
            )
        mock_logger.error.assert_not_called()
        assert len(events) == 1
        event = events[0]
        assert event.type is BuildEventType.WORKLOAD_STATUS_EVENT
        assert isinstance(event.payload, BuildEventWorkloadStatusPayload)
        assert event.payload.message == expected_message
        # Pins the parser's current behaviour (no coercion to Status), not desired behaviour.
        assert event.payload.status == "RUNNING"

    @pytest.mark.parametrize("message", [_STEP_LINE, ""])
    def test_stored_event_json_round_trip_keeps_message(self, message):
        """The event-storage JSON path (to_json_dict -> from_json_dict) keeps the message."""
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="b"),
            type=BuildEventType.WORKLOAD_STATUS_EVENT,
            payload=BuildEventWorkloadStatusPayload(
                status=Status.RUNNING, message=message
            ),
        )
        reread = BuildEvent.from_json_dict(json.loads(json.dumps(event.to_json_dict())))
        assert isinstance(reread.payload, BuildEventWorkloadStatusPayload)
        assert reread.payload.message == message
        assert reread.payload.status == Status.RUNNING

    @pytest.mark.parametrize(
        "message, fmt, args",
        [
            (
                _STEP_LINE,
                "Workload status is %s (%s) for build %s",
                (Status.RUNNING, _STEP_LINE),
            ),
            ("", "Workload status is %s for build %s", (Status.RUNNING,)),
        ],
    )
    def test_build_runner_logs_message_when_present(self, message, fmt, args):
        """The build runner's workload-status log line includes a non-empty message."""
        run_metadata = EntityRunMetadata(build_id="b")
        event = BuildEvent(
            run_metadata=run_metadata,
            type=BuildEventType.WORKLOAD_STATUS_EVENT,
            payload=BuildEventWorkloadStatusPayload(
                status=Status.RUNNING, message=message
            ),
        )
        with patch("gbserver.buildrunner.buildrunner.logger") as mock_logger:
            # The handler only reads the event, so a stand-in self is enough.
            # pylint: disable-next=protected-access
            handler = BuildRunner._BuildRunner__process_workload_status_event  # type: ignore[attr-defined]
            handler(MagicMock(), event=event)
        mock_logger.info.assert_called_once_with(fmt, *args, run_metadata)
