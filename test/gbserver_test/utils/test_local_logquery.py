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

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from gbserver.storage.stored_event import StoredEvent
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.logs import Item, LogqueryResponse, QueryDef, QueryParams
from gbserver.utils.local_logquery import LocalLogQueryAPI


def _make_stored_event(
    build_id: str = "build-1",
    step_id: str = "step-1",
    source: str = "hello",
    msg: str = "hello world",
    timestamp: datetime | None = None,
) -> StoredEvent:
    """Helper to create a StoredEvent with a MESSAGE_EVENT payload."""
    if timestamp is None:
        timestamp = datetime(2026, 3, 26, 12, 0, 0)
    return StoredEvent(
        build_event=BuildEvent(
            run_metadata=EntityRunMetadata(
                build_id=build_id,
                targetsteprun_id=step_id,
            ),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=msg),
            timestamp=timestamp,
            source=source,
        )
    )


def _make_query(
    build_id: str = "build-1",
    step_id: str | None = None,
    step_name: str | None = None,
) -> Item:
    """Helper to create an Item query like gbcli sends."""
    json_object = {
        "kubernetes.labels.granite-dot-build/build-id": [build_id],
    }
    if step_id is not None:
        json_object["kubernetes.labels.granite-dot-build/build-step-id"] = [step_id]
    if step_name is not None:
        json_object["kubernetes.labels.granite-dot-build/build-step-name"] = [step_name]
    return Item(
        queryDef=QueryDef(
            startDate=0,
            endDate=int(datetime(2026, 12, 31).timestamp() * 1000),
            type="freeText",
            queryParams=QueryParams(
                metadata={"applicationName": ["granite-build"]},
                jsonObject=json_object,
            ),
        )
    )


class TestLocalLogQueryAPI:
    """Tests for LocalLogQueryAPI.query_cloud_logquery()."""

    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_basic_build_id_query(self, mock_get_storage):
        """Querying by build_id returns matching MESSAGE_EVENTs as LogqueryResponseLogs."""
        events = [
            _make_stored_event(build_id="build-1", msg="line 1"),
            _make_stored_event(build_id="build-1", msg="line 2"),
        ]
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = events
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        resp = api.query_cloud_logquery(_make_query(build_id="build-1"))

        assert isinstance(resp, LogqueryResponse)
        assert resp.status == 200
        assert len(resp.logs) == 2
        assert resp.total == 2

        # Verify text is JSON-wrapped with "log" key
        for i, log_entry in enumerate(resp.logs):
            parsed = json.loads(log_entry.text)
            assert parsed["log"] == f"line {i + 1}"
            assert log_entry.logId is not None
            assert log_entry.timestamp is not None

    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_step_id_filter(self, mock_get_storage):
        """Querying with step_id passes it through to event storage filter."""
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = []
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        api.query_cloud_logquery(_make_query(build_id="build-1", step_id="step-42"))

        call_args = mock_storage.event_storage.get_by_where.call_args[0][0]
        assert call_args["build_id"] == "build-1"
        assert call_args["step_id"] == "step-42"
        assert call_args["type"] == "MESSAGE_EVENT"

    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_step_name_filter(self, mock_get_storage):
        """Querying with step_name maps to source column filter."""
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = []
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        api.query_cloud_logquery(_make_query(build_id="build-1", step_name="hello"))

        call_args = mock_storage.event_storage.get_by_where.call_args[0][0]
        assert call_args["source"] == "hello"

    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_response_compatible_with_output_format_plain(self, mock_get_storage):
        """Response.output_format_plain() works correctly with local log entries."""
        events = [_make_stored_event(msg="hello from docker")]
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = events
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        resp = api.query_cloud_logquery(_make_query())

        plain = resp.output_format_plain()
        assert "hello from docker" in plain

    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_get_build_logs(self, mock_get_storage):
        """get_build_logs() returns plain text log content for a build."""
        events = [
            _make_stored_event(build_id="build-1", msg="step started"),
            _make_stored_event(build_id="build-1", msg="step completed"),
        ]
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = events
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        logs_str = api.get_build_logs(build_id="build-1")

        assert "step started" in logs_str
        assert "step completed" in logs_str

    @patch("gbserver.utils.local_logquery.sleep")
    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_get_build_logs_retries_on_empty(self, mock_get_storage, mock_sleep):
        """get_build_logs() retries when logs are empty, matching IBM Cloud behavior."""
        empty_events = []
        real_events = [_make_stored_event(build_id="build-1", msg="appeared")]
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.side_effect = [
            empty_events,
            empty_events,
            real_events,
        ]
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        logs_str = api.get_build_logs(
            build_id="build-1", max_retries=5, retry_interval=1
        )

        assert "appeared" in logs_str
        assert mock_storage.event_storage.get_by_where.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1)

    @patch("gbserver.utils.local_logquery.sleep")
    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_get_build_logs_empty_ok_skips_retry(self, mock_get_storage, mock_sleep):
        """get_build_logs() returns immediately when empty_ok=True and logs are empty."""
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = []
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        logs_str = api.get_build_logs(
            build_id="build-1", empty_ok=True, max_retries=5, retry_interval=1
        )

        assert logs_str.strip() == ""
        assert mock_storage.event_storage.get_by_where.call_count == 1
        mock_sleep.assert_not_called()

    @patch("gbserver.utils.local_logquery.sleep")
    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_get_build_logs_exhausts_retries(self, mock_get_storage, mock_sleep):
        """get_build_logs() returns empty string after exhausting retries."""
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = []
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        logs_str = api.get_build_logs(
            build_id="build-1", max_retries=3, retry_interval=1
        )

        assert logs_str.strip() == ""
        assert mock_storage.event_storage.get_by_where.call_count == 3
        assert mock_sleep.call_count == 3

    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_empty_results(self, mock_get_storage):
        """Returns empty LogqueryResponse when no events match."""
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.return_value = []
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        resp = api.query_cloud_logquery(_make_query(build_id="nonexistent"))

        assert resp.status == 200
        assert len(resp.logs) == 0
        assert resp.total == 0

    @patch("gbserver.utils.local_logquery.get_admin_storage")
    def test_error_handling(self, mock_get_storage):
        """Returns error LogqueryResponse on storage exception."""
        mock_storage = MagicMock()
        mock_storage.event_storage.get_by_where.side_effect = RuntimeError("db error")
        mock_get_storage.return_value = mock_storage

        api = LocalLogQueryAPI()
        resp = api.query_cloud_logquery(_make_query())

        assert resp.status == 400
        assert "db error" in resp.error


class TestGetLogManagerStandalone:
    """Tests for get_log_manager() and get_log_server_manager() standalone mode."""

    @patch.dict("os.environ", {"GB_ENVIRONMENT": "STANDALONE"})
    def test_get_log_manager_returns_local_in_standalone(self):
        """get_log_manager() returns LocalLogQueryAPI when standalone mode is on."""
        import gbserver.utils.cloud_logquery as mod

        mod._LOG_MANAGER = None  # Reset singleton
        try:
            manager = mod.get_log_manager()
            assert isinstance(manager, LocalLogQueryAPI)
        finally:
            mod._LOG_MANAGER = None  # Clean up singleton

    @patch.dict("os.environ", {"GB_ENVIRONMENT": "STANDALONE"})
    def test_get_log_server_manager_returns_local_in_standalone(self):
        """get_log_server_manager() returns LocalLogQueryAPI when standalone mode is on."""
        import gbserver.utils.cloud_logquery_server as mod

        mod._LOG_SERVER_MANAGER = None  # Reset singleton
        try:
            manager = mod.get_log_server_manager()
            assert isinstance(manager, LocalLogQueryAPI)
        finally:
            mod._LOG_SERVER_MANAGER = None  # Clean up singleton
