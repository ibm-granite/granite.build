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

"""Local logquery module."""

import json
from time import sleep
from typing import List, Self

from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_event import StoredEvent
from gbserver.types.constants import (
    FETCH_CLOUD_LOGS_MAX_RETRIES,
    FETCH_CLOUD_LOGS_RETRY_INTERVAL,
)
from gbserver.types.logs import Item, LogqueryResponse, LogqueryResponseLogs
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class LocalLogQueryAPI:
    """Log query backend for standalone mode.

    Reads MESSAGE_EVENT entries from the local gb_events table
    and returns LogqueryResponse objects compatible with gbcli.
    """

    def get_build_logs(
        self: Self,
        build_id: str,
        empty_ok: bool = False,
        max_retries: int = FETCH_CLOUD_LOGS_MAX_RETRIES,
        retry_interval: int = FETCH_CLOUD_LOGS_RETRY_INTERVAL,
        max_size: int = -1,
    ) -> str:
        """Get the logs of a build as a plain text string.

        Mirrors IBMCloudLogQueryAPI.get_build_logs() retry-until-non-empty
        contract so that callers like unwrap_errors.py work in standalone mode.
        """
        retry_count = 0
        logs_str = ""
        query = Item.get_logs_for_build(build_id=build_id)
        while (logs_str.strip() == "") and (retry_count < max_retries):
            try:
                logger.info("get_build_logs retry_count: %d", retry_count)
                retry_count += 1
                logs = self.query_cloud_logquery(query)
                logs_str = logs.output_format_plain(
                    reverse=True,
                    max_size=max_size,
                )
                logger.info("get_build_logs logs length: %d", len(logs_str))
                if logs_str.strip() == "":
                    logger.info("get_build_logs got empty logs")
                    if empty_ok:
                        break
                    logger.info("get_build_logs sleep for %d seconds", retry_interval)
                    sleep(retry_interval)
            except Exception as e:
                logger.error(
                    "failed to fetch the logs for the build %s : %s",
                    build_id,
                    e,
                )
        logger.info("get_build_logs end logs length: %d", len(logs_str))
        return logs_str

    def query_cloud_logquery(self: Self, query: Item) -> LogqueryResponse:
        """Query local event storage for log entries.

        Translates IBM Cloud Logs query parameters to gb_events filters,
        fetches matching MESSAGE_EVENT entries, and maps them to the
        LogqueryResponse format that gbcli expects.
        """
        try:
            where = self._build_where_filter(query)
            storage = get_admin_storage()
            events: List[StoredEvent] = storage.event_storage.get_by_where(where)

            logs = [self._event_to_log(event) for event in events]

            return LogqueryResponse(
                status=200,
                logs=logs,
                total=len(logs),
            )
        except Exception as e:
            logger.error("LocalLogQueryAPI query failed: %s", e)
            return LogqueryResponse(status=400, error=str(e))

    def _build_where_filter(self: Self, query: Item) -> dict:
        """Translate Item query parameters to gb_events column filters."""
        where: dict = {"type": "MESSAGE_EVENT"}

        query_params = query.queryDef.queryParams
        if query_params is None:
            return where

        json_object = query_params.jsonObject
        if json_object is None or not isinstance(json_object, dict):
            return where

        # build_id
        build_ids = json_object.get("kubernetes.labels.granite-dot-build/build-id", [])
        if build_ids:
            where["build_id"] = build_ids[0]

        # step_id
        step_ids = json_object.get("kubernetes.labels.granite-dot-build/build-step-id", [])
        if step_ids:
            where["step_id"] = step_ids[0]

        # step_name -> source column
        step_names = json_object.get("kubernetes.labels.granite-dot-build/build-step-name", [])
        if step_names:
            where["source"] = step_names[0]

        return where

    @staticmethod
    def _event_to_log(event: StoredEvent) -> LogqueryResponseLogs:
        """Convert a StoredEvent to a LogqueryResponseLogs entry.

        The text field must be a JSON string with a "log" key because
        LogqueryResponse.output_format_plain() calls json.loads(log.text)
        and extracts .get("log").
        """
        msg = ""
        if event.build_event.payload is not None:
            msg = getattr(event.build_event.payload, "msg", "") or ""

        timestamp = event.build_event.timestamp.timestamp()

        return LogqueryResponseLogs(
            timestamp=timestamp,
            text=json.dumps({"log": msg}),
            logId=event.uuid,
        )
