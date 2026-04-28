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
import time
from typing import Any, List, Self, Type

from pydantic import BaseModel, Field

from gbserver.types.constants import (
    FETCH_CLOUD_LOGS_MAX_PAGE_SIZE,
    FETCH_CLOUD_LOGS_TIME_RANGE,
)


class QueryItem(BaseModel):
    text: str
    type: str
    syntax: str


class QueryParams(BaseModel):
    query: QueryItem | None = None
    metadata: Any = None
    jsonObject: Any = None


class SortModel(BaseModel):
    field: str
    ordering: str
    missing: str


class QueryDef(BaseModel):
    nowDate: int | None = None
    startDate: int  # in milliseconds
    endDate: int  # in milliseconds
    pageSize: int | None = None
    pageIndex: int | None = None
    type: str
    queryParams: QueryParams | None = None
    sortModel: List[SortModel] | None = None


class Item(BaseModel):
    """IBM Cloud Logs API query."""

    queryDef: QueryDef

    @classmethod
    def get_logs_for_build(cls: Type[Self], build_id: str) -> Self:
        """Returns a query that fetches logs for a particular build."""
        endDate = int(time.time())
        startDate = endDate - FETCH_CLOUD_LOGS_TIME_RANGE
        queryParams = QueryParams(
            metadata={"applicationName": ["granite-build"]},
            jsonObject={"kubernetes.labels.granite-dot-build/build-id": [build_id]},
        )
        sortModel = SortModel(
            field="timestamp",
            ordering="desc",
            missing="_last",
        )
        self = cls(
            queryDef=QueryDef(
                startDate=startDate * 1000,  # in milliseconds
                endDate=endDate * 1000,  # in milliseconds
                pageSize=FETCH_CLOUD_LOGS_MAX_PAGE_SIZE,
                pageIndex=0,
                type="freeText",
                queryParams=queryParams,
                sortModel=[sortModel],
            )
        )
        return self


class LogqueryResponseLogs(BaseModel):
    """A single log line from the IBM Cloug Logging API."""

    templateType: int | None = None
    branchId: str | None = None
    metadata: Any = None
    logId: str | None = None
    jsonUuid: str | None = None
    templateId: str | None = None
    timestamp: float | None = None
    logIndex: str | None = None
    text: str | None = None
    index: int = 0


class LogqueryResponse(BaseModel):
    """Response from IBM Cloug Logging API."""

    status: int = 0
    logs: List[LogqueryResponseLogs] = Field(default_factory=list)
    total: int = 0
    page_index: int = 0
    error: str = ""

    def output_format_plain(
        self: Self, reverse: bool = False, max_size: int = -1
    ) -> str:
        """Output all/some of the log lines joined by new lines."""
        log_msgs = []
        log_list = self.logs
        if reverse:
            log_list = log_list[::-1]
        for log in log_list:
            if log.text is None:
                continue
            log_json = json.loads(log.text)
            assert isinstance(log_json, dict)
            log_msgs.append(log_json.get("log", "<null>"))
        if max_size <= 0 or len(log_msgs) == 0:
            return "\n".join(log_msgs)
        # reverse
        log_msgs_reversed = log_msgs[::-1]
        # filter last few lines
        log_msgs_smaller = []
        so_far = 0
        for log_msg in log_msgs_reversed:
            new_len = so_far + 1 + len(log_msg)
            if new_len > max_size:
                break
            log_msgs_smaller.append(log_msg)
            so_far = new_len
        # reverse again
        log_msgs_smaller_rev = log_msgs_smaller[::-1]
        return "\n".join(log_msgs_smaller_rev)
