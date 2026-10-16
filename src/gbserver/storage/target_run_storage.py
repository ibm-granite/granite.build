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

from datetime import datetime
from typing import Self

from gbserver.storage.storage import (
    BaseItemStorage,
    IItemStorage,
    Pagination,
    QueryControl,
    SortOrder,
)
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import GB_TARGET_RUNS_TABLE_NAME
from gbserver.types.status import Status


class IStoredTargetRunStorage(IItemStorage[StoredTargetRun]):

    def get_successful_finished_since(
        self, cutoff_utc: datetime, page_index: int, page_size: int
    ) -> list[StoredTargetRun]:
        """One newest-finished-first page of successful targets near ``cutoff_utc``.

        A read-only prefilter, never the exact answer: a backend may return rows
        before the cutoff (this base form returns every successful target), and the
        caller re-checks each ``finished_at``. It must never drop a row at or after
        the cutoff, nor one whose time it cannot read.
        """
        raise NotImplementedError


class BaseStoredTargetRunStorage(
    BaseItemStorage[StoredTargetRun], IStoredTargetRunStorage
):

    def __init__(self: Self, **kwargs):
        kwargs["item_class"] = StoredTargetRun
        if (
            kwargs.get("table_name") is None
        ):  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_TARGET_RUNS_TABLE_NAME
        super().__init__(**kwargs)

    def get_successful_finished_since(
        self, cutoff_utc: datetime, page_index: int, page_size: int
    ) -> list[StoredTargetRun]:
        """The fallback: every successful target, paged as the lineage scan pages."""
        return self.get_by_where(
            {"status": Status.SUCCESS.name},
            query_control=QueryControl(
                pagination=Pagination(index=page_index, size=page_size),
                sort_orders=[SortOrder(column="finished_at", ascending=False)],
            ),
        )

    def _get_column_values(self: Self, item: StoredTargetRun) -> dict:
        fields_to_include = {
            "name",
            "build_id",
            "status",
            "target_hash",
            # Exposed as a column so lineage reconciliation can sort/paginate
            # successful targets newest-first by completion time and stop at its
            # watermark, fetching only newly-finished targets each scan.
            "finished_at",
        }
        json = item.model_dump(include=fields_to_include)
        json["status"] = item.status.name
        return json

    @classmethod
    def _get_sample_item(cls) -> StoredTargetRun:
        """Implemented per superclass requirements to return an item for use by BaseItemStorage"""
        item = StoredTargetRun(
            build_id="build_id",
            environment_uri="space://some-env",
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        return item
