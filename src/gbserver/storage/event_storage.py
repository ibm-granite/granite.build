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
from abc import abstractmethod
from typing import Optional

from gbserver.storage.storage import (
    CREATED_TIME_FIELD_NAME,
    UUID_COLUMN_NAME,
    BaseItemStorage,
    IItemStorage,
)
from gbserver.storage.stored_event import StoredEvent
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
)
from gbserver.types.constants import GB_EVENTS_TABLE_NAME


class IStoredEventStorage(IItemStorage[StoredEvent]):

    @abstractmethod
    def get_sorted_build_events(
        self, build_id: str, where: Optional[dict] = None
    ) -> list[StoredEvent]:
        """Return the matching build events by the order they were inserted into storage.
        This relies on the fact that an autoincrementing column named 'index' is available in the table.

        Args:
            build_id (str): The build for which events are being requested.
            where (Optional[dict], optional): additional query parameters, for example on source or type. Defaults to None.

        Returns:
            list[StoredEvent]: list of stored events sorted by the order they were added to storage.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def get_max_index(self) -> int:
        """Get the maximum index value from gb_events table.

        Used by LineageWatcher to seed its watermark at startup.

        Returns:
            int: Maximum index value, or 0 if no events exist.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )

    @abstractmethod
    def get_events_after_index(self, min_index: int) -> list[tuple[int, StoredEvent]]:
        """Get all events with index > min_index, ordered ascending.

        Used by LineageWatcher to poll for new events.

        Args:
            min_index (int): Return events with index > min_index.

        Returns:
            list[tuple[int, StoredEvent]]: (index, event) pairs ordered by
            ascending index.
        """
        raise NotImplementedError(
            f"Sub-class {self.__class__.__name__} did not implement method throwing this exception"
        )


class BaseStoredEventStorage(BaseItemStorage[StoredEvent], IStoredEventStorage):

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredEvent
        if (
            kwargs.get("table_name") is None
        ):  # Allow for testing using alternate table names.
            kwargs["table_name"] = GB_EVENTS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredEvent) -> dict:
        values = {}
        values["build_id"] = item.build_event.run_metadata.build_id
        values["target_id"] = item.build_event.run_metadata.targetrun_id
        values["step_id"] = item.build_event.run_metadata.targetsteprun_id
        values["type"] = item.build_event.type.name
        values["source"] = item.build_event.source
        values[CREATED_TIME_FIELD_NAME] = item.build_event.timestamp
        values["username"] = item.build_event.run_metadata.username
        return values

    def _convert_item_to_json_str(self, item: StoredEvent) -> str:
        json_dict = {}
        json_dict[UUID_COLUMN_NAME] = item.uuid
        build_event_dict = item.build_event.to_json_dict()
        json_dict["build_event"] = build_event_dict
        json_str = json.dumps(json_dict)
        return json_str

    def _convert_json_str_to_item(self, json_str: str) -> StoredEvent:
        json_dict = json.loads(json_str)
        uuid = json_dict[UUID_COLUMN_NAME]
        build_event_dict = json_dict["build_event"]
        assert isinstance(build_event_dict, dict)
        build_event = BuildEvent.from_json_dict(build_event_dict)
        item = StoredEvent(uuid=uuid, build_event=build_event)
        return item

    def get_sorted_build_events(
        self, build_id: str, where: Optional[dict] = None
    ) -> list[StoredEvent]:
        """Return the matching build events by the order they were inserted into storage.
        This relies on the fact that an autoincrementing column named 'index' is available in the table.

        Args:
            build_id (str): The build for which events are being requested.
            where (Optional[dict], optional): additional query parameters, for example on source or type. Defaults to None.

        Returns:
            list[StoredEvent]: list of stored events sorted by the order they were added to storage.
        """
        # Get the requested build  events
        inner_where = {"build_id": build_id}
        if where:
            inner_where = inner_where | where
        events = self.get_by_where(inner_where)

        # Key them by their uuid so we can sort them later.
        event_dict = {}
        for event in events:
            event_dict[event.uuid] = event

        # Get the indexes of these builds and sort them
        rows = self._get_by_where_row_dicts(
            inner_where
        )  # This method is the way to get back the 'index' column values
        sorted_rows = sorted(rows, key=lambda row: row["index"])

        # Sort the builds by the list of sorted indexes
        sorted_events = []
        for row in sorted_rows:
            uuid = row[UUID_COLUMN_NAME]
            sorted_events.append(event_dict[uuid])
        return sorted_events

    def get_max_index(self) -> int:
        """Get the maximum index value from gb_events table.

        Used by LineageWatcher to seed its watermark at startup.

        The query orders by ``index`` descending and pages a single row, so the
        database returns one row rather than the whole table.

        Returns:
            int: Maximum index value, or 0 if no events exist.
        """
        from gbserver.storage.storage import Pagination, QueryControl, SortOrder

        try:
            query_control = QueryControl(
                sort_orders=[SortOrder(column="index", ascending=False)],
                pagination=Pagination(index=0, size=1),
            )
            rows = self._get_by_where_row_dicts(where=None, query_control=query_control)
            if not rows:
                return 0
            # `or 0` guards a NULL index column: `.get("index", 0)` only covers a
            # missing key, not a present-but-None value, which would otherwise
            # break the `> min_index` comparison in get_events_after_index.
            return rows[0].get("index") or 0
        except Exception as e:
            # Returning 0 here conflates "no events" with "query failed"; that is
            # acceptable because this only seeds the watermark at startup and
            # lineage recording is replay-safe (deterministic runIds +
            # resume="allow" + content-dedupe), so a low reseed only reprocesses.
            self.logger.warning(f"Failed to get max index: {e}")
            return 0

    def get_events_after_index(self, min_index: int) -> list[tuple[int, StoredEvent]]:
        """Get all events with index > min_index, ordered ascending.

        Used by LineageWatcher to poll for new events. The row index is returned
        alongside each event because StoredEvent does not carry the autoincrement
        index itself.

        The database is paged in descending ``index`` order and paging stops as
        soon as a row at or below ``min_index`` is seen. In steady state (few new
        events per poll) this reads a single small page rather than scanning the
        whole ``gb_events`` table. Paging restarts at page 0 on every call, so a
        large accumulated backlog (watcher fell behind, or a burst exceeding
        ``page_size`` between polls) re-reads from the top each poll; cost is
        bounded by the backlog size, not the table size, and drains as the
        watcher catches up.

        Args:
            min_index (int): Return events with index > min_index.

        Returns:
            list[tuple[int, StoredEvent]]: (index, event) pairs ordered by
            ascending index.
        """
        from gbserver.storage.storage import Pagination, QueryControl, SortOrder

        page_size = 100
        try:
            new_rows: list[dict] = []
            page_index = 0
            done = False
            while not done:
                query_control = QueryControl(
                    sort_orders=[SortOrder(column="index", ascending=False)],
                    pagination=Pagination(index=page_index, size=page_size),
                )
                rows = self._get_by_where_row_dicts(
                    where=None, query_control=query_control
                )
                if not rows:
                    break
                for row in rows:
                    # `or 0` guards a NULL index column (see get_max_index).
                    if (row.get("index") or 0) > min_index:
                        new_rows.append(row)
                    else:
                        # Rows are descending, so the first row at or below the
                        # watermark means every remaining row is older too.
                        done = True
                        break
                if len(rows) < page_size:
                    break
                page_index += 1

            if not new_rows:
                return []

            # Fetch the matching events in a single bulk query, then pair each
            # with its index (StoredEvent does not carry the autoincrement
            # index). Ascending index order for the caller.
            new_rows.sort(key=lambda row: row.get("index") or 0)
            uuids = [
                row[UUID_COLUMN_NAME] for row in new_rows if row.get(UUID_COLUMN_NAME)
            ]
            if not uuids:
                return []
            # A single IN (...) query rather than one get_by_uuid per row.
            events_by_uuid = {
                event.uuid: event
                for event in self.get_by_where({UUID_COLUMN_NAME: uuids})
            }

            result: list[tuple[int, StoredEvent]] = []
            for row in new_rows:
                uuid = row.get(UUID_COLUMN_NAME)
                if not uuid:
                    continue
                event = events_by_uuid.get(uuid)
                if event:
                    result.append((row.get("index") or 0, event))
            return result
        except Exception as e:
            # Match get_max_index: a query failure degrades to "no new events"
            # (empty) and is logged at warning, not error. The watcher retries on
            # its next poll, so a transient failure is not lost.
            self.logger.warning(f"Failed to get events after index {min_index}: {e}")
            return []

    @classmethod
    def _get_sample_item(cls) -> StoredEvent:
        """Implemented per superclass requirements to return an item for use by BaseItemStorage"""
        build_event_type = BuildEventType.STATUS_EVENT
        run_metadata = EntityRunMetadata()
        payload_data = {}  # type: ignore[var-annotated]
        build_event = BuildEvent(
            run_metadata=run_metadata,
            type=build_event_type,
            payload=EventPayload.payload_parser(
                event_type=build_event_type,
                data=payload_data,
            ),
        )
        item = StoredEvent(build_event=build_event)
        return item
