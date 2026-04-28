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

    @classmethod
    def _get_sample_item(cls) -> StoredEvent:
        """Implemented per superclass requirements to return an item for use by BaseItemStorage"""
        build_event_type = BuildEventType.STATUS_EVENT
        run_metadata = EntityRunMetadata()
        payload_data = {}
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
