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

from sqlalchemy import desc, func, or_

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.storage import JSON_COLUMN_NAME, UUID_COLUMN_NAME
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.storage.target_run_storage import (
    BaseStoredTargetRunStorage,
    IStoredTargetRunStorage,
)
from gbserver.types.status import Status


class SQLTargetRunStorage(
    BaseSQLItemStorage[StoredTargetRun],
    BaseStoredTargetRunStorage,
    IStoredTargetRunStorage,
):

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)

    def get_successful_finished_since(
        self, cutoff_utc: datetime, page_index: int, page_size: int
    ) -> list[StoredTargetRun]:
        """Filter on completion time in SQL; reads only, writes no row.

        Like any first read, it may initialise or adjust the table schema to match
        the item (see ``_ensure_table``).

        SQLite: the ``finished_at`` *column* holds two spellings, one of them local
        wall-clock with its offset dropped, so it cannot be compared to an instant.
        The JSON blob's ``finished_at`` always carries its offset, and it is what
        the item is rebuilt from, so the filter reads that through ``julianday``,
        which honours the offset. Two kinds of value are kept for the caller's exact
        check rather than risk losing the row: one ``julianday`` cannot parse, and
        one with no offset suffix, which ``julianday`` reads as UTC but Python as
        local time.

        Postgres: ``finished_at`` is a real timestamp column, compared directly;
        a NULL is kept, as on SQLite.

        ``uuid`` breaks ties in the order, so offset paging cannot skip or repeat a
        row between pages.
        Any other dialect falls back to the unfiltered read.
        """
        assert self._engine is not None
        dialect = self._engine.dialect.name
        if dialect not in ("sqlite", "postgresql"):
            return super().get_successful_finished_since(
                cutoff_utc, page_index, page_size
            )
        if not self._ensure_table():
            return []
        model = self._sql_alchemy_model
        if dialect == "sqlite":
            raw = func.json_extract(getattr(model, JSON_COLUMN_NAME), "$.finished_at")
            recorded = func.julianday(raw)
            since = or_(
                recorded >= func.julianday(cutoff_utc.isoformat()),
                recorded.is_(None),
                ~or_(raw.like("%Z"), raw.op("GLOB")("*[+-][0-9][0-9]:[0-9][0-9]")),
            )
        else:
            since = or_(model.finished_at >= cutoff_utc, model.finished_at.is_(None))
        session = self._get_session_without_retry()
        try:
            rows = (
                session.query(model)
                .filter(model.status == Status.SUCCESS.name, since)
                .order_by(
                    desc(model.finished_at), desc(getattr(model, UUID_COLUMN_NAME))
                )
                .offset(page_index * page_size)
                .limit(page_size)
                .all()
            )
            assert self._column_types is not None
            return [
                self._convert_row_dict_to_item(
                    {key: getattr(row, key) for key in self._column_types}
                )
                for row in rows
            ]
        finally:
            session.close()
