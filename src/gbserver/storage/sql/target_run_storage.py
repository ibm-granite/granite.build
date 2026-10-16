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
from typing import Any, List, Optional, Self, Tuple

from sqlalchemy import and_, desc, func, literal_column, or_, tuple_

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage, retry_read
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
        # Serves get_successful_finished_since, which the lineage indexer runs every
        # scan: ``status = SUCCESS`` seeks into the index and ``finished_at`` comes
        # out in the query's order, so a page is read newest-first from the index
        # and stops at its LIMIT instead of sorting the whole table. Only an index:
        # no target row is written to add it. On Postgres it also serves the range
        # filter; on SQLite that reads the blob (see the method), but the ordered
        # scan with an early stop is most of the win.
        kwargs["composite_indexes"] = [("status", "finished_at")]
        super().__init__(**kwargs)

    @retry_read
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
        if not self._ensure_table_without_retry():
            return []
        model = self._sql_alchemy_model
        if dialect == "sqlite":
            # Not bounded on the finished_at column: a margin of the widest UTC
            # offset would cover the local-wall-clock spelling, but the rows kept
            # for the caller (null, unparseable or offset-less JSON) carry no
            # relation to the column's value, and OR-ing them back in would scan
            # anyway. The index still serves status and the order, so a page stops
            # at its LIMIT; only a sparse tail past the cutoff reads to the end.
            raw: Any = literal_column(self._SQLITE_FINISHED_RAW)
            recorded = func.julianday(raw)
            # A malformed blob cannot be rebuilt into an item, so it is left out
            # rather than failing the whole page.
            since = and_(
                func.json_valid(getattr(model, JSON_COLUMN_NAME)),
                or_(
                    recorded >= func.julianday(cutoff_utc.isoformat()),
                    recorded.is_(None),
                    ~or_(raw.like("%Z"), raw.op("GLOB")("*[+-][0-9][0-9]:[0-9][0-9]")),
                ),
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

    # The instant a target finished, as SQLite orders it: ``julianday`` of the JSON
    # blob's offset-carrying ``finished_at`` (the column cannot be compared, see
    # get_successful_finished_since). Literal SQL, not a bound expression, so a
    # query matches the expression index (:meth:`_finished_key_index_ddl`)
    # exactly; SQLite never matches one against a bound parameter. ``json_valid``
    # keeps a malformed blob from failing every write to the table once indexed.
    _SQLITE_FINISHED_RAW = (
        f"(CASE WHEN json_valid({JSON_COLUMN_NAME}) "
        f"THEN json_extract({JSON_COLUMN_NAME}, '$.finished_at') END)"
    )
    _SQLITE_FINISHED_KEY = f"julianday{_SQLITE_FINISHED_RAW}"
    # The widest UTC offset, in days: how far ``julianday`` can misplace a value
    # with no offset suffix, which it reads as UTC.
    _MAX_OFFSET_DAYS = 14 / 24

    def _finished_key_index_ddl(self) -> str:
        """The SQLite expression index the keyset read seeks on.

        Only an index: no row is rewritten, and dropping it restores the table as
        it was. ``uuid`` is in it so the keyset's tie-break is served too.
        """
        return (
            f"CREATE INDEX IF NOT EXISTS ix_{self.table_name}_status_finished_key "
            f"ON {self.table_name} (status, {self._SQLITE_FINISHED_KEY}, "
            f"{UUID_COLUMN_NAME})"
        )

    @retry_read
    def get_successful_finished_after(
        self,
        cutoff_utc: Optional[datetime],
        after: Optional[Tuple[Any, str]],
        page_size: int,
    ) -> List[Tuple[Any, StoredTargetRun]]:
        """Keyset page of successful targets, oldest finished first.

        SQLite: the key is :data:`_SQLITE_FINISHED_KEY`, read from the same JSON
        value the item is rebuilt from, so it follows the item's instant; SQLite
        holds times to the millisecond, so instants within one millisecond tie and
        the caller orders them. A row whose JSON time SQLite cannot place (null key
        while a value is present, or a value with no offset, which ``julianday``
        reads as UTC but Python as local time) would be misordered, so while any
        exists this raises and the caller reads the old way.

        Postgres: the key is the ``finished_at`` timestamp column. A successful
        row with no column value (written before the column existed) cannot be
        placed, and raises the same way.
        """
        assert self._engine is not None
        dialect = self._engine.dialect.name
        if dialect not in ("sqlite", "postgresql"):
            raise NotImplementedError(f"no keyset read on {dialect}")
        if not self._ensure_table_without_retry():
            return []
        model = self._sql_alchemy_model
        uuid_col = getattr(model, UUID_COLUMN_NAME)
        success = model.status == Status.SUCCESS.name
        if dialect == "sqlite":
            key: Any = literal_column(self._SQLITE_FINISHED_KEY)
            raw: Any = literal_column(self._SQLITE_FINISHED_RAW)
            cutoff_key: Any = (
                func.julianday(cutoff_utc.isoformat()) if cutoff_utc else None
            )
            offset_less = ~or_(
                raw.like("%Z"), raw.op("GLOB")("*[+-][0-9][0-9]:[0-9][0-9]")
            )
            # Two probes, each seeking on the expression index: a value SQLite
            # cannot parse (null key), and one with no offset that julianday may
            # misplace by up to the widest offset -- only near or past the cutoff,
            # since one further back is behind it under any reading. A malformed
            # blob reads as no value at all (``raw`` is null): it cannot be rebuilt
            # into an item either way, so it is passed over, not a reason to stop.
            near = (
                key >= cutoff_key - self._MAX_OFFSET_DAYS
                if cutoff_key is not None
                else key.isnot(None)
            )
            unplaceable = [
                and_(success, key.is_(None), raw.isnot(None)),
                and_(success, near, offset_less),
            ]
        else:
            key = model.finished_at
            unplaceable = [and_(success, model.finished_at.is_(None))]
            cutoff_key = cutoff_utc
        session = self._get_session_without_retry()
        try:
            if after is None:
                # On every scan's first page, not once per process: a row written
                # later that cannot be placed would otherwise be passed over for
                # good. Each probe is an index seek, so this stays cheap.
                for probe in unplaceable:
                    if session.query(uuid_col).filter(probe).first() is not None:
                        raise NotImplementedError(
                            "successful targets with a completion time that cannot "
                            "be ordered in SQL"
                        )
            query = session.query(model, key).filter(success, key.isnot(None))
            if cutoff_key is not None:
                query = query.filter(key >= cutoff_key)
            if after is not None:
                query = query.filter(tuple_(key, uuid_col) > tuple_(*after))
            rows = query.order_by(key, uuid_col).limit(page_size).all()
            assert self._column_types is not None
            return [
                (
                    row_key,
                    self._convert_row_dict_to_item(
                        {name: getattr(row, name) for name in self._column_types}
                    ),
                )
                for row, row_key in rows
            ]
        finally:
            session.close()
