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

"""SQL storage implementation for lineage rows."""

from gbserver.storage.lineage_row_storage import (
    BaseLineageRowStorage,
    ILineageRowStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_lineage_row import StoredLineageRow


class SQLLineageRowStorage(
    BaseSQLItemStorage[StoredLineageRow],
    BaseLineageRowStorage,
    ILineageRowStorage,
):
    """SQL-based storage implementation for lineage rows.

    Schema decisions, and why each column is indexed:

    - ``input`` / ``output`` carry the graph traversal. Every hop is
      ``WHERE input IN (frontier)`` or ``WHERE output IN (frontier)``, so without
      these two indexes a walk degrades to a full scan per level. They hold
      normalized URIs, which is also what makes the *root* lookup an index seek:
      a request names an artifact by URI, so there is finally something indexed to
      match it against.
    - ``job_id`` groups the rows of one execution, which is what makes an N*M
      decomposition regroupable, and it also backs the sink's presence-based dedup,
      run on every scan. The prototype joins and groups on it but leaves it
      unindexed -- a gap corrected here.
    - ``recorded_at`` is this index's own write time, UTC ISO-8601, and the default
      pagination order. Indexed because it is what a future high-water-mark
      incremental import will range over. Deliberately not in the unique key: it
      differs on every write, so including it would defeat the dedup.

    There is no autoincrement ``index`` column: nothing reads it, so ``uuid`` is the
    primary key.

    There is deliberately no ``build_id`` or ``target_run_uuid`` column. Those are
    granite.build's process concepts, empty on every imported row, so indexing them
    would index blanks for most of the table. A caller wanting a process-scoped view
    resolves that scope in its own system and seeds the walk with the resulting
    URIs, which keeps this index answering exactly one question: what is the lineage
    of this artifact.

    Every indexed column is text. ``get_by_where`` builds an ``IN`` clause only for
    string-typed columns and silently degrades to ``column == [list]`` otherwise, so
    a non-text indexed column is a latent wrong-results bug rather than merely a
    slow one. ``source_system`` lives in the JSON blob rather than in a column: it is
    read to label a run node, never to filter one.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = [
            "input",
            "output",
            "job_id",
            "recorded_at",
        ]
        # One row per (job, input, output). A second source reporting the same
        # relation is a no-op, which is what makes re-ingest idempotent -- a
        # property the prototype lacks entirely (it has no key at all and
        # executemany's without a guard, so it duplicates silently).
        #
        # This is why input/output hold the "" sentinel rather than NULL for
        # terminals: in SQL, NULL never equals NULL, so NULL endpoints would slip
        # past this index and leave creation/deletion rows unprotected. Note also
        # that unique indexes are only created with the table, so this cannot be
        # added later without recreating it -- and that __create_unique_indexes
        # only *warns* on failure, so an index too wide for the backend's key
        # limit costs idempotence silently. That is why the URI columns are 512
        # and not 1024 (see MAX_LINEAGE_URI_LENGTH).
        kwargs["unique_columns"] = {("job_id", "input", "output"): None}
        kwargs["default_pagination_sort_by_column"] = "recorded_at"
        super().__init__(**kwargs)
