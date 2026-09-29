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

"""SQL storage implementation for lineage job tags."""

from gbserver.storage.lineage_job_tag_storage import (
    BaseLineageJobTagStorage,
    ILineageJobTagStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_lineage_job_tag import StoredLineageJobTag


class SQLLineageJobTagStorage(
    BaseSQLItemStorage[StoredLineageJobTag],
    BaseLineageJobTagStorage,
    ILineageJobTagStorage,
):
    """SQL-based storage implementation for lineage job tags.

    - ``tag`` is indexed: it is what every filter queries.
    - ``job_id`` is indexed: it is how a response fetches the tags of the jobs it
      already holds.
    - ``(job_id, tag)`` is unique, so re-recording a job's tags is a no-op rather
      than a duplicate. As in the job table, a unique index is only created with the
      table, and that is why no column here is widened past the inferred 256.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = ["job_id", "tag", "recorded_at"]
        kwargs["unique_columns"] = {("job_id", "tag"): None}
        kwargs["default_pagination_sort_by_column"] = "recorded_at"
        super().__init__(**kwargs)
