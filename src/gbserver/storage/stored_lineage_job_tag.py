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

"""Storage model for one tag on one lineage job.

One row per ``(job_id, tag)``. Tags are free-form strings, by convention ``k=v``
as in W&B (``build_id=...``, ``target_run_uuid=...``, ``space_name=...``, a
user's own build tags). A build is one kind of tag among many, which is why this is
a tag table rather than a ``build_id`` column: a column would be blank on every
imported job, while a tag is simply absent.

Why a table and not a comma-joined ``tags`` column, as the artifact registry does:

- **Width.** A promoted column is ``String(256)`` for *all* of a job's tags
  together; a handful of ``k=<uuid>`` tags already exceeds it. Widening goes
  through ``_WIDE_STRING_COLUMNS``, which is keyed on column name and so would
  widen every table's ``tags``. Here 256 is the limit per tag.
- **Queries.** A joined column is searched with ``LIKE '%tag%'``, which no index
  serves, then filtered exactly in Python. Here ``tag`` is indexed and a filter is
  an exact ``IN``.
- **Duplication does not arise.** The job table is already one row per job, so
  tags are stored once per execution, never once per lineage row.

Every column is a string, for the reason given in
:mod:`gbserver.storage.stored_lineage_job`: ``get_by_where`` builds an ``IN``
only for string-typed columns.
"""

from pydantic import Field

from gbserver.storage.stored_build import BaseStoredItem
from gbserver.storage.stored_lineage_row import utc_now_iso

# The inferred width of a promoted string column. A longer tag is rejected rather
# than truncated: a truncated tag would match a different, shorter tag's queries.
MAX_TAG_LENGTH = 256


class StoredLineageJobTag(BaseStoredItem):
    """One tag attached to one job execution.

    Attributes:
        job_id: the tagged execution; the join key to ``gb_lineage_job.job_id``.
        tag: the tag, verbatim. Matched exactly, never as a substring.
        recorded_at: when this index wrote the tag, UTC ISO-8601.
    """

    job_id: str = Field(..., description="Identity of the tagged job execution")
    tag: str = Field(..., description="The tag, by convention k=v")
    recorded_at: str = Field(
        default_factory=utc_now_iso,
        description="UTC ISO-8601 time this index wrote the tag",
    )


def is_storable_tag(tag: object) -> bool:
    """Whether ``tag`` can be stored: a non-empty string that fits its column."""
    return isinstance(tag, str) and 0 < len(tag) <= MAX_TAG_LENGTH
