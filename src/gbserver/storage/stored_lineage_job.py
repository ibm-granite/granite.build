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

"""Storage model for one job execution.

One row per ``job_id``, where :mod:`gbserver.storage.stored_lineage_row` is one row
per ``(input, job_id, output)``. That difference is the whole point of this table.

A job with N inputs and M outputs decomposes into N*M lineage rows, and the job's
metadata is copied onto every one of them (``decompose.JOB_METADATA_KEYS``). Four
payloads were therefore excluded from the row blob -- ``job_input_params`` (the full
step configs), ``execution_stats``, ``job_output_stats`` and ``source_code_details``
-- because each would be stored N*M times and the index would be dominated by data
it never queries.

**Here N*M is 1, so the reason for excluding them does not apply** and they are
carried in full, in :attr:`attributes`. That closes the gap recorded in
``lineage/graph_builder.py``: a run entry served from the index no longer has those
four fields empty, and a caller no longer has to fall back to
``GET /lineage/target/{id}`` to see a step config.

As in the row table, the split is "what a query filters on" versus "what a response
carries". :attr:`attributes` is ``Text``, so nothing in it is queryable, indexable,
or validated on write; promoting a column is a deliberate bar to clear.

Three constraints on the promoted columns, each already load-bearing elsewhere in
this schema:

**Every promoted column is a string, by construction.** ``get_by_where`` builds an
``IN`` clause only for string-typed columns and otherwise degrades *silently* to
``column == [list]`` -- a meaningless predicate that returns plausible but wrong rows
with no error. Keeping the promoted set all-text makes that failure unreachable
rather than merely avoided by convention.

**Column widths are inferred, not declared.** A promoted ``str`` becomes
``String(256)``; the only wider-column mechanism is the module-level
``_WIDE_STRING_COLUMNS`` frozenset in the SQL layer, which is keyed on column *name*
and so applies to every table in the system having a column of that name. That is
why nothing here is named ``input`` or ``output`` (those are 512 globally, for the
row table's URIs) and why 256 is treated as the ceiling. ``job_namespace`` is
``"<space_name>/<build_name>"`` and fits comfortably.

**Timestamps keep their source's string form.** :attr:`started_at` is stored as
``gb_targets`` spells it. It is never rewritten to UTC here, and nothing in this
module touches ``gb_targets``.

There is deliberately no ``tags`` column yet, and no authorization logic anywhere in
this module: this table's only job right now is to persist the full job record.
"""

from typing import Any, Dict

from pydantic import Field

from gbserver.storage.stored_build import BaseStoredItem
from gbserver.storage.stored_lineage_row import utc_now_iso


class StoredLineageJob(BaseStoredItem):
    """One job execution, with its full metadata.

    Attributes:
        job_id: identity of the execution, and the join key to
            ``gb_lineage_index.job_id``. Unique. It is the only identifier every lineage
            source has by definition, which is why it -- rather than any process id
            -- keys this table: a build or a target run is granite.build's own
            concept and is absent from every imported job.
        job_namespace: the execution's namespace, ``"<space_name>/<build_name>"`` as
            the producers spell it. Carried verbatim rather than split, so this
            column and the row blob agree on one spelling. Load-bearing for
            authorization on the read paths that consume it: a job that loses it
            fails closed.
        space_name: space the job ran in, when the source has that notion; ``""``
            otherwise (Lakehouse has no such concept).
        owner: username the execution is attributed to, or ``""`` when unknown.
        source_system: which system produced this job -- ``granite.build``,
            ``lakehouse``, ``openlineage``. A column here, unlike in the row table,
            because a per-job listing is a plausible filter.
        status: job status as the source reported it.
        started_at: start timestamp in the source's own string form; see the module
            docstring.
        recorded_at: when this index wrote the record, UTC ISO-8601, stamped at
            write time. Distinct from :attr:`started_at` (the source's form, never
            rewritten): this is our clock, and the basis for a future
            high-water-mark incremental import.
        attributes: everything a response carries and no query filters on -- the
            light job detail (name, type, category, completion time), the
            originating system's ids, and the four large payloads this table exists
            to hold. Not queryable.
    """

    job_id: str = Field(..., description="Identity of the job execution")
    job_namespace: str = Field(
        default="",
        description='Execution namespace, "<space_name>/<build_name>"',
    )
    space_name: str = Field(
        default="", description="Space the job ran in; empty if the source has none"
    )
    owner: str = Field(default="", description="Username the job is attributed to")
    source_system: str = Field(
        default="", description="Which system produced this job"
    )
    status: str = Field(default="", description="Job status as reported")
    started_at: str = Field(
        default="", description="Start timestamp in the source's own string form"
    )
    recorded_at: str = Field(
        default_factory=utc_now_iso,
        description="UTC ISO-8601 time this index wrote the record",
    )

    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job detail, origin ids and the large payloads; not queryable",
    )
