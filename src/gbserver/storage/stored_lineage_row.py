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

"""Storage model for one flat lineage row.

A row is an ``(input, job_id, output)`` triple where the two endpoints are
**normalized URIs**. The URI is the identity: two rows describe the same artifact
exactly when their endpoint strings are equal, which is what lets lineage from a
source with no per-artifact uuid (Lakehouse, dmf-ng) be a first-class citizen
instead of something an alias table has to compensate for.

The graph has no node table. Its nodes are the distinct ``input``/``output``
values, and the traversal walks by matching one row's endpoint against another's
-- which is why those two columns are indexed.

Everything that is not needed to *find* a row lives in :attr:`attributes`, the
JSON blob. The blob is ``Text`` and therefore neither queryable nor indexable, so
the split is exactly "what a query filters on" versus "what a response carries".

**Nothing here is specific to granite.build.** A row is a job and two artifacts,
which is all any lineage source has; there is deliberately no ``build_id`` or
``target_run_uuid`` column, because a build and a target run are granite.build's
process concepts and every imported source leaves them empty -- an indexed column
that is blank on most rows indexes nothing. Lakehouse, for instance, has a job id
and no notion of either. Callers that need a process-scoped view resolve that
scope in their own system first and seed the walk with the URIs it produced, so
the index only ever answers one question: what is the lineage of this artifact.

That also means ``job_id`` carries the whole burden of "which rows were written
together": it is the only identifier every source has by definition, so it is what
the sink's dedup asks about.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict

from pydantic import Field

from gbserver.storage.stored_build import BaseStoredItem

# Terminal marker for input/output. NOT NULL: in SQL, NULL never equals NULL, so
# NULL endpoints would slip past the (job_id, input, output) unique index and
# leave creation/deletion rows -- the least visible ones -- as the only rows a
# re-ingest could duplicate. The traversal runs in Python and the unique key
# exists, so the sentinel costs nothing: the walk stops on a falsy endpoint
# exactly as it would on None.
#
# This is also what ``normalize_uri`` returns for a URI it cannot identify, and
# the coincidence is deliberate: "no artifact on this side" and "no identifiable
# artifact on this side" must both terminate a path rather than mint a node that
# silently fails to merge.
TERMINAL = ""

# Width of the ``input``/``output`` columns, restated for the SQL layer so the DB
# column and the guard that keeps a URI inside it cannot drift.
#
# 512 rather than 1024, even though the other URI-bearing columns in this schema
# get 1024: both of these are indexed AND both sit in the
# ``(job_id, input, output)`` unique index, and MySQL caps an index key at 3072
# bytes. Widening them makes that index larger still, and
# ``__create_unique_indexes`` only *warns* when an index cannot be created -- so
# an over-wide column would not fail loudly, it would silently cost the unique
# index, and with it re-ingest idempotence.
#
# It is wider than the generic 256 a promoted string column gets because real
# artifact URIs have already overrun that: a Lakehouse dataset name long enough to
# pass 256 on its own has been observed (a red-teaming dataset whose name repeats
# as its table name).
#
# A URI longer than this is DROPPED rather than truncated. The column truncates
# silently, and two distinct artifacts sharing a long prefix would then collapse
# into one graph node -- provenance invented where nobody can see it. Losing a node
# is visible and fixable; that is not.
MAX_LINEAGE_URI_LENGTH = 512


class JobStore(str, Enum):
    """Where a row's full job data lives; the index itself only carries labels.

    Names the store to look in, not the system that produced the job: a Lakehouse
    job imported into ``gb_lineage_job`` is ``LINEAGE_JOB``. The key to look it up
    by is in the row's ``attributes.retrieve``.

    A ``str`` enum so the column stays text (see ``_get_column_values``) and a row
    serializes to the plain value.
    """

    LINEAGE_JOB = "lineage_job"  # gb_lineage_job
    WANDB = "wandb"  # a W&B run
    TARGETS = "targets"  # gb_targets (and its build)
    OTHER = "other"  # an external system; the reference is opaque


def utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601, the ``recorded_at`` form.

    Fixed-width (microseconds always present, ``+00:00`` offset), so the column's
    lexicographic order is its chronological order.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class StoredLineageRow(BaseStoredItem):
    """One lineage row: an input, the job that ran, and an output.

    Attributes:
        job_id: identity of the job execution. The same value on every row of one
            job, which is what keeps an N*M decomposition regroupable: the rows of
            a job with 3 inputs and 2 outputs still say which inputs and which
            outputs that execution had. It is also what the read path groups on to
            rebuild a run node.
        input: normalized URI of the input artifact, or :data:`TERMINAL` when the
            job had no input (a creation).
        output: normalized URI of the output artifact, or :data:`TERMINAL` when
            the job produced none (a deletion).
        attributes: everything else -- artifact kind and name per endpoint, the
            carried job metadata (name, status, owner, timestamps), the space and
            owner the read path reports, and any process ids the originating system
            had (a build, a target run, a pipeline). Lives in the JSON blob,
            so nothing here is queryable; anything that needs filtering has to
            become a column first, and that is a deliberate bar to clear.
        job_store: where the job's full data lives (:class:`JobStore`). The read
            path dispatches on it to fetch a job's detail; the row's ``attributes``
            only carry what is needed to draw and label the graph. Which system
            *produced* the job is not a row column: it is on the job record
            (``attributes.origin.system``), once per job rather than N*M times.
        recorded_at: when this index wrote the row, UTC ISO-8601, stamped at write
            time. Distinct from the job's ``started_at``, which keeps the source's
            own form: this is *our* clock, not the producer's, and is the basis for
            a future high-water-mark incremental import. Not part of the unique
            key, so a re-ingest of the same relation still dedups.
    """

    job_id: str = Field(..., description="Identity of the job execution")
    input: str = Field(
        default=TERMINAL,
        description="Normalized URI of the input artifact; TERMINAL if none",
    )
    output: str = Field(
        default=TERMINAL,
        description="Normalized URI of the output artifact; TERMINAL if none",
    )

    job_store: JobStore = Field(
        default=JobStore.OTHER,
        description="Where the full job data lives",
    )

    recorded_at: str = Field(
        default_factory=utc_now_iso,
        description="UTC ISO-8601 time this index wrote the row",
    )

    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node, job and provenance detail; not queryable",
    )

    def is_self_loop(self) -> bool:
        """Whether the job rewrote its own input.

        Legitimate for unversioned entities: several runs rewriting one table
        converge on a single node. The traversal includes such a row but does not
        chain through it.
        """
        return self.input == self.output and self.input != TERMINAL
