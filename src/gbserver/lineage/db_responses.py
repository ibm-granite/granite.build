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

"""Turns what the ``db`` provider stores into what each API contract expects.

One place for that translation, so :mod:`gbserver.lineage.db_service` only has to
query. The two sides it bridges are genuinely independent:

**What is stored** is this provider's own schema -- ``gb_lineage_index`` holds one
flat row per edge, ``gb_lineage_job`` one record per execution with its tags in a
promoted column. No other provider shares it, and none has to: W&B keeps the same
lineage as runs and artifacts in its own service.

**What the API returns** is the shared contract. Every ``ILineageStore`` /
``LineageService`` implementation answers the same endpoint signatures, so a
caller cannot tell the providers apart by the shape of a response -- only by what
is in it. That is why the conversions live here rather than leaking either way:
the storage schema is free to change without touching a route, and a route's shape
is free to change without touching the tables.

Two conversions, because the endpoints ask two different questions:

- :func:`job_listing_entry` -- the flat entry ``GET /lineage/jobs`` and the graph
  routes read, built from a job's index rows.
- :func:`job_search_event` -- the OpenLineage-shaped envelope
  ``POST /lineage/search`` reads, built from a job record. Nested because that
  route is provider-resolved and W&B answers it with real OpenLineage events, so
  the route reaches into ``run.facets`` to filter access.
"""

from typing import Any, Dict, List

from gbserver.lineage.attributes import (
    JOB_NAMESPACE,
    JOB_OWNER,
    JOB_STARTED_AT,
    JOB_STATUS,
    PAYLOAD,
    PAYLOAD_EXECUTION_STATS,
    PAYLOAD_INPUT_PARAMS,
    job_detail,
    origin_detail,
    origin_system,
)
from gbserver.storage.lineage_row_storage import tags_to_map
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import TERMINAL


def job_listing_entry(job_id: str, rows: List, tags: List[str]) -> Dict:
    """One entry of the job listing, read from the job's index rows.

    Job-first: a caller reaching here wants to know which executions matched, and
    what each read and wrote. The endpoints come from the job's own rows, terminals
    left out, so a self-rewrite shows its artifact on both sides. Every row of one
    job carries the same ``job`` group, so the first one speaks for all.

    Flat on purpose: the routes that read this take it as-is. Only the tag search
    needs the nested envelope -- see :func:`job_search_event`.
    """
    attributes = rows[0].attributes if rows else {}
    job = job_detail(attributes)
    namespace = str(job.get(JOB_NAMESPACE, "") or "")
    return {
        "job_id": job_id,
        "job_namespace": namespace,
        # The space is the namespace's first segment, as the access filter reads it.
        "space_name": namespace.split("/", 1)[0] if namespace else "",
        "owner": str(job.get(JOB_OWNER, "") or ""),
        "source_system": origin_system(attributes),
        "status": str(job.get(JOB_STATUS, "") or ""),
        "started_at": str(job.get(JOB_STARTED_AT, "") or ""),
        "tags": tags,
        "inputs": sorted({r.input for r in rows if r.input and r.input != TERMINAL}),
        "outputs": sorted(
            {r.output for r in rows if r.output and r.output != TERMINAL}
        ),
        "job": job,
        "origin": origin_detail(attributes),
    }


def job_search_event(job: StoredLineageJob) -> Dict:
    """One job record in the envelope ``POST /lineage/search`` reads.

    That route is provider-resolved and shared, so it cannot know which provider
    answered: it reaches into ``run.facets.job_details.owner`` and
    ``run.facets.tags.space_name`` to filter access, and masks
    ``run.facets.job_input_params``. A result that omits those is dropped for every
    caller, which looks like an empty index rather than a bug -- so the nesting is
    part of the contract even though nothing is stored that way.

    **The tags facet is reconstructed, not stored twice.** The tags column holds
    exactly what ``job_tags`` derived from the entry's ``run.facets.tags`` (plus the
    origin ids and the build's own labels), so splitting them back into a map
    returns the facet the producer emitted, ``space_name`` and ``username``
    included.

    ``inputs``/``outputs`` are left empty: this answers "which jobs match these
    tags", and what a match read and wrote is a second question the artifact graph
    and ``GET /lineage/jobs/{id}`` already answer from the index rows.
    """
    attributes = job.attributes or {}
    detail = job_detail(attributes)
    payload = attributes.get(PAYLOAD) or {}

    # The facet the producer emitted, recovered from the column.
    tags_facet = dict(tags_to_map(job.tags or []))
    # The column is authoritative for the space: the route fails closed on a
    # missing one, so a record whose tags lost it would be invisible.
    if job.space_name:
        tags_facet.setdefault("space_name", job.space_name)

    job_details: Dict[str, Any] = {"job_id": job.job_id}
    if job.owner:
        job_details["owner"] = job.owner
    if job.status:
        job_details["job_status"] = job.status
    if job.started_at:
        job_details["job_started_at"] = job.started_at

    run_facets: Dict[str, Any] = {"tags": tags_facet, "job_details": job_details}
    # Carried so the route's redaction has something to mask; an absent key stays
    # absent rather than becoming an empty facet.
    for key in (PAYLOAD_INPUT_PARAMS, PAYLOAD_EXECUTION_STATS):
        if payload.get(key) is not None:
            run_facets[key] = payload[key]

    return {
        "eventType": "OTHER",
        "eventTime": job.started_at or job.recorded_at,
        "run": {"runId": job.job_id, "facets": run_facets},
        "job": {
            "namespace": job.job_namespace,
            "name": str(detail.get("name") or ""),
            "facets": {},
        },
        "inputs": [],
        "outputs": [],
    }
