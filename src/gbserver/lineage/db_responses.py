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

from typing import Any, Dict, Iterable, List, Optional

from gbserver.lineage.attributes import (
    INPUT,
    JOB_CATEGORY,
    JOB_COMPLETED_AT,
    JOB_NAME,
    JOB_NAMESPACE,
    JOB_OWNER,
    JOB_STARTED_AT,
    JOB_STATUS,
    JOB_TYPE,
    OUTPUT,
    PAYLOAD,
    PAYLOAD_EXECUTION_STATS,
    PAYLOAD_INPUT_PARAMS,
    endpoint_kind,
    endpoint_name,
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

    Owner and namespace decide who may read the job, so they are taken only when
    every row agrees; rows that disagree give empty ones, which the access check
    fails closed on. On W&B anyone can post an event under an existing job id
    (``POST /lineage/`` is unauthenticated), so trusting the first row would let a
    forged row with the poster as owner unlock the real job's detail.
    """
    attributes = rows[0].attributes if rows else {}
    job = job_detail(attributes)
    namespace = _agreed(rows, JOB_NAMESPACE)
    return {
        "job_id": job_id,
        "job_namespace": namespace,
        # The space is the namespace's first segment, as the access filter reads it.
        "space_name": namespace.split("/", 1)[0] if namespace else "",
        "owner": _agreed(rows, JOB_OWNER),
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


def _agreed(rows: List, key: str) -> str:
    """The ``job`` group's ``key`` when every row that sets it agrees, else ``""``.

    A row that leaves it blank does not count against the others: blanks grant
    nothing, and the merge fills them in only on a later write.
    """
    values = {str(job_detail(r.attributes).get(key, "") or "") for r in rows} - {""}
    return values.pop() if len(values) == 1 else ""


# What any caller may see of a job, whatever its space: the fields a lineage trace
# needs to draw and label a run. ``POST /lineage/graph`` crosses spaces by design,
# and ``GET /lineage/jobs`` only expands a graph node, so both are held to this set.
# Owner, namespace/space, tags and the origin's ids describe a granite.build job
# beyond the trace and are returned only behind the space check, by
# ``GET /lineage/jobs/{id}``.
GLOBAL_JOB_FIELDS = (
    "job_id",
    "source_system",
    "status",
    "started_at",
    "inputs",
    "outputs",
)
GLOBAL_JOB_GROUP_KEYS = (
    JOB_NAME,
    JOB_TYPE,
    JOB_STATUS,
    JOB_STARTED_AT,
    JOB_COMPLETED_AT,
    JOB_CATEGORY,
)


def global_job_entry(entry: Dict) -> Dict:
    """``entry`` trimmed to :data:`GLOBAL_JOB_FIELDS`, for a cross-space reader."""
    trimmed = {key: entry[key] for key in GLOBAL_JOB_FIELDS if key in entry}
    job = entry.get("job") or {}
    trimmed["job"] = {key: job[key] for key in GLOBAL_JOB_GROUP_KEYS if key in job}
    return trimmed


def _search_datasets(rows: Iterable[Any], side: str) -> List[Dict]:
    """The distinct artifacts on one side of a job's rows, as OpenLineage datasets.

    The same shape the W&B provider returns (``namespace``, ``name``, ``uri``,
    ``facets``); ``N/A`` where the index has no namespace, as W&B does. Terminal
    endpoints (a creation's source, a deletion's target) are not artifacts.
    """
    datasets: Dict[str, Dict] = {}
    for row in rows:
        uri = getattr(row, side)
        if not uri or uri == TERMINAL or uri in datasets:
            continue
        kind = endpoint_kind(row.attributes, side)
        datasets[uri] = {
            "namespace": "N/A",
            "name": endpoint_name(row.attributes, side) or uri,
            "uri": uri,
            "facets": {"artifact_type": kind} if kind else {},
        }
    return list(datasets.values())


def job_search_event(job: StoredLineageJob, rows: Optional[List[Any]] = None) -> Dict:
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

    ``inputs``/``outputs`` come from the job's index ``rows``, as the W&B provider
    fills them from the run's artifacts; a job not indexed yet has none.
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
        "inputs": _search_datasets(rows or [], INPUT),
        "outputs": _search_datasets(rows or [], OUTPUT),
    }
