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

"""Serve artifact lineage out of the local index instead of W&B.

``POST /lineage/graph`` delegates to :meth:`DBLineageService.query_graph`, which
answers from the lineage table, so a deployment with no W&B still has lineage.

Only the graph is implemented here. The event-emission and tag-search methods of
the interface belong to the W&B-shaped write path (this service reads an index
another writer populates), so they are explicit no-ops rather than pretending to
succeed -- see each method for why its degenerate value is the safe one.

Seeds are URIs. A request may spell an artifact however it holds it -- a browser
URL or the runtime's own URI -- and it is normalized to the identity the index
columns store, so the root resolves in one indexed lookup.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from gbserver.lineage.attributes import origin_id
from gbserver.lineage.db_responses import (
    global_job_entry,
    job_listing_entry,
    job_search_event,
)
from gbserver.lineage.graph_builder import build_graph_dict
from gbserver.lineage.job_detail import fetch_job_detail
from gbserver.lineage.openlineage_service import LineageService
from gbserver.lineage.uri_normalize import normalize_uri
from gbserver.lineage.walk import (
    DEFAULT_MAX_NODES_PER_LEVEL,
    Direction,
    LineageGraph,
    walk_lineage,
)
from gbserver.storage.lineage_job_storage import ILineageJobStorage
from gbserver.storage.lineage_row_storage import (
    TERMINAL_INPUT,
    TERMINAL_OUTPUT,
    ILineageRowStorage,
    row_tags,
    tag_strings,
)
from gbserver.storage.stored_lineage_row import TERMINAL

logger = logging.getLogger(__name__)

# The wire keeps W&B's direction words; the walk names directions for what they
# walk toward. The mapping is explicit rather than positional because the words are
# easy to invert by accident: the prototype's "downstream" walks toward origins,
# but the live W&B backend treats downstream as used_by() -- toward descendants --
# and the frontend sends it with that meaning. This follows the live backend.

# Ceiling on one page of the job listing. Generous, because the entries are small and
# the whole point of the listing is to make a 68,905-run artifact reachable -- but
# bounded, so a caller cannot ask for all of them in one response and recreate the
# problem the graph collapse exists to avoid.
_MAX_JOBS_PAGE = 1000

# Job ids per ``IN`` when reading whole jobs back: well under SQLite's bound-
# parameter limit, and the walk's frontier batches are no larger.
_JOB_ROWS_BATCH = 500
# Rows read back to group runs by whole-job signature, per graph. The walk bounds
# the jobs in a graph, not their rows: one job with many inputs is many rows. Past
# this, the remaining jobs group on the rows the walk read (see
# ``graph_builder._same_signature_jobs``), which only costs grouping precision.
_GROUP_ROWS_BUDGET = 50_000

_WIRE_DIRECTIONS = {
    "downstream": Direction.DESCENDANTS,
    "upstream": Direction.ANCESTORS,
    "both": Direction.BOTH,
}


class DBLineageService(LineageService):
    """Read artifact lineage from the local lineage index.

    Args:
        storage: the lineage row storage to read. Defaults to the process-wide
            admin storage, resolved lazily so importing this module does not
            require a configured database.
    """

    def __init__(
        self,
        storage: Optional[ILineageRowStorage] = None,
        job_storage: Optional[ILineageJobStorage] = None,
    ) -> None:
        self._storage = storage
        self._job_storage = job_storage

    @property
    def storage(self) -> ILineageRowStorage:
        """The lineage row storage, resolved on first use."""
        if self._storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            self._storage = get_admin_storage().lineage_row_storage
        return self._storage

    @property
    def job_storage(self) -> Optional[ILineageJobStorage]:
        """The lineage job table, resolved on first use, or ``None``.

        Only the tag search reads it: a tag is a property of an execution, and the
        rows carry a copy per edge, so the one place a tag query belongs is the job
        table. ``None`` rather than raising, because this is the *read* side -- a
        search with no table to read answers "no matches", which is what an
        unreachable index means to a caller, whereas raising would turn it into a
        500 on a request that is not wrong.
        """
        if self._job_storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            try:
                self._job_storage = get_admin_storage().lineage_job_storage
            except Exception:
                logger.warning("No lineage job storage available for tag search")
                return None
        return self._job_storage

    def query_graph(
        self,
        uri: Optional[str] = None,
        job_id: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 10,
        max_nodes_per_level: Optional[int] = None,
        group_runs: bool = True,
        job_ids: Optional[List[str]] = None,
        authorize_job: Optional[Callable[[Dict], bool]] = None,
    ) -> Dict:
        """Return a lineage graph seeded by an artifact, a job, or both.

        The general entry point: a caller asks however it holds the artifact, rather
        than the index dictating one lookup shape.

        - ``uri`` -- seeds from that artifact, in any spelling.
        - ``job_id`` -- seeds from every endpoint of that execution.
        - ``job_ids`` -- the same for each of several executions: the jobs behind a
          grouped run node, so expanding it walks every input, not just the first.
        - any combination -- seeds from the union, so a job's inputs and one
          specific artifact can be expanded together.

        One of the two is required. A graph needs somewhere to start, and "the most
        recent activity" is not a question anyone asks of a lineage graph -- it is an
        arbitrary slice of unrelated chains. The job listing answers "what ran
        lately" instead.

        Never returns ``None``: there is nothing to report as "unknown", because an
        artifact with no lineage recorded is a legitimate answer. An empty graph means "nothing recorded", which the caller
        must not render as an error.

        Args:
            uri: the artifact's URI, normalized here.
            job_id: the job execution to seed from.
            direction: ``downstream``, ``upstream`` or ``both``, in wire terms.
            max_depth: how many hops to expand beyond the seeds.
            max_nodes_per_level: raise the per-level frontier ceiling for this one
                request, for an explicit "show the full graph" action. ``None`` uses
                the server default.
            group_runs: fold the jobs with the same input and output into one node
                (the default); ``False`` returns one node per job. See
                :func:`~gbserver.lineage.graph_builder.build_graph_dict`.
            job_ids: more executions to seed from, as ``job_id``.
            authorize_job: whether the caller may read a job, given its listing
                entry -- the same check ``GET /lineage/jobs/{id}`` applies. A single
                run node whose job passes gets ``gb_build_id`` and
                ``gb_target_run_uuid``; ``None`` adds them to none. See
                :meth:`_add_authorized_origin_ids`.

        Returns:
            ``{root_id, nodes, edges, truncated, unexpanded}``. ``root_id`` is the
            resolved URI when exactly one artifact was named, else ``""``: a job-seeded
            query has several roots, and flagging one arbitrarily would
            misreport what was asked about.

        Raises:
            ValueError: if none of ``uri``, ``job_id`` or ``job_ids`` is given, or
                ``direction`` is not a wire direction. The API layer validates both
                first and answers 422.
        """
        job_seeds = [j for j in ([job_id] if job_id else []) + list(job_ids or []) if j]
        if not uri and not job_seeds:
            raise ValueError("Either uri, job_id or job_ids must be provided")
        walk_direction = _WIRE_DIRECTIONS.get(direction)
        if walk_direction is None:
            raise ValueError(
                f"direction must be one of {sorted(_WIRE_DIRECTIONS)}, got {direction!r}"
            )

        root_uri = normalize_uri(uri or "")
        seeds: set = set()
        if root_uri:
            seeds.add(root_uri)
        if job_seeds:
            seeds.update(self._job_endpoint_uris(list(dict.fromkeys(job_seeds))))

        if not seeds:
            # The caller named something this index cannot key on. An empty graph
            # rather than an error: "nothing matches" is a real answer, and the URI
            # drop is already logged by normalize_uri.
            return build_graph_dict(
                LineageGraph(), root_uri=root_uri, group_runs=group_runs
            )

        graph = walk_lineage(
            storage=self.storage,
            seeds=seeds,
            direction=walk_direction,
            max_depth=max_depth,
            max_nodes_per_level=max_nodes_per_level or DEFAULT_MAX_NODES_PER_LEVEL,
            # The grouped graph draws all of an artifact's self-loops as one node,
            # so it needs one row and a count, not every row.
            collapse_self_loops=group_runs,
        )
        # Only a single-artifact query has one root to flag; anything else has many.
        single_root = bool(root_uri) and not job_seeds
        result = build_graph_dict(
            graph,
            root_uri=root_uri,
            root_is_artifact=single_root,
            group_runs=group_runs,
            job_rows_of=self._rows_of_jobs_for_grouping,
        )
        if authorize_job is not None:
            self._add_authorized_origin_ids(result.get("nodes", []), authorize_job)
        return result

    def _add_authorized_origin_ids(
        self, nodes: List[Dict], authorize_job: Callable[[Dict], bool]
    ) -> None:
        """Add the build and target-run ids to each run node the caller may read.

        The graph is cross-space, so its nodes carry only the global fields (see
        ``db_responses.GLOBAL_JOB_FIELDS``). The origin's ids are behind the space
        check -- the one ``GET /lineage/jobs/{id}`` applies, passed in as
        ``authorize_job`` so the two cannot drift. A caller who could open the
        job's detail gets them on the node too; nobody else does. Grouped nodes
        (``runs:``) stand for many jobs and get neither.

        Batched reads of the run nodes' rows under ``_GROUP_ROWS_BUDGET`` (a job
        past it gets no ids); the check runs once per distinct owner and space, on
        all of each job's rows. A job with no namespace or owner (an import) fails
        closed.
        """
        by_job = {
            node["id"][len("run:") :]: node
            for node in nodes
            if str(node.get("id", "")).startswith("run:")
        }
        if not by_job:
            return
        rows_by_job: Dict[str, List] = {}
        # Every row, not the first: the entry takes owner and namespace only where
        # all rows agree, so a forged row under a real job id fails closed.
        for row in self._rows_of_jobs_for_grouping(sorted(by_job)):
            rows_by_job.setdefault(row.job_id, []).append(row)
        allowed: Dict[Tuple[str, str], bool] = {}
        for job_id, node in by_job.items():
            rows = rows_by_job.get(job_id)
            if not rows:
                continue
            entry = job_listing_entry(job_id, rows, [])
            key = (entry["owner"], entry["space_name"])
            if key not in allowed:
                try:
                    allowed[key] = bool(authorize_job(entry))
                except Exception:
                    logger.exception("Job access check failed for %s", job_id)
                    allowed[key] = False
            if not allowed[key]:
                continue
            metadata = node.setdefault("metadata", {})
            for key_name in ("build_id", "target_run_uuid"):
                value = origin_id(rows[0].attributes, key_name)
                if value:
                    metadata[f"gb_{key_name}"] = value

    def _rows_of_jobs_for_grouping(self, job_ids: List[str]) -> List:
        """:meth:`_rows_of_jobs`, stopping once ``_GROUP_ROWS_BUDGET`` rows are read.

        Each batch holds whole jobs, so a job is either read in full or not at all;
        a job not read groups on its walked rows.
        """
        rows: List = []
        for start in range(0, len(job_ids), _JOB_ROWS_BATCH):
            if len(rows) >= _GROUP_ROWS_BUDGET:
                logger.warning(
                    "Lineage graph grouping read %d rows; %d of %d jobs group on "
                    "their walked rows only",
                    len(rows),
                    len(job_ids) - start,
                    len(job_ids),
                )
                break
            rows.extend(
                self.storage.get_rows_by_jobs(job_ids[start : start + _JOB_ROWS_BATCH])
            )
        return rows

    def _rows_of_jobs(self, job_ids: List[str]) -> List:
        """Every stored row of ``job_ids``, batched so one ``IN`` stays bounded."""
        rows: List = []
        for start in range(0, len(job_ids), _JOB_ROWS_BATCH):
            rows.extend(
                self.storage.get_rows_by_jobs(job_ids[start : start + _JOB_ROWS_BATCH])
            )
        return rows

    def list_jobs(
        self,
        uri: Optional[str] = None,
        job_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
        job_ids: Optional[List[str]] = None,
    ) -> Dict:
        """List job executions matching every given filter, paged.

        The one listing over the index. Filters AND together, and ``uri``,
        ``job_id`` or ``job_ids`` is required (with neither, the listing is empty):

        - ``uri`` -- jobs that consumed **or** produced the artifact, in any
          spelling. The drill-down for a graph node's ``run_count``:
          ``build_graph_dict`` folds an artifact's in-place rewrites into one node,
          because one run node per append produced a 55 MB response for a single
          dataset, and a count with no way to expand it would be a dead end.
        - ``self_loop`` (with ``uri``) -- only the jobs whose input and output are
          both that artifact: exactly the runs behind its looped node in the graph.
        - ``output`` (with ``uri``) -- only the jobs with a ``uri -> output`` row:
          the runs behind a node grouping jobs with the same inputs and outputs.
        - ``terminal`` (with ``uri``) -- ``"input"`` lists the jobs with a
          ``(nothing) -> uri`` row, ``"output"`` those with ``uri -> (nothing)``: the
          runs behind a grouped node with no recorded input or output.
        - ``job_id`` -- that one execution.
        - ``job_ids`` -- those executions: the runs behind a graph node grouping
          jobs with the same inputs and outputs (its ``jobs_query``).

        Paged rather than capped, unlike the graph: a flat list has no shape to
        preserve, so a caller can walk the whole thing.

        The narrowest filter is resolved first. A lone ``uri`` is paged and counted
        in SQL, because its artifact may have tens of thousands of jobs; combined
        with another filter it only checks that filter's (small) job set.

        Returns:
            ``{jobs, total, limit, offset}``. ``total`` is the unpaged count of
            distinct jobs -- never rows, which overstate a many-input job. Filtered jobs
            are ordered by ``job_id`` so paging is stable.

        Raises:
            ValueError: ``limit`` outside ``1..1000``, ``offset`` negative, or an
                unknown ``terminal``. The route validates these first (422).
            Exception: a storage error propagates, so the route answers 503 rather
                than an empty page that reads as "no jobs".
        """
        if not 1 <= int(limit) <= _MAX_JOBS_PAGE:
            raise ValueError(f"limit must be between 1 and {_MAX_JOBS_PAGE}")
        if int(offset) < 0:
            raise ValueError("offset must not be negative")
        if terminal not in (None, TERMINAL_INPUT, TERMINAL_OUTPUT):
            raise ValueError(f"unknown terminal {terminal!r}")
        empty = {"jobs": [], "total": 0, "limit": limit, "offset": offset}

        # None means "not constrained yet", which is not the same as empty.
        candidates: Optional[Set[str]] = None
        if job_id:
            candidates = {job_id} if self.storage.get_rows_by_job(job_id) else set()
        if job_ids:
            wanted = set(job_ids) if candidates is None else candidates & set(job_ids)
            recorded = {row.job_id for row in self._rows_of_jobs(sorted(wanted))}
            candidates = wanted & recorded

        if uri:
            normalized = normalize_uri(uri)
            if not normalized:
                return empty
            target = normalize_uri(output) if output else None
            if output and not target:
                return empty
            if candidates is None:
                total = self.storage.count_jobs_touching(
                    normalized, self_loop, output=target, terminal=terminal
                )
                page = self.storage.get_job_ids_touching(
                    normalized,
                    limit,
                    offset,
                    self_loop,
                    output=target,
                    terminal=terminal,
                )
                return {**empty, "jobs": self._job_entries(page), "total": total}
            candidates = self.storage.filter_jobs_touching(
                normalized, candidates, self_loop, output=target, terminal=terminal
            )

        if candidates is None:
            return empty
        ordered = sorted(candidates)
        page, total = ordered[offset : offset + limit], len(ordered)
        return {**empty, "jobs": self._job_entries(page), "total": total}

    def get_job_detail(
        self,
        job_id: str,
        authorize_build: Callable[[Any], None],
        admin_storage: Any = None,
        authorize_entry: Optional[Callable[[Dict], bool]] = None,
    ) -> Optional[Dict]:
        """One job's listing entry plus its full content, or ``None`` if unknown.

        The listing reads the index alone; this reads a granite.build job's build
        and target run from this server -- see :mod:`gbserver.lineage.job_detail`. One job at a
        time on purpose: the side panel asks for it when a run is opened, never for
        a whole page.
        """
        rows = self.storage.get_rows_by_job(job_id) if job_id else []
        if not rows:
            return None
        tags = sorted({tag for row in rows for tag in tag_strings(row_tags(row))})
        if admin_storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            admin_storage = get_admin_storage()
        return fetch_job_detail(
            job_listing_entry(job_id, rows, tags),
            rows,
            admin_storage,
            authorize_build,
            authorize_entry=authorize_entry,
        )

    def _job_entries(self, job_ids: List[str]) -> List[Dict]:
        """The listing entries for one page, in the page's order.

        One batched query on ``job_id`` whatever the page size. Everything comes
        from the index rows: the listing reads ``gb_lineage_index`` alone, so the
        job table's payloads (step configs, stats) are not part of it.
        """
        if not job_ids:
            return []
        rows_by_job: Dict[str, List] = {}
        for row in self.storage.get_rows_by_jobs(job_ids):
            rows_by_job.setdefault(row.job_id, []).append(row)
        # Cross-space listing: only the global fields (see GLOBAL_JOB_FIELDS), which
        # exclude tags, so they are not read.
        return [
            global_job_entry(job_listing_entry(job_id, rows_by_job.get(job_id, []), []))
            for job_id in job_ids
        ]

    def _job_endpoint_uris(self, job_ids: List[str]) -> set:
        """Every endpoint of the given job executions, as seeds.

        Batched indexed queries on ``job_id`` (see :meth:`_rows_of_jobs`), which is
        the only identifier every lineage source has -- so this works for imported
        rows too, where no process id does.

        A storage error propagates (the route answers 503): swallowing it would
        return an empty graph, indistinguishable from "this job recorded nothing".
        """
        seeds: set = set()
        for row in self._rows_of_jobs(job_ids):
            for endpoint in (row.input, row.output):
                if endpoint and endpoint != TERMINAL:
                    seeds.add(endpoint)
        return seeds

    # -- Write-path methods. This service reads an index that another writer
    # populates, so none of these apply; each returns the value that makes a caller
    # behave correctly rather than one that merely avoids an exception.

    def emit_event(self, event: Dict) -> None:
        """Ignore an emitted event.

        Rows are written by the lineage sink from build state, not by callers
        pushing events here. Raising would break a recorder that fans out to every
        configured provider.
        """
        return None

    def search_lineage_by_tags(
        self, tags: List[str], limit: int = 10, offset: int = 0
    ) -> Tuple[int, List[Dict]]:
        """Jobs carrying any of ``tags``, as OpenLineage events, and the total.

        ``tags`` is a disjunction -- the same ``$in`` the W&B provider applies --
        and an empty list matches every job, which is what ``POST /lineage/search``
        means by one. Answered from ``gb_lineage_job``'s promoted ``tags`` column,
        so it is a SQL filter rather than a scan of a JSON blob.

        **The owner and space must reach the facets.** The route filters access by
        reading ``run.facets.job_details.owner`` and ``run.facets.tags.space_name``
        off each result, not by querying the database, so a result that omits them
        is dropped for every caller -- which looks exactly like an empty index
        rather than like a bug. They are columns on the record, so they are carried
        verbatim.
        """
        storage = self.job_storage
        if storage is None:
            # Not "no matches": the route turns this into a 503, as for any other
            # storage failure, instead of an empty 200.
            raise RuntimeError("No lineage job storage available for tag search")
        # A storage error propagates: an empty result would be indistinguishable
        # from "no job has these tags", and dmg-ng reads this route.
        total, jobs = storage.search_by_tags(tags, limit=limit, offset=offset)
        rows_by_job: Dict[str, List] = {}
        for row in self._rows_of_jobs([job.job_id for job in jobs]):
            rows_by_job.setdefault(row.job_id, []).append(row)
        return total, [
            job_search_event(job, rows_by_job.get(job.job_id, [])) for job in jobs
        ]

    def count_events_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        """How many jobs match; one event per job here, so the same count."""
        return self.count_runs_by_tags(tags, required_tags)

    def count_runs_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        """How many jobs carry any of ``tags`` and all of ``required_tags``.

        A missing job table or a storage error raises, as for
        :meth:`search_lineage_by_tags`: a 0 would read as "no such job" and let a
        caller re-record or report nothing on what is really an outage.
        """
        storage = self.job_storage
        if storage is None:
            raise RuntimeError("No lineage job storage available for tag count")
        total, _ = storage.search_by_tags(tags, all_of=required_tags, limit=0, offset=0)
        return total

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        """Report every candidate as unrecorded.

        The interface documents that implementations must fail toward
        re-recording, because recording is idempotent and this is only an
        efficiency filter. Returning ``target_ids`` unchanged is that safe answer.

        It is deliberately not answered from the index here: this service is the
        *read* side, and the sink that writes rows does its own presence-based
        dedup against the same table (``get_recorded_target_runs``). Answering here
        too would put the same decision in two places, keyed differently -- target
        id here versus target run uuid there -- and the two would disagree.
        """
        return target_ids
