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

``POST /lineage/artifact`` delegates to a :class:`LineageService` and returns 404
when that service returns ``None`` -- which is what ``NoopLineageService`` does for
everything, so standalone shows "Lineage is not available" for every artifact
today. This implementation answers the same question from the lineage table, so a
deployment with no W&B still has lineage.

Only the graph is implemented here. The event-emission and tag-search methods of
the interface belong to the W&B-shaped write path (this service reads an index
another writer populates), so they are explicit no-ops rather than pretending to
succeed -- see each method for why its degenerate value is the safe one.

Root resolution is the part worth reading. The request identifies an artifact the
way a UI can -- a name, or a URL -- while the index is keyed by canonical
identifier. Rather than reconstruct an identifier from a name (which cannot be done:
the identifier needs a namespace and a type the request does not carry), the lookup
goes the other way and matches against what rows already store. That makes lineage
imported from a source with no uuid reachable, which was the point of keying the
index by identifier in the first place.
"""

import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

from gbserver.lineage.attributes import (
    INPUT,
    OUTPUT,
    PAYLOAD_INPUT_PARAMS,
    endpoint_kind,
    job_detail,
    origin_detail,
    origin_system,
    payload_detail,
    run_detail,
)
from gbserver.lineage.graph_builder import build_graph_dict
from gbserver.lineage.openlineage_service import LineageService
from gbserver.lineage.uri_normalize import normalize_uri
from gbserver.lineage.walk import (
    DEFAULT_MAX_NODES_PER_LEVEL,
    Direction,
    LineageGraph,
    walk_lineage,
)
from gbserver.storage.lineage_job_storage import ILineageJobStorage
from gbserver.storage.lineage_job_tag_storage import ILineageJobTagStorage
from gbserver.storage.lineage_row_storage import (
    TERMINAL_INPUT,
    TERMINAL_OUTPUT,
    ILineageRowStorage,
)
from gbserver.storage.stored_lineage_row import TERMINAL
from gbserver.utils.redaction import redact_sensitive

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
        job_storage: the lineage job storage, resolved the same way.
        tag_storage: the lineage job tag storage, resolved the same way.
    """

    def __init__(
        self,
        storage: Optional[ILineageRowStorage] = None,
        job_storage: Optional[ILineageJobStorage] = None,
        tag_storage: Optional[ILineageJobTagStorage] = None,
    ) -> None:
        self._storage = storage
        self._job_storage = job_storage
        self._tag_storage = tag_storage

    @property
    def storage(self) -> ILineageRowStorage:
        """The lineage row storage, resolved on first use."""
        if self._storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            self._storage = get_admin_storage().lineage_row_storage
        return self._storage

    def get_artifact_graph(
        self,
        artifact_name: Optional[str] = None,
        artifact_url: Optional[str] = None,
        artifact_type: Optional[str] = None,
        max_depth: int = 10,
        direction: str = "downstream",
    ) -> Optional[Dict]:
        """Return the lineage graph for one artifact, or ``None`` if unknown.

        The root resolves in **one indexed lookup**. This used to be a paged full
        scan of the table on every request, and not by oversight: the indexed
        columns held canonical identifiers while a request carries a URL, so there
        was nothing to match on. With the URI as the identity, normalizing the
        request's URL produces exactly the value those columns hold.

        Args:
            artifact_name: the artifact's name. Only usable as a URI -- see below.
            artifact_url: the artifact's URI, in any spelling; it is normalized
                here, so a browser URL and the runtime's own URI resolve alike.
            artifact_type: when given, the resolved root must have this type, and a
                mismatch is an error rather than a miss -- the caller asserted
                something about the artifact that turned out to be false.
            max_depth: how many hops to expand.
            direction: ``downstream``, ``upstream`` or ``both``, in wire terms.

        Returns:
            ``{root_id, nodes, edges, truncated, unexpanded}``, or ``None`` when the
            request names nothing this index can key on. ``None`` becomes the 404 the
            frontend shows as "not available", so it must mean "unknown here", never
            "no lineage": an artifact that normalizes fine but has no edges yet
            returns a graph with just its own node.

        Raises:
            ValueError: if ``direction`` is not a wire direction, or if
                ``artifact_type`` contradicts the resolved root. The API layer maps
                this to a 400.
        """
        walk_direction = _WIRE_DIRECTIONS.get(direction)
        if walk_direction is None:
            raise ValueError(
                f"direction must be one of {sorted(_WIRE_DIRECTIONS)}, got {direction!r}"
            )

        # A name is not an identity. The index keys on URIs, and a bare name has no
        # scheme, so it can only be resolved if it already *is* one -- which is why
        # the URL is tried first and a name is only a fallback for a caller that
        # passed a URI in the name field. Guessing a scheme for a bare name would
        # invent an artifact that may not exist.
        root_uri = normalize_uri(artifact_url or "") or normalize_uri(
            artifact_name or ""
        )
        if not root_uri:
            return None

        graph = walk_lineage(
            storage=self.storage,
            seeds=[root_uri],
            direction=walk_direction,
            max_depth=max_depth,
            max_nodes_per_level=DEFAULT_MAX_NODES_PER_LEVEL,
        )

        if artifact_type:
            root_kind = self._kind_of(graph, root_uri)
            if root_kind and artifact_type != root_kind:
                raise ValueError(
                    f"Artifact type mismatch: expected {artifact_type!r}, but "
                    f"{root_uri!r} has type {root_kind!r}"
                )

        return build_graph_dict(graph, root_uri=root_uri)

    @staticmethod
    def _kind_of(graph, uri: str) -> str:
        """The artifact type recorded for ``uri`` in a walked graph, if any.

        A URI has one type by decision, so the first row mentioning it settles the
        answer and there is nothing to reconcile.
        """
        for row in graph.rows:
            if row.input == uri:
                kind = endpoint_kind(row.attributes, INPUT)
                if kind:
                    return kind
            if row.output == uri:
                kind = endpoint_kind(row.attributes, OUTPUT)
                if kind:
                    return kind
        return ""

    def query_graph(
        self,
        uri: Optional[str] = None,
        job_id: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 10,
        max_nodes_per_level: Optional[int] = None,
        group_runs: bool = True,
    ) -> Dict:
        """Return a lineage graph seeded by an artifact, a job, or both.

        The general entry point: a caller asks however it holds the artifact, rather
        than the index dictating one lookup shape.

        - ``uri`` -- seeds from that artifact, in any spelling.
        - ``job_id`` -- seeds from every endpoint of that execution.
        - both -- seeds from the union, so a job's inputs and one specific artifact
          can be expanded together.

        One of the two is required. A graph needs somewhere to start, and "the most
        recent activity" is not a question anyone asks of a lineage graph -- it is an
        arbitrary slice of unrelated chains. The job listing answers "what ran
        lately" instead.

        Never returns ``None``: unlike :meth:`get_artifact_graph` there is nothing to
        report as "unknown", because an artifact with no lineage recorded is a
        legitimate answer. An empty graph means "nothing recorded", which the caller
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

        Returns:
            ``{root_id, nodes, edges, truncated, unexpanded}``. ``root_id`` is the
            resolved URI when exactly one artifact was named, else ``""``: a job-seeded
            query has several roots, and flagging one arbitrarily would
            misreport what was asked about.

        Raises:
            ValueError: if neither ``uri`` nor ``job_id`` is given, or ``direction``
                is not a wire direction. The API layer maps this to a 400.
        """
        if not uri and not job_id:
            raise ValueError("Either uri or job_id must be provided")
        walk_direction = _WIRE_DIRECTIONS.get(direction)
        if walk_direction is None:
            raise ValueError(
                f"direction must be one of {sorted(_WIRE_DIRECTIONS)}, got {direction!r}"
            )

        root_uri = normalize_uri(uri or "")
        seeds: set = set()
        if root_uri:
            seeds.add(root_uri)
        if job_id:
            seeds.update(self._job_endpoint_uris(job_id))

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
        )
        # Only a single-artifact query has one root to flag; anything else has many.
        single_root = bool(root_uri) and not job_id
        return build_graph_dict(
            graph,
            root_uri=root_uri,
            root_is_artifact=single_root,
            group_runs=group_runs,
        )

    def list_jobs(
        self,
        uri: Optional[str] = None,
        job_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        required_tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> Dict:
        """List job executions matching every given filter, paged.

        The one listing over the index. Filters AND together, so
        ``uri=X&tags=build_id=Y`` is "the jobs that touched X within build Y":

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
        - ``tags`` / ``required_tags`` -- W&B's run-tag filter: **any** of ``tags``
          and **all** of ``required_tags``, matched exactly.
        - none -- the most recently recorded jobs.

        Paged rather than capped, unlike the graph: a flat list has no shape to
        preserve, so a caller can walk the whole thing.

        The narrowest filter is resolved first. A lone ``uri`` is paged and counted
        in SQL, because its artifact may have tens of thousands of jobs; combined
        with another filter it only checks that filter's (small) job set.

        Returns:
            ``{jobs, total, limit, offset}``. ``total`` is the unpaged count of
            distinct jobs -- never rows, which overstate a many-input job. Filtered jobs
            are ordered by ``job_id`` so paging is stable; unfiltered ones newest
            first. A failed read degrades to empty rather than raising.
        """
        limit = max(1, min(int(limit), _MAX_JOBS_PAGE))
        offset = max(0, int(offset))
        empty = {"jobs": [], "total": 0, "limit": limit, "offset": offset}

        try:
            # None means "not constrained yet", which is not the same as empty.
            candidates: Optional[Set[str]] = None
            if job_id:
                candidates = {job_id} if self.storage.get_rows_by_job(job_id) else set()
            if tags or required_tags:
                tag_storage = self._resolved("_tag_storage", "lineage_job_tag_storage")
                if tag_storage is None:
                    return empty
                tagged = tag_storage.get_job_ids_by_tags(
                    tags or [], all_of=required_tags
                )
                candidates = tagged if candidates is None else candidates & tagged

            if uri:
                normalized = normalize_uri(uri)
                if not normalized:
                    return empty
                target = normalize_uri(output) if output else None
                if output and not target:
                    return empty
                if terminal not in (None, TERMINAL_INPUT, TERMINAL_OUTPUT):
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
                page, total = self._recent_job_ids(limit, offset)
            else:
                ordered = sorted(candidates)
                page, total = ordered[offset : offset + limit], len(ordered)
            return {**empty, "jobs": self._job_entries(page), "total": total}
        except Exception:
            logger.exception("Lineage job listing failed")
            return empty

    def _recent_job_ids(self, limit: int, offset: int) -> Tuple[List[str], int]:
        """One page of the newest job records, and how many there are.

        Streamed and stopped once the window is filled, so an early page costs an
        early exit rather than a read of the whole table.
        """
        job_storage = self._resolved("_job_storage", "lineage_job_storage")
        if job_storage is None:
            return [], 0
        page: List[str] = []
        position = 0
        for chunk in job_storage.get_paged():
            for job in chunk:
                if position >= offset:
                    page.append(job.job_id)
                position += 1
                if len(page) >= limit:
                    return page, int(job_storage.count())
        return page, int(job_storage.count())

    def _job_entries(self, job_ids: List[str]) -> List[Dict]:
        """The listing entries for one page, in the page's order.

        Three batched queries whatever the page size: job records, tags and rows.
        A job with rows but no record -- rows can precede or outlive it -- still
        lists, its detail read from the rows instead.
        """
        if not job_ids:
            return []
        job_storage = self._resolved("_job_storage", "lineage_job_storage")
        tag_storage = self._resolved("_tag_storage", "lineage_job_tag_storage")
        jobs = job_storage.get_jobs_by_id(job_ids) if job_storage else {}
        tags_by_job = tag_storage.get_tags(job_ids) if tag_storage else {}
        rows_by_job: Dict[str, List] = {}
        for row in self.storage.get_rows_by_jobs(job_ids):
            rows_by_job.setdefault(row.job_id, []).append(row)
        return [
            _job_entry(
                job_id,
                jobs.get(job_id),
                rows_by_job.get(job_id, []),
                tags_by_job.get(job_id, []),
            )
            for job_id in job_ids
        ]

    def _resolved(self, attr: str, admin_attr: str):
        """A storage given at construction, else the admin one, else ``None``.

        Only the row storage is required by this service; the job and tag
        storages serve the job listing alone, so a missing one degrades that
        listing -- empty for a tag filter, detail-less entries otherwise --
        rather than failing construction.
        """
        if getattr(self, attr) is None:
            if self._storage is not None:
                return None
            from gbserver.storage.singleton_storage import get_admin_storage

            try:
                setattr(self, attr, getattr(get_admin_storage(), admin_attr))
            except Exception as exc:
                logger.debug("No %s available: %s", admin_attr, exc)
                return None
        return getattr(self, attr)

    def _job_endpoint_uris(self, job_id: str) -> set:
        """Every endpoint of one job execution, as seeds.

        One indexed query on ``job_id``, which is the only identifier every lineage
        source has -- so this works for imported rows too, where no process id does.
        """
        try:
            rows = self.storage.get_rows_by_job(job_id)
        except Exception:
            logger.exception("Could not read rows for job %s", job_id)
            return set()
        seeds: set = set()
        for row in rows:
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
        """Return no results: the index stores no run tags to search by."""
        return 0, []

    def count_events_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        """Return 0: see :meth:`search_lineage_by_tags`."""
        return 0

    def count_runs_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        """Return 0: see :meth:`search_lineage_by_tags`."""
        return 0

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


def _job_entry(job_id: str, job, rows: List, tags: List[str]) -> Dict:
    """One entry of the job listing.

    Job-first: a caller reaching here wants to know which executions matched, and
    what each read and wrote. The endpoints come from the job's own rows, terminals
    left out, so a self-rewrite shows its artifact on both sides.
    """
    attributes = rows[0].attributes if rows else {}
    record = job.attributes if job else {}
    payload = dict(payload_detail(record))
    # The step configs. Redacted unconditionally: this listing is readable by any
    # space member (see payload_detail).
    input_params = redact_sensitive(payload.pop(PAYLOAD_INPUT_PARAMS, None) or {})
    return {
        "job_id": job_id,
        "job_namespace": job.job_namespace if job else "",
        "space_name": job.space_name if job else "",
        "owner": job.owner if job else "",
        "source_system": (job.source_system if job else "")
        or origin_system(attributes),
        "status": job.status if job else "",
        "started_at": job.started_at if job else "",
        "tags": tags,
        "inputs": sorted({r.input for r in rows if r.input and r.input != TERMINAL}),
        "outputs": sorted(
            {r.output for r in rows if r.output and r.output != TERMINAL}
        ),
        # The record's own job group first: it is the one written per execution,
        # and rows imported without a record still have theirs.
        "job": job_detail(record) or job_detail(attributes),
        # Whether the execution has its own record in the job table. Rows can be
        # imported without one, and then everything above that reads ``job`` is empty.
        "job_recorded": job is not None,
        "job_input_params": input_params,
        # The rest of the record, so nothing it holds is lost on the way out.
        "payload": payload,
        "run": run_detail(record),
        "origin": origin_detail(record) or origin_detail(attributes),
    }
