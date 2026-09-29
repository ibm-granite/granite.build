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
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from gbserver.lineage.graph_builder import build_graph_dict
from gbserver.lineage.openlineage_service import LineageService
from gbserver.lineage.attributes import (
    INPUT,
    OUTPUT,
    endpoint_kind,
    job_detail,
    origin_system,
)
from gbserver.lineage.uri_normalize import normalize_uri
from gbserver.lineage.walk import (
    DEFAULT_MAX_NODES_PER_LEVEL,
    Direction,
    LineageGraph,
    walk_lineage,
)
from gbserver.storage.lineage_job_storage import ILineageJobStorage
from gbserver.storage.lineage_job_tag_storage import ILineageJobTagStorage
from gbserver.storage.lineage_row_storage import ILineageRowStorage
from gbserver.storage.stored_lineage_row import TERMINAL

logger = logging.getLogger(__name__)

# The wire keeps W&B's direction words; the walk names directions for what they
# walk toward. The mapping is explicit rather than positional because the words are
# easy to invert by accident: the prototype's "downstream" walks toward origins,
# but the live W&B backend treats downstream as used_by() -- toward descendants --
# and the frontend sends it with that meaning. This follows the live backend.
# How many of the newest rows seed an unfiltered query. A graph of "everything" is
# neither useful nor bounded, so the newest activity stands in for it.
_RECENT_ACTIVITY_ROWS = 50

# Ceiling on one page of the run listing. Generous, because the rows are small and the
# whole point of the endpoint is to make a 68,905-run artifact reachable -- but bounded,
# so a caller cannot ask for all of them in one response and recreate the problem the
# graph collapse exists to avoid.
_MAX_RUNS_PAGE = 1000

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
    ) -> Dict:
        """Return a lineage graph for any combination of optional filters.

        The general entry point: a caller asks however it holds the artifact, rather
        than the index dictating one lookup shape.

        - ``uri`` -- seeds from that artifact, in any spelling.
        - ``job_id`` -- seeds from every endpoint of that execution.
        - both -- seeds from the union, so a job's inputs and one specific artifact
          can be expanded together.
        - neither -- seeds from the most recent lineage activity, capped.

        Never returns ``None``: unlike :meth:`get_artifact_graph` there is nothing to
        report as "unknown", because a query with no filters is a legitimate request
        and an empty index is a legitimate answer. An empty graph means "nothing
        recorded", which the caller must not render as an error.

        Args:
            uri: the artifact's URI, normalized here.
            job_id: the job execution to seed from.
            direction: ``downstream``, ``upstream`` or ``both``, in wire terms.
            max_depth: how many hops to expand beyond the seeds.
            max_nodes_per_level: raise the per-level frontier ceiling for this one
                request, for an explicit "show the full graph" action. ``None`` uses
                the server default.

        Returns:
            ``{root_id, nodes, edges, truncated, unexpanded}``. ``root_id`` is the
            resolved URI when exactly one artifact was named, else ``""``: a job-seeded or
            unfiltered query has several roots, and flagging one arbitrarily would
            misreport what was asked about.

        Raises:
            ValueError: if ``direction`` is not a wire direction. The API layer maps
                this to a 400.
        """
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
            if uri or job_id:
                # The caller named something this index cannot key on. An empty graph
                # rather than an error: "nothing matches" is a real answer, and the
                # URI drop is already logged by normalize_uri.
                return build_graph_dict(LineageGraph(), root_uri=root_uri)
            seeds = self._recent_activity_seeds()

        graph = walk_lineage(
            storage=self.storage,
            seeds=seeds,
            direction=walk_direction,
            max_depth=max_depth,
            max_nodes_per_level=max_nodes_per_level or DEFAULT_MAX_NODES_PER_LEVEL,
        )
        # Only a single-artifact query has one root to flag; anything else has many.
        single_root = bool(root_uri) and not job_id
        return build_graph_dict(graph, root_uri=root_uri, root_is_artifact=single_root)

    def list_jobs_by_tags(
        self,
        tags: List[str],
        required_tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict:
        """List the jobs matching a tag filter, paged -- W&B's run-tag filter.

        A job matches when it carries **any** of ``tags`` and **all** of
        ``required_tags``, the same ``$in`` + required shape as
        ``count_runs_by_tags``. Tags are free-form (``build_id=...``,
        ``space_name=...``, a user's own); matching is exact, never a substring.

        One indexed ``IN`` on the tag table resolves the job ids; the page's job
        records and tags are then fetched by ``job_id`` in one query each. Ordered
        by ``job_id`` so paging is stable.

        Returns:
            ``{jobs, total, limit, offset}``. Empty for an empty filter: an
            unfiltered request is not a tag query.
        """
        limit = max(1, min(int(limit), _MAX_RUNS_PAGE))
        offset = max(0, int(offset))
        empty = {"jobs": [], "total": 0, "limit": limit, "offset": offset}

        tag_storage = self._resolved("_tag_storage", "lineage_job_tag_storage")
        job_storage = self._resolved("_job_storage", "lineage_job_storage")
        if tag_storage is None or job_storage is None:
            return empty
        try:
            job_ids = sorted(
                tag_storage.get_job_ids_by_tags(tags, all_of=required_tags)
            )
            page = job_ids[offset : offset + limit]
            jobs = job_storage.get_jobs_by_id(page)
            tags_by_job = tag_storage.get_tags(page)
        except Exception:
            logger.exception("Lineage job tag query failed")
            return empty

        entries = []
        for job_id in page:
            job = jobs.get(job_id)
            entries.append(
                {
                    "job_id": job_id,
                    "job_namespace": job.job_namespace if job else "",
                    "space_name": job.space_name if job else "",
                    "owner": job.owner if job else "",
                    "source_system": job.source_system if job else "",
                    "status": job.status if job else "",
                    "started_at": job.started_at if job else "",
                    "tags": tags_by_job.get(job_id, []),
                }
            )
        return {**empty, "jobs": entries, "total": len(job_ids)}

    def _resolved(self, attr: str, admin_attr: str):
        """A storage given at construction, else the admin one, else ``None``.

        Only the row storage is required by this service; the job and tag
        storages serve the tag listing alone, so a missing one degrades that
        listing to empty rather than failing construction.
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

    def list_runs(
        self,
        uri: Optional[str] = None,
        job_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict:
        """List the job executions touching an artifact, paged.

        The drill-down the graph deliberately does not carry. ``build_graph_dict``
        collapses an artifact's in-place rewrites into one node with a ``run_count``,
        because rendering 68,905 of them produced a 55 MB response describing a single
        dataset -- and a count with no way to expand it would just be a dead end. This
        is that way.

        Paged rather than capped: unlike the graph, a flat list has no shape to
        preserve, so there is no reason to truncate it instead of letting a caller walk
        it.

        Args:
            uri: the artifact whose runs to list, in any spelling; normalized here.
                Matches a run that consumed it OR produced it.
            job_id: list the rows of one execution instead.
            limit: page size, capped at :data:`_MAX_RUNS_PAGE`.
            offset: rows to skip.

        Returns:
            ``{runs, total, limit, offset}``. ``runs`` carries one entry per row, each
            with its job id, endpoints and job detail.

            ``total`` is exact, from three indexed SQL counts: rows with this artifact
            as source, plus as target, minus the self-loops that are both (without that
            third term an in-place-rewritten artifact reports double).
        """
        limit = max(1, min(int(limit), _MAX_RUNS_PAGE))
        offset = max(0, int(offset))

        if job_id:
            rows = self._safe(lambda: self.storage.get_rows_by_job(job_id))
            return {
                "runs": [_run_entry(row) for row in rows[offset : offset + limit]],
                "total": len(rows),
                "limit": limit,
                "offset": offset,
            }

        normalized = normalize_uri(uri or "")
        if not normalized:
            return {"runs": [], "total": 0, "limit": limit, "offset": offset}

        # Both directions, because "the runs touching this artifact" means the ones
        # that consumed it and the ones that produced it.
        wheres = ({"input": normalized}, {"output": normalized})

        # The total comes from SQL COUNT, not from walking the rows. An artifact with
        # 68,905 runs is exactly the case this endpoint exists for, and counting it in
        # Python cost ~3s per request whatever the page size -- recreating in the read
        # path the expense the graph collapse removed.
        #
        # Three counts, not two: a self-loop row matches BOTH the source and the target
        # query, so summing them double-counts every in-place rewrite. On the real hub
        # that reported 137,811 runs for an artifact with 68,906 -- a number wrong by
        # 2x, which is worse than slow. Subtracting the overlap makes it exact and still
        # costs only one more indexed COUNT.
        total = (
            self._safe_count({"input": normalized})
            + self._safe_count({"output": normalized})
            - self._safe_count({"input": normalized, "output": normalized})
        )

        # Streamed, and stopped as soon as the window is filled: pages are pulled only
        # until the requested slice exists, so an early offset costs an early exit.
        seen: set = set()
        page: List = []
        position = 0
        for where in wheres:
            if len(page) >= limit:
                break
            for chunk in self._safe_pages(where):
                for row in chunk:
                    key = (row.job_id, row.input, row.output)
                    if key in seen:
                        continue
                    seen.add(key)
                    if position >= offset:
                        page.append(row)
                    position += 1
                    if len(page) >= limit:
                        break
                if len(page) >= limit:
                    break

        return {
            "runs": [_run_entry(row) for row in page],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def _safe_count(self, where: Dict) -> int:
        """Count matching rows in SQL, reporting a failure as zero."""
        try:
            return int(self.storage.count(where))
        except Exception:
            logger.exception("Lineage run count query failed")
            return 0

    def _safe_pages(self, where: Dict) -> Iterator[List]:
        """Yield pages for a where clause, reporting a failure as "no more".

        A failed read must not turn a listing into a 500; the caller sees a smaller
        total and the exception is logged.
        """
        try:
            yield from self.storage.get_paged(where)
        except Exception:
            logger.exception("Lineage run listing query failed")

    @staticmethod
    def _safe(fetch) -> List:
        """Run a storage read, reporting a failure as "nothing" rather than raising.

        One failed page must not turn a listing into a 500; the caller sees a smaller
        total, and the exception is logged.
        """
        try:
            return list(fetch())
        except Exception:
            logger.exception("Lineage run listing query failed")
            return []

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

    def _recent_activity_seeds(self) -> set:
        """Seeds for an unfiltered query: the most recently recorded endpoints.

        Deliberately capped and taken from the first page only. "Everything" is not a
        useful answer for a graph and would be an unbounded walk; the newest rows are
        what an operator opening an empty view actually wants to see.
        """
        try:
            for page in self.storage.get_paged():
                seeds: set = set()
                for row in page[:_RECENT_ACTIVITY_ROWS]:
                    for endpoint in (row.input, row.output):
                        if endpoint and endpoint != TERMINAL:
                            seeds.add(endpoint)
                return seeds
        except Exception:
            logger.exception("Could not read recent lineage activity")
        return set()

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


def _run_entry(row) -> Dict:
    """One row of a run listing.

    Flat and endpoint-first: a caller reaching here already has the artifact and wants
    to know which executions touched it, so the job detail matters more than the graph
    shape. Terminals are reported as empty strings, exactly as stored.
    """
    return {
        "job_id": row.job_id,
        "source": row.input,
        "target": row.output,
        "is_self_loop": row.is_self_loop(),
        "job": job_detail(row.attributes),
        "source_system": origin_system(row.attributes),
    }
