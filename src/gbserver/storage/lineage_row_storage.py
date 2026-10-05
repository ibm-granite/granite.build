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

"""Base storage interface and implementation for lineage rows.

The queries here are the ones the traversal and the sink need: one batched hop in
each direction, seeding by build, and a presence check for dedup. Each is a single
indexed query, so a graph walk costs one query per level rather than a scan.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow
from gbserver.types.constants import GB_LINEAGE_TABLE_NAME

TERMINAL_INPUT = "input"
TERMINAL_OUTPUT = "output"

DOWNSTREAM = "downstream"
UPSTREAM = "upstream"


@dataclass(frozen=True)
class GroupedEdge:
    """Every job between one ``input`` and one ``output``, folded into a single edge.

    This is what the graph draws as a stacked edge; the jobs behind it are listed
    on demand from the rows themselves.
    """

    input: str
    output: str
    job_count: int
    last_recorded_at: str
    sample_job_id: str


def endpoint_pair(
    uri: str,
    self_loop: bool = False,
    output: Optional[str] = None,
    terminal: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """The exact ``(input, output)`` row a job listing filters on, or ``None`` for "either side".

    ``self_loop`` is ``uri -> uri``; ``output`` is ``uri -> output``; ``terminal`` names
    the side of ``uri``'s row that is empty: ``"input"`` is ``TERMINAL -> uri`` (jobs
    that wrote ``uri`` from nothing recorded), ``"output"`` is ``uri -> TERMINAL``
    (jobs that read ``uri`` and recorded no output). These are the groups
    ``build_graph_dict`` folds into one node.
    """
    if self_loop:
        return uri, uri
    if terminal == TERMINAL_INPUT:
        return TERMINAL, uri
    if terminal == TERMINAL_OUTPUT:
        return uri, TERMINAL
    if output:
        return uri, output
    return None


class ILineageRowStorage(IItemStorage[StoredLineageRow]):
    """Interface for lineage row storage implementations."""

    def get_rows_by_input(self, inputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``input`` is one of ``inputs`` (descendant hop)."""
        raise NotImplementedError

    def get_rows_by_output(self, outputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``output`` is one of ``outputs`` (ancestor hop)."""
        raise NotImplementedError

    def get_rows_by_job(self, job_id: str) -> List[StoredLineageRow]:
        """Return every row of one job execution."""
        raise NotImplementedError

    def get_rows_by_jobs(self, job_ids: List[str]) -> List[StoredLineageRow]:
        """Return every row of several job executions, in one query."""
        raise NotImplementedError

    def has_rows_for_job(self, job_id: str) -> bool:
        """Whether any row is already recorded for a job."""
        raise NotImplementedError

    def count_jobs_touching(
        self,
        uri: str,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> int:
        """Count the distinct jobs that consumed or produced ``uri``."""
        raise NotImplementedError

    def get_job_ids_touching(
        self,
        uri: str,
        limit: int,
        offset: int,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> List[str]:
        """Return one page of the distinct jobs touching ``uri``, by ``job_id``."""
        raise NotImplementedError

    def filter_jobs_touching(
        self,
        uri: str,
        job_ids: Iterable[str],
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> Set[str]:
        """Return which of ``job_ids`` consumed or produced ``uri``."""
        raise NotImplementedError

    def get_recorded_jobs(self, job_ids: List[str]) -> set:
        """Return which of ``job_ids`` already have rows."""
        raise NotImplementedError

    def get_job_ids_by_tags(
        self, any_of: List[str], all_of: Optional[List[str]] = None
    ) -> Set[str]:
        """Return the jobs carrying any of ``any_of`` and all of ``all_of``."""
        raise NotImplementedError

    def get_tags(self, job_ids: List[str]) -> Dict[str, List[str]]:
        """Return each job's tags as sorted ``k=v`` strings, keyed by ``job_id``."""
        raise NotImplementedError

    def grouped_edges(
        self, frontier: List[str], direction: str, limit: Optional[int] = None
    ) -> List[GroupedEdge]:
        """Return one level of the graph walk, one :class:`GroupedEdge` per edge.

        ``direction`` is :data:`DOWNSTREAM` (rows whose ``input`` is in
        ``frontier``) or :data:`UPSTREAM` (rows whose ``output`` is). ``limit``
        caps the number of edges, ordered most recent first.
        """
        raise NotImplementedError


class BaseLineageRowStorage(BaseItemStorage[StoredLineageRow], ILineageRowStorage):
    """Base storage implementation for lineage rows.

    Provides the shared query logic across backends (SQL, SQLite).
    """

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredLineageRow
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_LINEAGE_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredLineageRow) -> dict:
        """Extract the queryable columns from a row.

        Every field returned here becomes a real column; everything else lives in
        the JSON blob, which is Text and therefore neither queryable nor
        indexable. The set is returned unconditionally -- a conditionally omitted
        key would be missing from the schema derived from the sample item.

        **Every promoted column is a string, by construction.** ``get_by_where``
        builds an ``IN`` clause only for string-typed columns and otherwise
        degrades *silently* to ``column == [list]`` -- a meaningless predicate that
        returns plausible but wrong rows with no error. Keeping the promoted set
        all-text makes that failure unreachable rather than merely avoided by
        convention.
        """
        fields_to_include = {
            "job_id",
            "input",
            "output",
            "origin",
            "recorded_at",
        }
        # mode="json" turns the LineageOrigin enum into its plain string value.
        return item.model_dump(include=fields_to_include, mode="json")

    @classmethod
    def _get_sample_item(cls) -> StoredLineageRow:
        """Return a sample row used to derive the table schema.

        Every column must be present and of its real type here: the SQL layer
        infers each column's type from this item's values. So this is the schema,
        not an example of one.
        """
        return StoredLineageRow(
            job_id="sample-job",
            input="lh://prod/ns/models/tbl/sample",
            output="hf://huggingface.co/models/org/sample",
        )

    def get_rows_by_input(self, inputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``input`` is one of ``inputs``.

        One batched, indexed query -- the descendant hop of a level-order walk.

        Args:
            inputs: normalized URIs of the current frontier. The terminal
                marker is dropped: a creation row's input identifies no artifact,
                so matching on it would join unrelated creations together.

        Returns:
            The matching rows, or an empty list when nothing is left to match.
        """
        wanted = self._batchable(inputs)
        if not wanted:
            return []
        return self.get_by_where({"input": wanted})

    def get_rows_by_output(self, outputs: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``output`` is one of ``outputs``.

        The ancestor hop; see :meth:`get_rows_by_input`.
        """
        wanted = self._batchable(outputs)
        if not wanted:
            return []
        return self.get_by_where({"output": wanted})

    @staticmethod
    def _batchable(identifiers: List[str]) -> List[str]:
        """Drop empty URIs and duplicates from a hop's frontier.

        The terminal marker must never reach a query: every creation row shares it,
        so a hop matching on it would treat unrelated creations as one node. The
        walk stops on a terminal instead of following it.
        """
        return sorted({value for value in identifiers if value and value != TERMINAL})

    def get_rows_by_job(self, job_id: str) -> List[StoredLineageRow]:
        """Return every row of one job execution.

        This is what makes the N*M decomposition lossless: the rows sharing a
        ``job_id`` still say which inputs and outputs that execution had, which is
        how the read path rebuilds a single run node from several rows.
        """
        if not job_id:
            return []
        return self.get_by_where({"job_id": job_id})

    def get_rows_by_jobs(self, job_ids: List[str]) -> List[StoredLineageRow]:
        """Return every row of several job executions, in one batched query."""
        wanted = self._batchable(list(job_ids))
        if not wanted:
            return []
        return self.get_by_where({"job_id": wanted})

    def count_jobs_touching(
        self,
        uri: str,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> int:
        """Count the distinct jobs that consumed or produced ``uri``.

        Jobs, not rows: a job can touch one artifact in several of its rows (an
        output is repeated on every row of a many-input job), so a row count
        overstates it. This fallback walks the rows; the SQL backend
        overrides it with one ``COUNT(DISTINCT job_id)``.
        """
        return len(self._jobs_touching(uri, self_loop, output, terminal))

    def get_job_ids_touching(
        self,
        uri: str,
        limit: int,
        offset: int,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> List[str]:
        """Return one page of the distinct jobs touching ``uri``, ordered by ``job_id``.

        Ordered by ``job_id`` so paging is stable and matches the tag listing. See
        :meth:`count_jobs_touching` for this fallback's cost.
        """
        return sorted(self._jobs_touching(uri, self_loop, output, terminal))[
            offset : offset + limit
        ]

    def filter_jobs_touching(
        self,
        uri: str,
        job_ids: Iterable[str],
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> Set[str]:
        """Return which of ``job_ids`` consumed or produced ``uri``.

        Two indexed queries bounded by ``job_ids``, so a narrow filter -- a build's
        tag -- stays cheap however many runs the artifact has.
        """
        wanted = self._batchable(list(job_ids))
        if not wanted or not uri or uri == TERMINAL:
            return set()
        pair = endpoint_pair(uri, self_loop, output, terminal)
        if pair:
            return {
                row.job_id
                for row in self.get_by_where(
                    {"input": pair[0], "output": pair[1], "job_id": wanted}
                )
            }
        return {
            row.job_id
            for column in ("input", "output")
            for row in self.get_by_where({column: uri, "job_id": wanted})
        }

    def _jobs_touching(
        self,
        uri: str,
        self_loop: bool = False,
        output: Optional[str] = None,
        terminal: Optional[str] = None,
    ) -> Set[str]:
        """Every job with ``uri`` as an input or an output, read row by row.

        With ``self_loop``, only the jobs with ``uri`` as both -- its in-place
        rewrites. With ``output``, only the jobs with a ``uri -> output`` row. With
        ``terminal``, see :func:`endpoint_pair`.
        """
        if not uri or uri == TERMINAL:
            return set()
        pair = endpoint_pair(uri, self_loop, output, terminal)
        if pair:
            return {
                row.job_id
                for page in self.get_paged({"input": pair[0], "output": pair[1]})
                for row in page
            }
        return {
            row.job_id
            for column in ("input", "output")
            for page in self.get_paged({column: uri})
            for row in page
        }

    def get_job_ids_by_tags(
        self, any_of: List[str], all_of: Optional[List[str]] = None
    ) -> Set[str]:
        """Return the jobs matching a tag filter, W&B's ``$in`` + required shape.

        Tags live in each row's ``attributes.job.tags`` and every row of a job
        carries the same map, so a row-level match is a job-level match. This
        fallback reads every row; the SQL backend filters on the JSON in the query.

        Args:
            any_of: a job must carry at least one of these. Empty means no ``$in``
                constraint, in which case ``all_of`` alone decides.
            all_of: a job must carry every one of these.

        Returns:
            The matching job ids. Empty when both lists are empty: an unfiltered
            request is not a tag query, and answering it would list every job.
        """
        wanted_any, wanted_all = _tag_pairs(any_of), _tag_pairs(all_of or [])
        if not wanted_any and not wanted_all:
            return set()
        return {
            row.job_id
            for page in self.get_paged()
            for row in page
            if _tags_match(row_tags(row), wanted_any, wanted_all)
        }

    def get_tags(self, job_ids: List[str]) -> Dict[str, List[str]]:
        """Return each job's tags, sorted ``k=v``, in one query over its rows.

        A job with no tags is absent from the result rather than mapped to ``[]``.
        """
        tags_by_job: Dict[str, Set[str]] = {}
        for row in self.get_rows_by_jobs(job_ids):
            tags = row_tags(row)
            if tags:
                tags_by_job.setdefault(row.job_id, set()).update(tag_strings(tags))
        return {job_id: sorted(tags) for job_id, tags in tags_by_job.items()}

    def grouped_edges(
        self, frontier: List[str], direction: str, limit: Optional[int] = None
    ) -> List[GroupedEdge]:
        """Group one hop's rows by edge, in Python.

        The fallback for backends without SQL; the SQL backend overrides it with a
        single ``GROUP BY`` answered from the composite indexes.
        """
        if direction == DOWNSTREAM:
            rows = self.get_rows_by_input(frontier)
        elif direction == UPSTREAM:
            rows = self.get_rows_by_output(frontier)
        else:
            raise ValueError(f"Unknown direction: {direction!r}")
        groups: dict = {}
        for row in rows:
            key = (row.input, row.output)
            jobs, last, sample = groups.get(key, (set(), "", row.job_id))
            jobs.add(row.job_id)
            groups[key] = (jobs, max(last, row.recorded_at), min(sample, row.job_id))
        edges = [
            GroupedEdge(i, o, len(jobs), last, sample)
            for (i, o), (jobs, last, sample) in groups.items()
        ]
        edges.sort(key=lambda edge: (edge.last_recorded_at, edge.input, edge.output), reverse=True)
        return edges[:limit] if limit is not None else edges

    def has_rows_for_job(self, job_id: str) -> bool:
        """Whether any row is already recorded for a job."""
        if not job_id:
            return False
        return bool(self.get_by_where({"job_id": job_id}))

    def get_recorded_jobs(self, job_ids: List[str]) -> set:
        """Return which of ``job_ids`` already have rows.

        The sink's dedup is presence-based: a job's rows are written together, so a
        job either has them or it does not. It deliberately does not compare a row
        count against the reconciler's ``expected_counts``, which counts one W&B run
        per output artifact -- a shape that never matches an N*M row count, and
        would report every job as unrecorded forever.

        Keyed on ``job_id`` rather than on any process id because ``job_id`` is the
        only identifier every lineage source has by definition. A build or a target
        run is granite.build's own concept and is absent from every imported row, so
        deduping on one would leave imported sources with no dedup at all.

        Args:
            job_ids: the job executions to check.

        Returns:
            The subset that already has at least one row.
        """
        wanted = self._batchable(list(job_ids))
        if not wanted:
            return set()
        rows = self.get_by_where({"job_id": wanted})
        return {row.job_id for row in rows if row.job_id}


def row_tags(row: StoredLineageRow) -> Dict[str, str]:
    """A row's ``attributes.job.tags`` map, or ``{}``."""
    job = (row.attributes or {}).get("job") or {}
    tags = job.get("tags") or {}
    return tags if isinstance(tags, dict) else {}


def _tag_pair(tag: str) -> Tuple[str, str]:
    """``k=v`` as ``(k, v)``; a bare tag (a user's ``nightly``) is ``(tag, "")``."""
    key, _, value = tag.partition("=")
    return key, value


def _tag_pairs(tags: Iterable[str]) -> List[Tuple[str, str]]:
    """Tag strings as sorted, deduped ``(k, v)`` pairs; empty ones dropped."""
    return sorted({_tag_pair(tag) for tag in tags if tag})


def tags_to_map(tags: Iterable[str]) -> Dict[str, str]:
    """Tag strings as the ``attributes.job.tags`` map a row stores."""
    return dict(_tag_pairs(tags))


def tag_strings(tags: Dict[str, str]) -> List[str]:
    """A stored tag map back as sorted strings: ``k=v``, or ``k`` for a bare tag."""
    return sorted(f"{key}={value}" if value else key for key, value in tags.items())


def _tags_match(
    tags: Dict[str, str],
    any_of: List[Tuple[str, str]],
    all_of: List[Tuple[str, str]],
) -> bool:
    if any(tags.get(key) != value for key, value in all_of):
        return False
    return not any_of or any(tags.get(key) == value for key, value in any_of)
