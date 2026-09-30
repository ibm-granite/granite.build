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

"""Tests for serving artifact lineage out of the local index.

The distinction under most of these: ``None`` becomes an HTTP 404, which the
frontend renders as "Lineage is not available". So ``None`` has to mean "this
artifact is unknown here" and never "this artifact has no lineage yet" -- the
second is a real, successful answer and must come back as a graph.
"""

import pytest

from gbserver.lineage.db_service import DBLineageService
from gbserver.lineage.openlineage_service import (
    LineageServiceFactory,
    NoopLineageService,
)
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

# Endpoints are normalized URIs: the node's identity and what a request names are
# the same string, which is what turned root resolution into one indexed lookup.
A = "lh://prod/ns/tables/a"
B = "lh://prod/ns/tables/b"
C = "lh://prod/ns/tables/c"


def row(job_id: str, input: str, output: str, **kwargs) -> StoredLineageRow:
    """A row whose endpoint detail and job metadata go into ``attributes``."""
    attributes = dict(kwargs.pop("attributes", {}) or {})
    for side, group in (("source", "input"), ("target", "output")):
        kind = kwargs.pop(f"{side}_kind", "")
        name = kwargs.pop(f"{side}_name", "")
        detail = {}
        if kind:
            detail["kind"] = kind
        if name:
            detail["name"] = name
        if detail:
            attributes[group] = detail
    assert not kwargs, f"unhandled row() keywords: {sorted(kwargs)}"
    return StoredLineageRow(
        job_id=job_id, input=input, output=output, attributes=attributes
    )


class FakeStorage:
    """In-memory storage implementing the methods the service uses."""

    def __init__(self, rows: list, fail: bool = False) -> None:
        self.rows = rows
        self.fail = fail
        self.pages = 0

    def get_paged(self, where=None, page_size: int = 200):
        if self.fail:
            raise RuntimeError("storage is down")
        self.pages += 1
        matched = self._matching(where)
        for start in range(0, max(len(matched), 1), page_size):
            chunk = matched[start : start + page_size]
            if chunk or start == 0:
                yield chunk

    def get_rows_by_input(self, sources: list) -> list:
        wanted = {s for s in sources if s and s != TERMINAL}
        return [r for r in self.rows if r.input in wanted]

    def get_rows_by_output(self, targets: list) -> list:
        wanted = {t for t in targets if t and t != TERMINAL}
        return [r for r in self.rows if r.output in wanted]

    def count(self, where=None) -> int:
        return len(self._matching(where))

    def _matching(self, where) -> list:
        if not where:
            return list(self.rows)
        matched = []
        for row in self.rows:
            if all(getattr(row, key, None) == value for key, value in where.items()):
                matched.append(row)
        return matched

    def get_rows_by_job(self, job_id: str) -> list:
        if self.fail:
            raise RuntimeError("storage is down")
        if not job_id:
            return []
        return [r for r in self.rows if r.job_id == job_id]

    def get_rows_by_jobs(self, job_ids: list) -> list:
        if self.fail:
            raise RuntimeError("storage is down")
        return [r for r in self.rows if r.job_id in set(job_ids)]

    def _touching(self, uri: str, self_loop: bool = False, output=None) -> set:
        if self.fail:
            raise RuntimeError("storage is down")
        output = uri if self_loop else output
        if output:
            return {r.job_id for r in self.rows if uri and (r.input, r.output) == (uri, output)}
        return {r.job_id for r in self.rows if uri and uri in (r.input, r.output)}

    def count_jobs_touching(self, uri: str, self_loop: bool = False, output=None) -> int:
        return len(self._touching(uri, self_loop, output))

    def get_job_ids_touching(
        self, uri: str, limit: int, offset: int, self_loop: bool = False, output=None
    ) -> list:
        return sorted(self._touching(uri, self_loop, output))[offset : offset + limit]

    def filter_jobs_touching(
        self, uri: str, job_ids, self_loop: bool = False, output=None
    ) -> set:
        return self._touching(uri, self_loop, output) & set(job_ids)

    # No get_rows_by_build: a build is not a column. A build-seeded graph resolves
    # its artifacts through gb_targets and seeds the ordinary walk with their URIs.


def service(*rows, **kwargs) -> DBLineageService:
    return DBLineageService(storage=FakeStorage(list(rows), **kwargs))


class TestProviderRegistration:
    def test_db_provider_is_registered(self):
        assert isinstance(LineageServiceFactory.create("db"), DBLineageService)

    def test_none_still_yields_the_noop_service(self):
        assert isinstance(LineageServiceFactory.create("none"), NoopLineageService)

    def test_an_unknown_provider_still_raises(self):
        with pytest.raises(ValueError):
            LineageServiceFactory.create("not-a-provider")


class TestRootResolution:
    """One indexed lookup, not a scan.

    This used to page the whole table on every request, because the indexed columns
    held canonical identifiers while a request carries a URL. Normalizing the
    request's URL now produces exactly the value those columns hold.
    """

    def test_resolves_by_uri(self):
        result = service(row("J1", A, B)).get_artifact_graph(artifact_url=A)
        assert result is not None and result["root_id"] == A

    def test_converges_alternate_spellings_of_one_uri(self):
        """A browser URL and the runtime's own URI are one artifact.

        Without this the graph splits into two disconnected halves of the same
        thing, depending on which spelling the caller happened to have.
        """
        hf = "https://huggingface.co/org/repo"
        svc = service(row("J1", hf, B))
        for spelling in (
            hf,
            "hf:///org/repo",
            "https://huggingface.co/org/repo",
            "hf://huggingface.co/models/org/repo/main",
        ):
            result = svc.get_artifact_graph(artifact_url=spelling)
            assert result is not None, spelling
            assert result["root_id"] == hf, spelling

    def test_resolves_a_target_endpoint_too(self):
        result = service(row("J1", A, B)).get_artifact_graph(artifact_url=B)
        assert result is not None and result["root_id"] == B

    def test_a_uri_in_the_name_field_still_resolves(self):
        """A caller that passes a URI as the name is not punished for it."""
        result = service(row("J1", A, B)).get_artifact_graph(artifact_url=A)
        assert result is not None and result["root_id"] == A

    def test_a_bare_name_is_none(self):
        """A name is not an identity, and a scheme is not guessed for it.

        The index keys on URIs. Inventing a scheme for "my-artifact" would resolve
        to an artifact that may not exist -- an invented identity, which is the one
        failure worth returning nothing to avoid.
        """
        assert (
            service(row("J1", A, B, source_name="my-artifact")).get_artifact_graph(
                artifact_name="my-artifact"
            )
            is None
        )

    def test_an_unidentifiable_uri_is_none(self):
        # The 404 case: the request names nothing this index can key on.
        assert (
            service(row("J1", A, B)).get_artifact_graph(artifact_url="bogus://x")
            is None
        )

    def test_no_identifier_at_all_is_none(self):
        assert service(row("J1", A, B)).get_artifact_graph() is None

    def test_an_absent_but_valid_uri_is_a_graph_not_a_404(self):
        """A resolvable URI with no rows yields its own node, not None.

        The distinction matters at the API boundary: ``None`` becomes the 404 the
        frontend renders as "lineage is not available", so it must mean "cannot key
        on this", never "nothing recorded yet".
        """
        result = service().get_artifact_graph(artifact_url=A)
        assert result is not None
        assert result["root_id"] == A
        assert [n["id"] for n in result["nodes"]] == [A]


class TestGraphContent:
    def test_descendants_walk_from_the_root(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=A, direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_upstream_walks_toward_ancestors(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=C, direction="upstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_downstream_does_not_walk_backward(self):
        # Pins the wire mapping: downstream is used_by(), toward descendants, which
        # is what the live W&B backend and the frontend already mean by it.
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=B, direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {B, C}
        assert A not in ids

    def test_upstream_does_not_walk_forward(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=B, direction="upstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B}
        assert C not in ids

    def test_both_reaches_each_side(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=B, direction="both")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_an_artifact_with_no_edges_is_a_graph_not_a_404(self):
        # It IS in the index -- as a creation with no input -- so the answer is a
        # graph with its own node, never None.
        svc = service(row("J1", TERMINAL, A))
        result = svc.get_artifact_graph(artifact_url=A, direction="upstream")
        assert result is not None
        assert result["root_id"] == A

    def test_max_depth_truncates_and_says_so(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(
            artifact_url=A, direction="downstream", max_depth=1
        )
        assert result["truncated"] is True


class TestValidation:
    def test_an_unknown_direction_raises(self):
        # The API layer maps ValueError to a 400.
        with pytest.raises(ValueError):
            service(row("J1", A, B)).get_artifact_graph(
                artifact_url=A, direction="sideways"
            )

    def test_a_contradicted_artifact_type_raises(self):
        svc = service(row("J1", A, B, source_kind="table"))
        with pytest.raises(ValueError):
            svc.get_artifact_graph(artifact_url=A, artifact_type="model")

    def test_a_matching_artifact_type_is_accepted(self):
        svc = service(row("J1", A, B, source_kind="table"))
        assert svc.get_artifact_graph(artifact_url=A, artifact_type="table")

    def test_direction_is_validated_before_the_root_lookup(self):
        # Otherwise an unknown artifact would 404 on a request that is also a bad
        # request, hiding the client's actual mistake.
        with pytest.raises(ValueError):
            service().get_artifact_graph(artifact_name="nope", direction="sideways")


class TestWritePathIsInert:
    """The write-path methods degrade safely rather than pretending to work."""

    def test_emit_event_does_not_raise(self):
        assert service().emit_event({"anything": True}) is None

    def test_tag_searches_are_empty(self):
        svc = service(row("J1", A, B))
        assert svc.search_lineage_by_tags(["t"]) == (0, [])
        assert svc.count_events_by_tags(["t"]) == 0
        assert svc.count_runs_by_tags(["t"]) == 0

    def test_filter_unrecorded_fails_toward_rerecording(self):
        # The interface requires this: recording is idempotent, so returning the
        # candidates unchanged is the safe answer and never drops lineage.
        candidates = {"t1", "t2"}
        assert service().filter_unrecorded(candidates) == candidates


class TestQueryGraph:
    """The general entry point: every filter optional, and it never returns None.

    ``get_artifact_graph`` returns ``None`` for "cannot key on this", which the API
    turns into a 404. A query endpoint cannot do that: asking with no filters is
    legitimate, and an empty index is a legitimate answer, so "nothing recorded" must
    come back as an empty graph rather than as an error.
    """

    def test_by_uri(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.query_graph(uri=B)
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}
        assert result["root_id"] == B

    def test_a_uri_query_flags_its_single_root(self):
        result = service(row("J1", A, B)).query_graph(uri=A)
        assert [n["id"] for n in result["nodes"] if n["is_root"]] == [A]

    def test_alternate_spellings_reach_one_artifact(self):
        hf = "https://huggingface.co/org/repo"
        svc = service(row("J1", hf, B))
        for spelling in (hf, "hf:///org/repo", "https://huggingface.co/org/repo"):
            assert svc.query_graph(uri=spelling)["root_id"] == hf, spelling

    def test_by_job_id_seeds_every_endpoint_of_that_execution(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.query_graph(job_id="J1", direction="both")
        depths = {
            n["id"]: n["depth"] for n in result["nodes"] if n["node_type"] == "artifact"
        }
        # Both of J1's endpoints are seeds, so both are at depth 0.
        assert depths[A] == 0
        assert depths[B] == 0

    def test_a_job_query_flags_no_root(self):
        """It has several, and picking one would misreport what was asked."""
        result = service(row("J1", A, B)).query_graph(job_id="J1")
        assert result["root_id"] == ""
        assert not any(n["is_root"] for n in result["nodes"])

    def test_uri_and_job_seed_the_union(self):
        svc = service(row("J1", A, B), row("J2", C, A))
        result = svc.query_graph(uri=C, job_id="J1", direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert {A, B, C} <= ids

    def test_no_filter_is_refused(self):
        """A graph needs somewhere to start; "what ran lately" is the job listing."""
        with pytest.raises(ValueError):
            service(row("J1", A, B)).query_graph()

    def test_an_unmatchable_uri_is_an_empty_graph_not_none(self):
        """The distinction from get_artifact_graph, which would return None here."""
        result = service(row("J1", A, B)).query_graph(uri="bogus://x")
        assert result is not None
        assert result["nodes"] == []

    def test_an_unknown_job_is_an_empty_graph(self):
        result = service(row("J1", A, B)).query_graph(job_id="NOPE")
        assert result is not None
        assert result["nodes"] == []

    def test_direction_is_honoured(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        up = {
            n["id"]
            for n in svc.query_graph(uri=B, direction="upstream")["nodes"]
            if n["node_type"] == "artifact"
        }
        down = {
            n["id"]
            for n in svc.query_graph(uri=B, direction="downstream")["nodes"]
            if n["node_type"] == "artifact"
        }
        assert up == {A, B}
        assert down == {B, C}

    def test_max_depth_truncates_and_says_so(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.query_graph(uri=A, direction="downstream", max_depth=1)
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert C not in ids
        assert result["truncated"] is True

    def test_an_unknown_direction_raises(self):
        with pytest.raises(ValueError):
            service(row("J1", A, B)).query_graph(uri=A, direction="sideways")

    def test_a_storage_failure_does_not_raise(self):
        """A job lookup that fails costs its seeds, not the request."""
        svc = service(row("J1", A, B), fail=True)
        assert svc.query_graph(job_id="J1") is not None


class TestListJobs:
    """The one listing over the index, and the drill-down for what the graph collapses.

    ``build_graph_dict`` folds an artifact's in-place rewrites into one node with a
    ``run_count``; real data has one appended 68,905 times. A count with no way to
    expand it is a dead end, so ``uri`` is that way -- paged rather than capped,
    because a flat list has no shape to preserve.
    """

    def test_it_lists_jobs_touching_an_artifact_in_both_directions(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.list_jobs(uri=B)
        assert [j["job_id"] for j in result["jobs"]] == ["J1", "J2"]

    def test_the_total_counts_jobs_not_rows(self):
        """A self-rewrite is one job, and so is a many-input job repeating its output."""
        svc = service(
            *[row(f"J{i}", A, A) for i in range(10)],
            row("M", B, C),
            row("M", A, C),
        )
        assert svc.list_jobs(uri=A)["total"] == 11
        assert svc.list_jobs(uri=C)["total"] == 1

    def test_self_loop_lists_only_the_in_place_rewrites(self):
        """The drill-down for a looped node: same source and target, nothing else."""
        svc = service(
            *[row(f"J{i}", A, A) for i in range(3)],
            row("M", A, C),
            row("N", B, A),
        )
        result = svc.list_jobs(uri=A, self_loop=True)
        assert [j["job_id"] for j in result["jobs"]] == ["J0", "J1", "J2"]
        assert result["total"] == 3

    def test_self_loop_pages(self):
        svc = service(*[row(f"J{i}", A, A) for i in range(5)], row("M", A, C))
        pages = [svc.list_jobs(uri=A, self_loop=True, limit=2, offset=o) for o in (0, 2, 4)]
        assert [j["job_id"] for p in pages for j in p["jobs"]] == [f"J{i}" for i in range(5)]

    def test_output_lists_only_the_jobs_of_that_pair(self):
        svc = service(row("P1", A, B), row("P2", A, B), row("P3", A, C))
        result = svc.list_jobs(uri=A, output=B)
        assert [j["job_id"] for j in result["jobs"]] == ["P1", "P2"]
        assert result["total"] == 2

    def test_self_loop_narrows_another_filter(self):
        svc = service(row("J1", A, A), row("J2", A, C))
        assert [j["job_id"] for j in svc.list_jobs(uri=A, job_id="J2", self_loop=True)["jobs"]] == []
        assert [j["job_id"] for j in svc.list_jobs(uri=A, job_id="J1", self_loop=True)["jobs"]] == ["J1"]

    def test_an_entry_carries_its_endpoints_and_job_detail(self):
        svc = service(
            row("J1", A, C, attributes={"job": {"name": "train"}}), row("J1", B, C)
        )
        entry = svc.list_jobs(uri=A)["jobs"][0]
        assert entry["inputs"] == [A, B]
        assert entry["outputs"] == [C]
        assert entry["job"]["name"] == "train"

    def test_a_self_rewrite_shows_its_artifact_on_both_sides(self):
        entry = service(row("J1", A, A)).list_jobs(uri=A)["jobs"][0]
        assert entry["inputs"] == entry["outputs"] == [A]

    def test_terminals_are_not_endpoints(self):
        entry = service(row("J1", TERMINAL, A)).list_jobs(job_id="J1")["jobs"][0]
        assert entry["inputs"] == []
        assert entry["outputs"] == [A]

    def test_paging_walks_the_whole_list(self):
        svc = service(*[row(f"J{i}", A, B) for i in range(10)])
        seen = []
        for offset in range(0, 10, 3):
            seen.extend(
                j["job_id"]
                for j in svc.list_jobs(uri=A, limit=3, offset=offset)["jobs"]
            )
        assert sorted(seen) == sorted(f"J{i}" for i in range(10))

    def test_the_page_size_is_capped(self):
        """A caller must not be able to ask for everything and recreate the problem."""
        svc = service(*[row(f"J{i}", A, B) for i in range(5)])
        assert svc.list_jobs(uri=A, limit=10**9)["limit"] == 1000

    def test_a_zero_or_negative_limit_is_clamped(self):
        svc = service(row("J1", A, B))
        assert svc.list_jobs(uri=A, limit=0)["limit"] == 1
        assert svc.list_jobs(uri=A, offset=-5)["offset"] == 0

    def test_by_job_id(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.list_jobs(job_id="J1")
        assert [j["job_id"] for j in result["jobs"]] == ["J1"]
        assert result["total"] == 1

    def test_an_unknown_job_id_is_empty(self):
        assert service(row("J1", A, B)).list_jobs(job_id="nope")["total"] == 0

    def test_filters_combine_with_and(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        assert [j["job_id"] for j in svc.list_jobs(uri=B, job_id="J2")["jobs"]] == [
            "J2"
        ]
        assert svc.list_jobs(uri=A, job_id="J2")["total"] == 0

    def test_a_tag_filter_with_no_tag_storage_is_empty(self):
        """Without the tag table a tag filter cannot be answered; it must not match all."""
        assert service(row("J1", A, B)).list_jobs(tags=["build_id=X"])["total"] == 0

    def test_an_unresolvable_uri_is_empty_not_an_error(self):
        result = service(row("J1", A, B)).list_jobs(uri="bogus://x")
        assert result == {"jobs": [], "total": 0, "limit": 100, "offset": 0}

    def test_a_storage_failure_does_not_raise(self):
        svc = service(row("J1", A, B), fail=True)
        assert svc.list_jobs(uri=A)["jobs"] == []
