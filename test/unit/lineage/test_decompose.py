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

"""Tests for job -> lineage row decomposition."""

import pytest

from gbserver.lineage.decompose import (
    JOB_METADATA_KEYS,
    LineageDecomposeError,
    LineageRowDraft,
    group_by_job,
    to_lineage_rows,
)


def artifact(name: str, **extra) -> dict:
    """An artifact dict whose URI is derived from a short name.

    Endpoints are normalized URIs now, so the fixtures have to be real URIs -- a
    bare name has no scheme and would be dropped, which is the behaviour the
    "unidentifiable" tests assert on purpose.
    """
    if "uri" not in extra:
        extra["uri"] = uri_for(name)
    return {"name": name, **extra}


def uri_for(name: str) -> str:
    """The URI the fixtures use for a given short name."""
    return f"s3://bucket/{name}"


def job(job_id: str = "J", sources=None, targets=None, **extra) -> dict:
    entry: dict = {"job_id": job_id}
    if sources is not None:
        entry["sources"] = sources
    if targets is not None:
        entry["targets"] = targets
    entry.update(extra)
    return entry


def pairs(rows) -> list[tuple]:
    """Endpoint pairs, mapped back to short names so assertions stay readable.

    A terminal stays "" -- it is a real value, not a missing one.
    """
    return [(unname(row.input), unname(row.output)) for row in rows]


def unname(endpoint: str) -> str:
    """Inverse of uri_for, so an assertion can talk in short names."""
    prefix = "s3://bucket/"
    return endpoint[len(prefix):] if endpoint.startswith(prefix) else endpoint


class TestTerminals:
    """An empty endpoint is real information, not a missing value.

    It is ``""`` rather than ``None`` so it survives SQL: NULL never equals NULL, so
    NULL endpoints would slip past the ``(job_id, input, output)`` unique index and
    leave creation/deletion rows as the only ones a re-ingest could duplicate.
    """

    def test_no_targets_is_a_deletion_terminal(self):
        rows = to_lineage_rows(
            job(sources=[artifact("a"), artifact("b")], targets=[])
        )
        assert pairs(rows) == [("a", ""), ("b", "")]

    def test_no_sources_is_a_creation_terminal(self):
        rows = to_lineage_rows(job(sources=[], targets=[artifact("out")]))
        assert pairs(rows) == [("", "out")]

    def test_creation_with_several_outputs(self):
        rows = to_lineage_rows(job(targets=[artifact("x"), artifact("y")]))
        assert pairs(rows) == [("", "x"), ("", "y")]

    def test_missing_keys_behave_like_empty_lists(self):
        rows = to_lineage_rows(job(targets=[artifact("out")]))
        assert pairs(rows) == [("", "out")]


class TestFanShapes:
    """The prototype's cases 2b and 3, with no special-casing."""

    def test_many_sources_one_target(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b"), artifact("c")],
                targets=[artifact("out")],
            ),
        )
        assert pairs(rows) == [("a", "out"), ("b", "out"), ("c", "out")]

    def test_one_source_many_targets(self):
        rows = to_lineage_rows(
            job(sources=[artifact("in")], targets=[artifact("x"), artifact("y")]),
        )
        assert pairs(rows) == [("in", "x"), ("in", "y")]

    def test_one_to_one(self):
        rows = to_lineage_rows(
            job(sources=[artifact("in")], targets=[artifact("out")])
        )
        assert pairs(rows) == [("in", "out")]


class TestCartesianCase:
    """N sources AND M targets -- rejected, as in the prototype.

    The guard raises when ``len(sources) > 1 and len(targets) > 1``, so every
    accepted job has ``min(#sources, #targets) <= 1``. No granite.build producer
    can build such a job: ``wandb_jobstats`` emits one event per output artifact,
    so a target run with 3 inputs and 2 outputs arrives as two jobs of 3 sources
    x 1 target, never as one 3x2 job.
    """

    def test_three_inputs_two_outputs_raises(self):
        with pytest.raises(LineageDecomposeError, match="too many sources 3"):
            to_lineage_rows(
                job(
                    sources=[artifact("i1"), artifact("i2"), artifact("i3")],
                    targets=[artifact("o1"), artifact("o2")],
                ),
            )

    def test_two_by_two_raises(self):
        with pytest.raises(LineageDecomposeError, match="too many sources 2"):
            to_lineage_rows(
                job(
                    sources=[artifact("a"), artifact("b")],
                    targets=[artifact("x"), artifact("y")],
                ),
            )

    def test_fan_out_on_one_side_is_accepted(self):
        """The guard bounds only the both-sides case; either fan-out is fine."""
        many_sources = to_lineage_rows(
            job(
                job_id="run-7",
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x")],
            ),
        )
        assert pairs(many_sources) == [("a", "x"), ("b", "x")]
        assert {row.job_id for row in many_sources} == {"run-7"}

        many_targets = to_lineage_rows(
            job(
                job_id="run-8",
                sources=[artifact("a")],
                targets=[artifact("x"), artifact("y")],
            ),
        )
        assert pairs(many_targets) == [("a", "x"), ("a", "y")]
        assert {row.job_id for row in many_targets} == {"run-8"}


class TestRegrouping:
    """Why the flattening is a projection and not a loss of information.

    The flat rows still say which inputs and which outputs one execution had,
    which is what makes the fan-out shapes the guard accepts safe to store flat.
    """

    def test_inputs_and_outputs_are_recoverable(self):
        """A 3-input target run, in the two per-output jobs a producer emits.

        The guard rejects a single 3x2 job, so this is the shape that actually
        reaches storage -- and both outputs still regroup to the same three
        inputs, because ``job_id`` is shared across each job's rows.
        """
        rows = to_lineage_rows(
            job(
                job_id="J",
                sources=[artifact("i1"), artifact("i2"), artifact("i3")],
                targets=[artifact("o1")],
            ),
        ) + to_lineage_rows(
            job(
                job_id="J2",
                sources=[artifact("i1"), artifact("i2"), artifact("i3")],
                targets=[artifact("o2")],
            ),
        )
        grouped = group_by_job(rows)
        expected_inputs = {uri_for("i1"), uri_for("i2"), uri_for("i3")}
        assert grouped["J"]["sources"] == expected_inputs
        assert grouped["J"]["targets"] == {uri_for("o1")}
        assert grouped["J2"]["sources"] == expected_inputs
        assert grouped["J2"]["targets"] == {uri_for("o2")}

    def test_terminals_regroup_without_none(self):
        creation = to_lineage_rows(job("C", targets=[artifact("o")]))
        deletion = to_lineage_rows(job("D", sources=[artifact("i")]))
        grouped = group_by_job(creation + deletion)
        assert grouped["C"]["sources"] == set()
        assert grouped["C"]["targets"] == {uri_for("o")}
        assert grouped["D"]["sources"] == {uri_for("i")}
        assert grouped["D"]["targets"] == set()

    def test_several_jobs_stay_separate(self):
        rows = to_lineage_rows(
            job("J1", sources=[artifact("a")], targets=[artifact("x")])
        ) + to_lineage_rows(
            job("J2", sources=[artifact("b")], targets=[artifact("y")])
        )
        grouped = group_by_job(rows)
        assert set(grouped) == {"J1", "J2"}
        assert grouped["J1"]["sources"] == {uri_for("a")}
        assert grouped["J2"]["sources"] == {uri_for("b")}


class TestSelfLoop:
    """A job rewriting its own input -- legitimate for unversioned entities."""

    def test_same_artifact_in_and_out(self):
        rows = to_lineage_rows(
            job(sources=[artifact("tbl")], targets=[artifact("tbl")])
        )
        assert pairs(rows) == [("tbl", "tbl")]


class TestMetadata:
    def test_metadata_is_copied_onto_every_row(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x")],
                job_name="train",
                owner="someone",
                job_status="SUCCESS",
            ),
        )
        for row in rows:
            assert row.metadata["job_name"] == "train"
            assert row.metadata["owner"] == "someone"
            assert row.metadata["job_status"] == "SUCCESS"

    def test_unknown_keys_are_not_carried(self):
        rows = to_lineage_rows(
            job(sources=[artifact("a")], targets=[artifact("x")], not_a_key="v"),
        )
        assert "not_a_key" not in rows[0].metadata

    def test_absent_metadata_is_omitted_not_defaulted(self):
        rows = to_lineage_rows(
            job(sources=[artifact("a")], targets=[artifact("x")])
        )
        assert rows[0].metadata == {"job_id": "J"}

    def test_each_row_owns_its_metadata(self):
        """Rows must not share one dict, or mutating one would edit them all."""
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x")],
                job_name="n",
            ),
        )
        rows[0].metadata["job_name"] = "changed"
        assert rows[1].metadata["job_name"] == "n"

    def test_job_id_is_a_metadata_key(self):
        assert "job_id" in JOB_METADATA_KEYS


class TestPartitionFiltersAreOutOfScope:
    """A draft carries no partition filter, on purpose.

    The old row had ``source_filter``/``target_filter`` columns that were always
    ``None``: ``ArtifactRegistration`` has no filter or partition field, so no
    producer could emit one, and the traversal never read them. They are gone rather
    than carried empty.

    If partitioned artifacts ever arrive, add normalization and traversal
    propagation *together* -- a raw array-shaped filter and an equivalent
    object-shaped one are different strings, so rows written before normalization
    would never match once propagation starts comparing them.
    """

    def test_a_filter_on_the_artifact_is_ignored(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a", filter='{"dt":"2024"}')],
                targets=[artifact("x", filter='{"dt":"2025"}')],
            ),
        )
        assert not hasattr(rows[0], "source_filter")
        assert not hasattr(rows[0], "target_filter")

    def test_a_filtered_artifact_is_still_one_node(self):
        """Two partitions of one table share a URI, so they share a node."""
        rows = to_lineage_rows(
            job(
                sources=[artifact("tbl", filter='{"dt":"2024"}')],
                targets=[artifact("out")],
            ),
        )
        other = to_lineage_rows(
            job(
                job_id="J2",
                sources=[artifact("tbl", filter='{"dt":"2025"}')],
                targets=[artifact("out2")],
            ),
        )
        assert rows[0].input == other[0].input


class TestArtifactsAreCarried:
    def test_source_and_target_artifacts_are_kept(self):
        source, target = artifact("a", type="model"), artifact("x", type="dataset")
        rows = to_lineage_rows(job(sources=[source], targets=[target]))
        assert rows[0].input_artifact == source
        assert rows[0].output_artifact == target

    def test_terminal_rows_carry_only_one_side(self):
        rows = to_lineage_rows(job(targets=[artifact("x")]))
        assert rows[0].input_artifact is None
        assert rows[0].output_artifact is not None


class TestRejectedInput:
    def test_missing_job_id_raises(self):
        with pytest.raises(LineageDecomposeError, match="job_id"):
            to_lineage_rows({"sources": [artifact("a")]})

    def test_empty_job_id_raises(self):
        with pytest.raises(LineageDecomposeError, match="job_id"):
            to_lineage_rows(job("", sources=[artifact("a")]))

    def test_no_sources_and_no_targets_raises(self):
        with pytest.raises(LineageDecomposeError, match="neither sources nor targets"):
            to_lineage_rows(job(sources=[], targets=[]))

    def test_an_unidentifiable_endpoint_becomes_terminal_not_an_error(self):
        """A URI with no identity rule ends a path; it does not abort the job.

        The endpoint is dropped rather than guessed -- an invented identity merges
        unrelated artifacts, which is the one failure worth losing a node to avoid --
        and the other endpoint of the row still records what it can. The drop is
        logged by ``normalize_uri`` so a producer emitting an unsupported shape is
        countable rather than silent.
        """
        rows = to_lineage_rows(
            job(
                sources=[{"name": "a", "uri": "bogus-scheme://nope"}],
                targets=[artifact("x")],
            ),
        )
        assert pairs(rows) == [("", "x")]

    def test_a_job_whose_every_endpoint_is_unidentifiable_still_decomposes(self):
        """Nothing raises; the caller decides what to do with a row that says nothing.

        ``db_jobstats`` drops such a row, because terminal-on-both-sides identifies
        no artifact and would join unrelated jobs through the empty endpoint.
        """
        rows = to_lineage_rows(
            job(
                sources=[{"name": "a", "uri": "bogus://x"}],
                targets=[{"name": "b", "uri": "bogus://y"}],
            ),
        )
        assert pairs(rows) == [("", "")]


class TestRowIdentity:
    def test_key_is_the_storage_unique_triple(self):
        row = LineageRowDraft(job_id="J", input="s3://b/a", output="s3://b/b")
        assert row.key() == ("J", "s3://b/a", "s3://b/b")

    def test_rows_of_one_job_have_distinct_keys(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b"), artifact("c")],
                targets=[artifact("x")],
            ),
        )
        assert len({row.key() for row in rows}) == len(rows)
