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

"""Tests for the sink that writes lineage into the local index.

Run against a real SQLite row storage, not a fake: the properties worth pinning
here -- that a re-ingest does not duplicate, that an already-recorded target is
skipped -- are enforced by the composite unique index and by presence queries, and
a fake would just re-implement (and possibly contradict) both.
"""

import os
import uuid as uuid_module

import pytest

from gbserver.lineage.attributes import (
    ALT_URIS,
    INPUT,
    OUTPUT,
    endpoint_kind,
    endpoint_name,
    job_detail,
    origin_id,
    origin_system,
)
from gbserver.lineage.db_jobstats import DBLineageStore
from gbserver.lineage.decompose import LineageRowDraft
from gbserver.lineage.row_indexing import LineageRowIndexer
from gbserver.lineage.row_indexing import row_from_draft as _row_from_draft
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="rows")
def rows_fixture():
    """A lineage row storage on a table unique to this test."""
    table = f"t_sink_{uuid_module.uuid4().hex[:8]}"
    return SqliteStorageFactory().create_lineage_row_storage(table_name=table)


@pytest.fixture(name="jobs")
def jobs_fixture():
    table = f"t_jobs_{uuid_module.uuid4().hex[:8]}"
    return SqliteStorageFactory().create_lineage_job_storage(table_name=table)


@pytest.fixture(name="sink")
def sink_fixture(jobs):
    """The store, which records into gb_lineage_job and nothing else."""
    return DBLineageStore(job_storage=jobs)


@pytest.fixture(name="indexer")
def indexer_fixture(rows):
    """The index writer, which owns gb_lineage_index. See row_indexing."""
    return LineageRowIndexer(storage=rows)


@pytest.fixture(name="write_job")
def write_job_fixture(sink, indexer, jobs):
    """Record a job entry, then derive its index rows -- the full write path.

    The store and the index are separate now (the store owns gb_lineage_job, the
    indexer owns gb_lineage_index), so a test that asserts on *rows* has to run
    both halves. This is what the indexer loop does per job, in order.
    """

    def run(job_entry, **kwargs):
        from gbserver.lineage.db_jobstats import _normalized_job

        sink._write_job(job_entry, **kwargs)
        # Normalize to find the id, because the entry arrives in either shape: a
        # flat ``job_id`` from the shared builders, or OpenLineage's nested
        # ``job_details.job_id``. This is the same lift the store itself does.
        record = jobs.get_job(_normalized_job(job_entry)["job_id"])
        if record is not None:
            indexer.index_job_record(record)

    return run


def job(job_id: str, sources: list, targets: list, **metadata) -> dict:
    """A job entry in the shape the shared builders emit."""
    return {
        "job_id": job_id,
        "sources": sources,
        "targets": targets,
        **metadata,
    }


def artifact(name: str, uri: str = "", space: str = "sp") -> dict:
    return {"name": name, "uri": uri, "space_name": space}


LH_TABLE = "lh://prod/ns/tables/raw_tbl"
LH_MODEL = "lh://prod/ns/models/mdl_tbl/trained/v1"


class TestDecomposition:
    """One job entry becomes max(N, M) rows sharing a job_id."""

    def test_one_input_one_output_is_one_row(self, write_job, rows):
        write_job(
            job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert len(rows.get_rows_by_job("J1")) == 1

    def test_two_inputs_one_output_is_two_rows(self, write_job, rows):
        write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert len(rows.get_rows_by_job("J1")) == 2

    def test_many_inputs_and_many_outputs_records_the_job_without_rows(
        self, write_job, rows, jobs
    ):
        """The job is recorded; only its index rows are refused.

        No producer emits the shape -- wandb_jobstats writes one event per output
        artifact -- so reaching it means a malformed entry. Splitting it into one
        job per target would keep the edges but fragment the run: the graph builder
        derives its run node from job_id, so one execution would render as two run
        nodes. So the decomposition refuses the rows, with a warning, but the record
        is kept: ``gb_lineage_job`` is the record of what ran, and a reindex cannot
        recover a record that was never written.
        """
        write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL), artifact("o2", "s3://b/o2")],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert [r for pg in rows.get_paged() for r in pg] == []
        assert jobs.get_job("J1") is not None

    def test_a_malformed_job_is_skipped_not_raised(self, write_job, rows, jobs):
        """A job with no endpoints is logged and yields no rows, not an error.

        One unrecordable entry must not abort the rest of a build's scan. Its
        record is still kept, like any other execution's.
        """
        write_job(job("J1", [], []), build_id="BLD", target_run_uuid="TR")
        assert [r for pg in rows.get_paged() for r in pg] == []
        assert jobs.get_job("J1") is not None

    def test_every_row_of_a_job_shares_its_job_id(self, write_job, rows):
        write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert {r.job_id for r in rows.get_rows_by_job("J1")} == {"J1"}

    def test_a_creation_records_a_terminal_source(self, write_job, rows):
        write_job(
            job("J1", [], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")
        assert len(stored) == 1
        assert stored[0].input == TERMINAL

    def test_a_deletion_records_a_terminal_target(self, write_job, rows):
        write_job(
            job("J1", [artifact("a", LH_TABLE)], []),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")
        assert len(stored) == 1
        assert stored[0].output == TERMINAL

    def test_an_undecomposable_job_is_skipped_not_raised(self, write_job, rows):
        # One unrecordable entry must not abort the rest of a build's lineage.
        write_job(
            job("", [artifact("a", LH_TABLE)], []),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert [r for pg in rows.get_paged() for r in pg] == []


class TestIdempotence:
    """Re-ingesting the same lineage must not duplicate rows."""

    def test_writing_the_same_job_twice_does_not_duplicate(self, write_job, rows):
        entry = job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)])
        for _ in range(2):
            write_job(
                entry,
                build_id="BLD",
                target_run_uuid="TR",
            )
        assert len(rows.get_rows_by_job("J1")) == 1

    def test_a_creation_row_is_not_duplicated_either(self, write_job, rows):
        # The least visible case: terminal endpoints are stored as "" rather than
        # NULL precisely so the unique index still catches them.
        entry = job("J1", [], [artifact("b", LH_MODEL)])
        for _ in range(2):
            write_job(
                entry,
                build_id="BLD",
                target_run_uuid="TR",
            )
        assert len(rows.get_rows_by_job("J1")) == 1


class TestDedupByPresence:
    """Dedup keys on ``job_id``, which for build lineage IS the target run uuid.

    ``_build_events_for_target`` stamps ``job_details.job_id = targetrun.uuid``
    (``wandb_jobstats.py:274``), so a target run and its job share one identifier.
    That is what lets the index drop its ``target_run_uuid`` column without
    weakening dedup -- and why these fixtures write a job whose id is the target
    run's rather than an unrelated one.
    """

    def test_filter_unrecorded_reports_targets_with_no_rows(self, sink):
        assert sink.filter_unrecorded({"t1", "t2"}) == {"t1", "t2"}

    def test_filter_unrecorded_drops_a_recorded_target(self, sink):
        sink._write_job(
            job("t1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="t1",
        )
        assert sink.filter_unrecorded({"t1", "t2"}) == {"t2"}

    def test_expected_counts_is_ignored(self, sink):
        # It counts one W&B run per output artifact -- a shape that need not equal
        # this sink's row count, since one such event still fans out over its
        # inputs. Honouring it would report every target unrecorded forever.
        sink._write_job(
            job("t1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="t1",
        )
        assert sink.filter_unrecorded({"t1"}, expected_counts={"t1": 999}) == set()

    def test_an_empty_candidate_set_is_empty(self, sink):
        assert sink.filter_unrecorded(set()) == set()

    def test_a_query_failure_fails_open_and_reports_the_error(self, rows):
        # Fails OPEN here (every candidate re-recorded, which is idempotent) but
        # must invoke the callback -- the reconciler is what fails closed on it.
        # Without the callback this silently duplicates work instead of skipping.
        class Failing:
            def get_jobs_by_id(self, job_ids):
                raise RuntimeError("db is down")

        seen = []
        sink = DBLineageStore(job_storage=Failing())
        result = sink.filter_unrecorded({"t1"}, on_query_error=seen.append)
        assert result == {"t1"}
        assert len(seen) == 1

    def test_a_query_failure_without_a_callback_still_does_not_raise(self):
        class Failing:
            def get_jobs_by_id(self, job_ids):
                raise RuntimeError("db is down")

        assert DBLineageStore(job_storage=Failing()).filter_unrecorded({"t1"}) == {"t1"}


class TestReleaseCounts:
    """release_id is a build_id; the UNIT is jobs, which is W&B's run shape."""

    def test_counts_one_job_per_execution_not_one_per_edge(self, sink):
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        # One execution, so 1 -- and W&B reports 1 run too (one per output), so a
        # caller's expected_count is comparable. This used to count index rows
        # (2 inputs x 1 output = 2), which could never match that.
        assert sink.count_release_ids("BLD") == 1

    def test_narrows_by_target_run(self, sink):
        for target_run in ("t1", "t2"):
            sink._write_job(
                job(
                    f"J-{target_run}",
                    [artifact("a", LH_TABLE)],
                    [artifact("b", LH_MODEL)],
                ),
                build_id="BLD",
                target_run_uuid=target_run,
            )
        assert sink.count_release_ids("BLD") == 2
        assert sink.count_release_ids("BLD", target_id="t1") == 1

    def test_an_unknown_release_is_zero(self, sink):
        assert sink.count_release_ids("nope") == 0

    def test_an_empty_release_id_is_zero(self, sink):
        assert sink.count_release_ids("") == 0

    def test_does_release_id_exist_compares_row_counts(self, sink):
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert sink.does_release_id_exist("BLD", 1) is True
        assert sink.does_release_id_exist("BLD", 5) is False


class TestRowContents:
    def test_endpoint_detail_comes_from_the_artifact_dict(self):
        """Kind and name are recorded, not derived from the URI.

        Inferring ``model`` from an ``lh://.../models/...`` path would be a second
        opinion about what an artifact is, free to diverge from the producer's.
        """
        row = _row_from_draft(
            LineageRowDraft(
                job_id="J1",
                input="lh://prod/ns/models/tbl/label",
                input_artifact={"name": "label", "artifact_type": "model"},
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert endpoint_kind(row.attributes, INPUT) == "model"
        assert endpoint_name(row.attributes, INPUT) == "label"

    def test_the_uri_is_not_repeated_inside_the_blob(self):
        """It is the row's identity; a second copy could only diverge from it."""
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", input="lh://prod/ns/tables/t"),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert "uri" not in (row.attributes.get(INPUT) or {})

    def test_none_endpoints_become_the_terminal_marker(self):
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", output="lh://prod/ns/tables/t"),
            build_id="BLD",
            target_run_uuid="TR",
        )
        # Not NULL: in SQL NULL never equals NULL, so NULL endpoints would slip
        # past the unique index.
        assert row.input == ""
        assert row.input == TERMINAL

    def test_rows_carry_the_slim_shape(self):
        """Producing system lives on the job record, not copied onto every row."""
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", input="lh://prod/ns/tables/t"),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert "origin" not in row.attributes
        assert origin_system(row.attributes) == ""

    def test_process_ids_are_carried_in_the_retrieve_group(self):
        """Carried but not indexed: they are empty for every imported source."""
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", input="lh://prod/ns/tables/t"),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert row.attributes["retrieve"] == {
            "build_id": "BLD",
            "target_run_uuid": "TR",
        }
        assert origin_id(row.attributes, "build_id") == "BLD"
        assert origin_id(row.attributes, "target_run_uuid") == "TR"

    def test_an_importer_carries_no_process_ids(self):
        """A source with no build concept is retrieved by its job id instead."""
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", input="lh://prod/ns/tables/t"),
            build_id="",
            target_run_uuid="",
        )
        assert row.attributes["retrieve"] == {"job_id": "J1"}
        assert origin_id(row.attributes, "build_id") == ""

    def test_rows_carry_what_authorization_and_the_run_node_read(self):
        """namespace and owner gate access; status and timestamps label the node."""
        row = _row_from_draft(
            LineageRowDraft(
                job_id="J1",
                input="lh://prod/ns/tables/t",
                metadata={
                    "job_name": "train",
                    "job_namespace": "space-a/proj",
                    "owner": "a@x.com",
                    "job_status": "COMPLETED",
                    "job_started_at": "2026-01-01T00:00:00Z",
                    "job_completed_at": "2026-01-01T01:00:00Z",
                    "category": "training",
                },
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert job_detail(row.attributes) == {
            "name": "train",
            "id": "J1",
            "namespace": "space-a/proj",
            "owner": "a@x.com",
            "status": "COMPLETED",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T01:00:00Z",
        }

    def test_an_unfinished_job_has_no_completed_at(self):
        row = _row_from_draft(
            LineageRowDraft(
                job_id="J1",
                input="lh://prod/ns/tables/t",
                metadata={"job_status": "RUNNING", "job_started_at": "t0"},
            ),
            build_id="",
            target_run_uuid="",
        )
        assert "completed_at" not in job_detail(row.attributes)

    def test_the_endpoints_are_the_normalized_uris(self, write_job, rows):
        """The URI is the identity, so it is normalized on the way in.

        ``hf:///org/repo`` and the browser URL for the same repo must converge, or
        one artifact becomes two disconnected halves of a graph.
        """
        write_job(
            job(
                "J1",
                [artifact("a", "s3://bkt/raw")],
                [artifact("b", "hf:///org/repo")],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")[0]
        assert stored.input == "s3://bkt/raw"
        assert stored.output == "https://huggingface.co/org/repo"

    def test_alternative_spellings_are_kept_in_the_blob(self, write_job, rows):
        """The forms normalization folds away are remembered, not lost.

        The canonical URI is the column; every other recorded spelling -- the raw
        ``uri``, the facet mirrors, and an explicit ``uri_aliases`` list such as
        the physical path behind an ``lh://`` artifact -- lands in ``alt_uris``.
        """
        lh_raw = "lh://prod/ns/models/mdl_tbl/trained/granite-dot-build"
        target = artifact("b", lh_raw)
        target["facets"] = {
            "gb-artifact-uri": lh_raw,
            "uri_aliases": ["s3://bkt/models/trained/"],
        }
        write_job(
            job("J1", [artifact("a", "hf:///org/repo")], [target]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")[0]
        assert stored.input == "https://huggingface.co/org/repo"
        assert stored.output == "lh://prod/ns/models/mdl_tbl/trained"
        assert (stored.attributes.get(INPUT) or {}).get("alt_uris", []) == [
            "hf:///org/repo"
        ]
        assert (stored.attributes.get(OUTPUT) or {}).get("alt_uris", []) == [
            lh_raw,
            "s3://bkt/models/trained/",
        ]

    def test_a_canonical_spelling_has_no_alternatives(self, write_job, rows):
        write_job(
            job("J1", [artifact("a", "s3://bkt/raw")], [artifact("b", LH_TABLE)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")[0]
        assert ALT_URIS not in stored.attributes[INPUT]
        assert ALT_URIS not in stored.attributes[OUTPUT]

    def test_carried_metadata_survives(self, write_job, rows):
        write_job(
            job(
                "J1",
                [artifact("a", LH_TABLE)],
                [artifact("b", LH_MODEL)],
                job_name="train",
                owner="alice",
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")[0]
        job_group = job_detail(stored.attributes)
        assert job_group["name"] == "train"
        assert job_group["owner"] == "alice"


class TestArtifactRegistrationRows:
    """The path with no target run (D11)."""

    def test_rows_carry_no_target_run(self):
        # build_id holds the artifact uuid so count_release_ids finds them, while
        # target_run_uuid is absent: there is no target run.
        row = _row_from_draft(
            LineageRowDraft(job_id="ART-UUID", input="lh://prod/ns/tables/t"),
            build_id="ART-UUID",
            target_run_uuid="",
        )
        assert origin_id(row.attributes, "target_run_uuid") == ""
        assert origin_id(row.attributes, "build_id") == "ART-UUID"
        # job_id still identifies the rows, so the composite unique keeps
        # protecting them without depending on a target run.
        assert row.job_id == "ART-UUID"


class TestWiring:
    def test_the_sink_records_centralized_lineage(self, sink):
        # lineage-watch exits immediately for a store reporting False, which would
        # leave the index permanently empty.
        assert sink.records_centralized_lineage is True

    def test_it_implements_the_store_interface(self, sink):
        from gbserver.lineage.jobstats import ILineageStore

        assert isinstance(sink, ILineageStore)

    def test_building_needs_no_wandb_instance(self):
        # The shared builders are pure; reusing them must not require wandb to be
        # installed or a W&B store to be constructed.
        import inspect

        from gbserver.lineage import db_jobstats

        source = inspect.getsource(db_jobstats)
        assert "WandBLineageStore()" not in source


class TestJobNormalization:
    """The builders nest the job identity; decomposition needs it at the top."""

    def test_job_id_is_lifted_out_of_job_details(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        # This is what the shared builders actually emit: sources/targets are
        # mirrored to the top level, but job_id lives under job_details. Reading
        # only the top level yielded job_id=None, which made to_lineage_rows reject
        # EVERY event -- the index stayed empty while recording looked successful.
        event = {
            "sources": [],
            "targets": [],
            "job_details": {"job_id": "TR-UUID", "owner": "alice"},
        }
        normalized = _normalized_job(event)
        assert normalized["job_id"] == "TR-UUID"
        assert normalized["owner"] == "alice"

    def test_a_top_level_job_id_is_preserved(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        event = {"job_id": "J1", "sources": [], "targets": []}
        assert _normalized_job(event)["job_id"] == "J1"

    def test_an_event_with_no_job_details_is_unchanged(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        event = {"job_id": "J1", "sources": [], "targets": []}
        assert _normalized_job(event)["job_id"] == "J1"


class TestNamespacePropagation:
    """``job_namespace`` must survive the write path, or authorization eats the graph.

    The read path splits it on the first ``/`` to recover the space and prunes nodes
    the caller cannot see, failing closed when it is absent. So a row that loses it is
    not merely missing a label: every node built from it disappears from every graph,
    for every caller -- authorization behaving correctly on absent provenance, which
    is indistinguishable from an empty index.

    It is nested a third way in the builder's event (under ``job.namespace``, not
    ``job_details``), which is exactly how it got dropped.
    """

    def test_the_namespace_is_lifted_out_of_the_job_block(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        event = {
            "job": {"namespace": "my-space/build-x", "name": "tgt"},
            "job_details": {"job_id": "TR"},
        }
        assert _normalized_job(event)["job_namespace"] == "my-space/build-x"

    def test_an_explicit_top_level_namespace_is_not_overwritten(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        event = {
            "job": {"namespace": "from-block"},
            "job_namespace": "from-top-level",
            "job_details": {"job_id": "TR"},
        }
        assert _normalized_job(event)["job_namespace"] == "from-top-level"

    def test_a_missing_job_block_does_not_raise(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        assert "job_namespace" not in _normalized_job({"job_details": {"job_id": "T"}})

    def test_the_namespace_reaches_the_stored_row(self, write_job, rows):
        """End to end: the blob's job group carries it, so the graph can authorize."""
        from gbserver.lineage.attributes import job_detail

        write_job(
            {
                "sources": [artifact("a", LH_TABLE)],
                "targets": [artifact("b", LH_MODEL)],
                "job": {"namespace": "my-space/build-x", "name": "tgt"},
                "job_details": {"job_id": "TR", "owner": "someone"},
            },
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("TR")[0]
        assert job_detail(stored.attributes)["namespace"] == "my-space/build-x"


class TestJobTags:
    """Every recorded job is tagged on its rows, and a release count queries them."""

    @pytest.fixture(name="tagged_sink")
    def tagged_sink_fixture(self, rows):
        suffix = uuid_module.uuid4().hex[:8]
        factory = SqliteStorageFactory()
        # Tags live on the rows' ``attributes.job.tags``; the row storage answers
        # them. The store records the job, the indexer derives the tagged rows, so
        # both halves are kept here and _write runs them in order.
        self.tags = rows
        self.jobs = factory.create_lineage_job_storage(table_name=f"t_job_{suffix}")
        self.indexer = LineageRowIndexer(storage=rows)
        return DBLineageStore(job_storage=self.jobs)

    def _write(self, sink, job_id, target_run, extra_tags=None, facet_tags=None):
        entry = job(job_id, [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)])
        if facet_tags:
            entry["run"] = {"facets": {"tags": facet_tags}}
        sink._write_job(
            entry,
            build_id="BLD",
            target_run_uuid=target_run,
            extra_tags=extra_tags,
        )
        record = self.jobs.get_job(job_id)
        if record is not None:
            self.indexer.index_job_record(record)

    def test_ids_facets_and_extra_tags_are_stored(self, tagged_sink):
        self._write(
            tagged_sink,
            "J1",
            "t1",
            extra_tags=["team=nlp"],
            facet_tags={"username": "alice", "space_name": "sp", "empty": ""},
        )
        assert self.tags.get_tags(["J1"])["J1"] == [
            "build_id=BLD",
            "space_name=sp",
            "target_run_uuid=t1",
            "team=nlp",
            "username=alice",
        ]

    def test_unstorable_tags_are_dropped_not_truncated(self, tagged_sink):
        self._write(tagged_sink, "J1", "t1", extra_tags=["", "x" * 300, "ok"])
        assert "ok" in self.tags.get_tags(["J1"])["J1"]
        assert not any(len(tag) > 256 for tag in self.tags.get_tags(["J1"])["J1"])

    def test_re_recording_does_not_duplicate_tags(self, tagged_sink):
        self._write(tagged_sink, "J1", "t1", extra_tags=["team=nlp"])
        self._write(tagged_sink, "J1", "t1", extra_tags=["team=nlp", "new=1"])
        tags = self.tags.get_tags(["J1"])["J1"]
        assert tags.count("team=nlp") == 1
        assert "new=1" in tags

    def test_release_count_counts_jobs_not_rows(self, tagged_sink):
        """The unit is one job per execution -- W&B's run shape, so comparable.

        It used to count index rows (one per input/output pair), which could never
        equal a caller's W&B-shaped expected_count. Each job here has 1 input and
        1 output, so rows and jobs would agree; the narrowed counts below are what
        distinguishes them.
        """
        self._write(tagged_sink, "J1", "t1")
        self._write(tagged_sink, "J2", "t2")
        assert tagged_sink.count_release_ids("BLD") == 2
        assert tagged_sink.count_release_ids("BLD", target_id="t1") == 1
        assert tagged_sink.count_release_ids("BLD", target_id="t3") == 0

    def test_the_read_service_combines_a_uri_with_a_job(self, tagged_sink, rows):
        """AND, not OR: the filters narrow one another."""
        from gbserver.lineage.db_service import DBLineageService

        self._write(tagged_sink, "J1", "t1", extra_tags=["team=nlp"])
        self._write(tagged_sink, "J2", "t2", extra_tags=["team=vision"])
        service = DBLineageService(storage=rows)
        both = service.list_jobs(uri=LH_TABLE)
        assert [job["job_id"] for job in both["jobs"]] == ["J1", "J2"]
        assert service.list_jobs(uri=LH_MODEL, job_id="J2")["total"] == 1
        assert service.list_jobs(uri="s3://b/elsewhere", job_id="J2")["total"] == 0

    def test_the_read_service_lists_nothing_with_no_filter(self, tagged_sink, rows):
        """An unfiltered listing would aggregate the whole index; the route 422s."""
        from gbserver.lineage.db_service import DBLineageService

        self._write(tagged_sink, "J1", "t1")
        service = DBLineageService(storage=rows)
        assert service.list_jobs()["total"] == 0


class TestJobsTouchingInSQL:
    """The SQL overrides that keep a lone-``uri`` listing off the rows.

    Counted in distinct jobs: a many-input job repeats its output on every row, so
    a row count would overstate it.
    """

    def _write(self, write_job, job_id, sources, targets):
        write_job(job(job_id, sources, targets), build_id="BLD", target_run_uuid=job_id)

    def test_a_job_repeating_an_artifact_counts_once(self, write_job, rows):
        self._write(
            write_job,
            "J1",
            [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
            [artifact("o1", LH_MODEL)],
        )
        assert len(rows.get_rows_by_job("J1")) == 2
        assert rows.count_jobs_touching(LH_MODEL) == 1
        assert rows.get_job_ids_touching(LH_MODEL, limit=10, offset=0) == ["J1"]

    def test_self_loop_counts_only_in_place_rewrites_in_sql(self, write_job, rows):
        self._write(
            write_job, "S1", [artifact("t", LH_TABLE)], [artifact("t", LH_TABLE)]
        )
        self._write(
            write_job, "S2", [artifact("t", LH_TABLE)], [artifact("t", LH_TABLE)]
        )
        self._write(
            write_job, "X", [artifact("t", LH_TABLE)], [artifact("o", LH_MODEL)]
        )
        assert rows.count_jobs_touching(LH_TABLE) == 3
        assert rows.count_jobs_touching(LH_TABLE, self_loop=True) == 2
        assert rows.get_job_ids_touching(LH_TABLE, 10, 0, self_loop=True) == [
            "S1",
            "S2",
        ]
        assert rows.filter_jobs_touching(LH_TABLE, ["S1", "X"], self_loop=True) == {
            "S1"
        }
        assert rows.count_jobs_touching(LH_TABLE, output=LH_MODEL) == 1
        assert rows.get_job_ids_touching(LH_TABLE, 10, 0, output=LH_MODEL) == ["X"]
        assert rows.filter_jobs_touching(LH_TABLE, ["S1", "X"], output=LH_MODEL) == {
            "X"
        }

    def test_both_directions_are_counted_and_paged(self, write_job, rows):
        for i in range(5):
            self._write(
                write_job, f"J{i}", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]
            )
        self._write(
            write_job, "K", [artifact("b", LH_MODEL)], [artifact("c", "s3://b/c")]
        )
        assert rows.count_jobs_touching(LH_MODEL) == 6
        pages = [rows.get_job_ids_touching(LH_MODEL, limit=4, offset=o) for o in (0, 4)]
        assert pages == [["J0", "J1", "J2", "J3"], ["J4", "K"]]

    def test_filtering_a_job_set_by_artifact(self, write_job, rows):
        self._write(
            write_job, "J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]
        )
        self._write(
            write_job, "J2", [artifact("b", LH_MODEL)], [artifact("c", "s3://b/c")]
        )
        assert rows.filter_jobs_touching(LH_TABLE, ["J1", "J2"]) == {"J1"}
        assert rows.filter_jobs_touching(LH_MODEL, ["J1", "J2", "nope"]) == {"J1", "J2"}

    def test_tags_filter_in_sql_alone_and_with_an_artifact(self, rows):
        """Matched on ``attributes.job.tags`` through the JSON operators, not in Python."""
        for job_id, tags in (
            ("J1", {"team": "nlp", "nightly": ""}),
            ("J2", {"team": "nlp"}),
            ("J3", {"team": "vision", "a.b": "x'y"}),
        ):
            rows.add(
                StoredLineageRow(
                    job_id=job_id,
                    input=LH_TABLE,
                    output=LH_MODEL if job_id != "J2" else "s3://b/o",
                    attributes={"job": {"tags": tags}},
                )
            )
        assert rows.count_jobs_touching("", tags=["team=nlp"]) == 2
        assert rows.get_job_ids_touching("", 10, 0, tags=["team=nlp"]) == ["J1", "J2"]
        assert rows.count_jobs_touching(LH_MODEL, tags=["team=nlp"]) == 1
        assert rows.count_jobs_touching("", tags=["team=nlp", "nightly"]) == 1
        assert rows.count_jobs_touching("", tags=["team"]) == 0
        assert rows.get_job_ids_touching("", 10, 0, tags=["a.b=x'y"]) == ["J3"]
        assert rows.filter_jobs_touching("", ["J1", "J3"], tags=["team=nlp"]) == {"J1"}
        assert rows.filter_jobs_touching(
            LH_TABLE, ["J1", "J2"], output=LH_MODEL, tags=["team=nlp"]
        ) == {"J1"}

    def test_a_malformed_blob_does_not_fail_a_tag_listing(self, rows):
        rows.add(
            StoredLineageRow(
                job_id="J1",
                input=LH_TABLE,
                output=LH_MODEL,
                attributes={"job": {"tags": {"team": "nlp"}}},
            )
        )
        from sqlalchemy import text

        with rows._engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {rows.table_name} (uuid, job_id, input, output, "
                    "job_store, recorded_at, json) VALUES "
                    "('u-bad', 'BAD', 'x', 'y', 'other', '', 'not json')"
                )
            )
        assert rows.count_jobs_touching("", tags=["team=nlp"]) == 1

    def test_a_terminal_or_empty_uri_touches_nothing(self, rows):
        assert rows.count_jobs_touching("") == 0
        assert rows.count_jobs_touching("", self_loop=True) == 0
        assert rows.get_job_ids_touching("", limit=10, offset=0) == []


class TestJobRecordBackfill:
    """The job record alone decides whether an execution is recorded.

    It is the only thing this store writes, so presence of a record is the whole
    answer. Index rows say nothing about it: they are derived afterwards by the
    indexer, which owns them.
    """

    @staticmethod
    def _sink(jobs: dict):
        from unittest.mock import MagicMock

        job_storage = MagicMock()
        job_storage.get_jobs_by_id.side_effect = lambda ids: {
            i: jobs[i] for i in ids if i in jobs
        }
        return DBLineageStore(job_storage=job_storage)

    def test_a_job_with_no_record_is_not_recorded_by_self(self):
        assert self._sink({}).recorded_by_self(["J"]) == set()

    def test_a_job_this_system_recorded_is_recorded_by_self(self):
        from types import SimpleNamespace

        from gbserver.lineage.db_jobstats import SOURCE_SYSTEM

        sink = self._sink({"J": SimpleNamespace(source_system=SOURCE_SYSTEM)})
        assert sink.recorded_by_self(["J"]) == {"J"}

    def test_has_job_record_reads_the_job_table(self):
        sink = self._sink({"J": object()})
        assert sink._has_job_record("J")
        assert not sink._has_job_record("OTHER")


class TestRejectedJobs:
    """A job the table refuses is not rebuilt and re-refused on every pass."""

    def test_a_rejected_job_counts_as_recorded(self, sink, jobs):
        long_id = "J" * 300
        sink._write_job(
            job(long_id, [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert jobs.get_job(long_id) is None
        assert sink.filter_unrecorded({long_id, "OTHER"}) == {"OTHER"}


class TestCommaTags:
    def test_a_tag_with_a_comma_is_dropped(self):
        from gbserver.lineage.db_jobstats import is_storable_tag

        assert is_storable_tag("team=nlp")
        assert not is_storable_tag("labels=a,b")


class TestTagSearchEscapesLike:
    """``%`` and ``_`` in a tag are literal, so they cannot widen the SQL filter."""

    def _job(self, job_id, tags):
        from gbserver.storage.stored_lineage_job import StoredLineageJob

        return StoredLineageJob(job_id=job_id, tags=tags)

    def test_percent_matches_only_itself(self, jobs):
        jobs.add(self._job("J1", ["pct=100%"]))
        jobs.add(self._job("J2", ["other=1"]))
        total, found = jobs.search_by_tags(["pct=100%"])
        assert total == 1
        assert [j.job_id for j in found] == ["J1"]
        assert jobs.search_by_tags(["%"])[0] == 0

    def test_underscore_is_not_a_wildcard(self, jobs):
        jobs.add(self._job("J1", ["run_1"]))
        jobs.add(self._job("J2", ["runX1"]))
        assert [j.job_id for j in jobs.search_by_tags(["run_1"])[1]] == ["J1"]


class TestUpsertRow:
    """The row merge: a lost race merges, and a stale terminal is repaired."""

    def _row(self, job_id, source, target, **attributes):
        from gbserver.storage.stored_lineage_row import StoredLineageRow

        return StoredLineageRow(
            job_id=job_id, input=source, output=target, attributes=attributes
        )

    def test_a_lost_race_merges_into_the_winner(self, rows, monkeypatch):
        from gbserver.lineage.merge import UPDATED, upsert_row

        rows.add(self._row("J1", "a", "b", winner={"x": 1}))
        # The racing writer read before the winner's add landed.
        real = type(rows).get_rows_by_job
        reads = iter([[]])
        monkeypatch.setattr(
            type(rows),
            "get_rows_by_job",
            lambda self, job_id: next(reads, None) or real(self, job_id),
        )
        assert upsert_row(rows, self._row("J1", "a", "b", loser={"y": 2})) == UPDATED
        (stored,) = real(rows, "J1")
        assert stored.attributes == {"winner": {"x": 1}, "loser": {"y": 2}}

    def test_a_stale_terminal_is_removed_by_the_next_write(self, rows):
        from gbserver.lineage.merge import UNCHANGED, upsert_row

        # The state a crash between add and delete leaves: both rows.
        rows.add(self._row("J1", "a", "b"))
        rows.add(self._row("J1", TERMINAL, "b"))
        assert upsert_row(rows, self._row("J1", "a", "b")) == UNCHANGED
        assert [(r.input, r.output) for r in rows.get_rows_by_job("J1")] == [("a", "b")]

    def test_a_real_edge_supersedes_and_keeps_the_terminals_attributes(self, rows):
        from gbserver.lineage.merge import ADDED, upsert_row

        rows.add(self._row("J1", TERMINAL, "b", seen={"lh": 1}))
        assert upsert_row(rows, self._row("J1", "a", "b")) == ADDED
        (stored,) = rows.get_rows_by_job("J1")
        assert (stored.input, stored.attributes) == ("a", {"seen": {"lh": 1}})


class TestFinishedStatusReplacesInProgress:
    def _job(self, status, completed_at=""):
        from gbserver.storage.stored_lineage_job import StoredLineageJob

        job_group = {"status": status}
        if completed_at:
            job_group["completed_at"] = completed_at
        return StoredLineageJob(
            job_id="J1", status=status, attributes={"job": job_group}
        )

    def test_running_moves_on_to_success(self, jobs):
        from gbserver.lineage.merge import UPDATED, upsert_job

        upsert_job(jobs, self._job("RUNNING"))
        assert upsert_job(jobs, self._job("SUCCESS", "2026-01-01T00:00:00Z")) == UPDATED
        stored = jobs.get_job("J1")
        assert stored.status == "SUCCESS"
        assert stored.attributes["job"] == {
            "status": "SUCCESS",
            "completed_at": "2026-01-01T00:00:00Z",
        }

    def test_a_finished_status_is_kept(self, jobs):
        from gbserver.lineage.merge import UNCHANGED, upsert_job

        upsert_job(jobs, self._job("FAILED"))
        assert upsert_job(jobs, self._job("SUCCESS")) == UNCHANGED
        assert jobs.get_job("J1").status == "FAILED"

    def test_an_in_progress_status_does_not_replace_a_finished_one(self, jobs):
        from gbserver.lineage.merge import upsert_job

        upsert_job(jobs, self._job("finished"))
        upsert_job(jobs, self._job("running"))
        assert jobs.get_job("J1").status == "finished"


class TestMergeStampsRecordedAtAtomically:
    """A merge and its recorded_at bump are one write: if it fails, neither is in."""

    def test_a_merge_moves_recorded_at_with_its_content(self, sink, jobs):
        import time

        entry = job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)])
        sink._write_job(entry, build_id="BLD", target_run_uuid="t1")
        before = jobs.get_job("J1")
        time.sleep(0.01)
        sink._write_job(entry, build_id="BLD", target_run_uuid="t1", extra_tags=["k=v"])
        after = jobs.get_job("J1")
        assert "k=v" in after.tags
        assert after.recorded_at > before.recorded_at

    def test_a_failed_merge_leaves_the_record_as_it_was(self, sink, jobs, monkeypatch):
        entry = job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)])
        sink._write_job(entry, build_id="BLD", target_run_uuid="t1")
        before = jobs.get_job("J1")

        def fail(db_item):
            raise ValueError("boom")

        monkeypatch.setattr(jobs, "_on_update_row", fail)
        with pytest.raises(ValueError):
            sink._write_job(
                entry, build_id="BLD", target_run_uuid="t1", extra_tags=["k=v"]
            )
        after = jobs.get_job("J1")
        assert after.tags == before.tags
        assert after.recorded_at == before.recorded_at
