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
    INPUT,
    endpoint_kind,
    endpoint_name,
    job_detail,
    origin_id,
    origin_system,
)
from gbserver.lineage.db_jobstats import DBLineageStore, _row_from_draft
from gbserver.lineage.decompose import LineageRowDraft
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="rows")
def rows_fixture():
    """A lineage row storage on a table unique to this test."""
    table = f"t_sink_{uuid_module.uuid4().hex[:8]}"
    return SqliteStorageFactory().create_lineage_row_storage(table_name=table)


@pytest.fixture(name="sink")
def sink_fixture(rows):
    return DBLineageStore(storage=rows)


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

    def test_one_input_one_output_is_one_row(self, sink, rows):
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert len(rows.get_rows_by_job("J1")) == 1

    def test_two_inputs_one_output_is_two_rows(self, sink, rows):
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert len(rows.get_rows_by_job("J1")) == 2

    def test_many_inputs_and_many_outputs_records_nothing(self, sink, rows):
        """The guard refuses this job, and _write_job skips rather than raises.

        No producer emits the shape -- wandb_jobstats writes one event per output
        artifact -- so reaching it means a malformed entry, and one such entry must
        not abort the rest of a build's scan. Splitting it into one job per target
        would keep the edges but fragment the run: the graph builder derives its run
        node from job_id (graph_builder.py:223), so one execution would render as
        two run nodes. Refusing is the honest outcome; the warning carries the
        reason so the loss is diagnosable.
        """
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL), artifact("o2", "s3://b/o2")],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert [r for pg in rows.get_paged() for r in pg] == []

    def test_a_malformed_job_is_skipped_not_raised(self, sink, rows):
        """A job the guard cannot rescue is logged and skipped, not fatal.

        One unrecordable entry must not abort the rest of a build's scan.
        """
        sink._write_job(job("J1", [], []), build_id="BLD", target_run_uuid="TR")
        assert [r for pg in rows.get_paged() for r in pg] == []

    def test_every_row_of_a_job_shares_its_job_id(self, sink, rows):
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert {r.job_id for r in rows.get_rows_by_job("J1")} == {"J1"}

    def test_a_creation_records_a_terminal_source(self, sink, rows):
        sink._write_job(
            job("J1", [], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")
        assert len(stored) == 1
        assert stored[0].is_creation()

    def test_a_deletion_records_a_terminal_target(self, sink, rows):
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], []),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")
        assert len(stored) == 1
        assert stored[0].is_deletion()

    def test_an_undecomposable_job_is_skipped_not_raised(self, sink, rows):
        # One unrecordable entry must not abort the rest of a build's lineage.
        sink._write_job(
            job("", [artifact("a", LH_TABLE)], []),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert [r for pg in rows.get_paged() for r in pg] == []


class TestIdempotence:
    """Re-ingesting the same lineage must not duplicate rows."""

    def test_writing_the_same_job_twice_does_not_duplicate(self, sink, rows):
        entry = job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)])
        for _ in range(2):
            sink._write_job(
                entry,
                build_id="BLD",
                target_run_uuid="TR",
            )
        assert len(rows.get_rows_by_job("J1")) == 1

    def test_a_creation_row_is_not_duplicated_either(self, sink, rows):
        # The least visible case: terminal endpoints are stored as "" rather than
        # NULL precisely so the unique index still catches them.
        entry = job("J1", [], [artifact("b", LH_MODEL)])
        for _ in range(2):
            sink._write_job(
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
            def get_recorded_target_runs(self, target_run_uuids):
                raise RuntimeError("db is down")

        seen = []
        sink = DBLineageStore(storage=Failing())
        result = sink.filter_unrecorded({"t1"}, on_query_error=seen.append)
        assert result == {"t1"}
        assert len(seen) == 1

    def test_a_query_failure_without_a_callback_still_does_not_raise(self):
        class Failing:
            def get_recorded_target_runs(self, target_run_uuids):
                raise RuntimeError("db is down")

        assert DBLineageStore(storage=Failing()).filter_unrecorded({"t1"}) == {"t1"}


class TestReleaseCounts:
    """release_id is a build_id; the UNIT is rows, not W&B runs."""

    def test_counts_rows_for_a_build(self, sink):
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        # Two inputs x one output = 2 rows. W&B would report 1 run (one per
        # output), which is exactly the shape mismatch the docstring warns about.
        assert sink.count_release_ids("BLD") == 2

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
        assert row.is_creation()

    def test_rows_record_the_producing_system(self):
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", input="lh://prod/ns/tables/t"),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert origin_system(row.attributes) == "granite.build"

    def test_process_ids_are_carried_in_the_origin_group(self):
        """Carried but not indexed: they are empty for every imported source."""
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", input="lh://prod/ns/tables/t"),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert origin_id(row.attributes, "build_id") == "BLD"
        assert origin_id(row.attributes, "target_run_uuid") == "TR"

    def test_an_importer_carries_no_process_ids(self):
        """A source with no build concept writes no empty id keys.

        Omitting the group lets a reader tell "not recorded" from "recorded empty".
        """
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", input="lh://prod/ns/tables/t"),
            build_id="",
            target_run_uuid="",
            source_system="lakehouse",
        )
        assert "ids" not in row.attributes["origin"]
        assert origin_system(row.attributes) == "lakehouse"

    def test_the_endpoints_are_the_normalized_uris(self, sink, rows):
        """The URI is the identity, so it is normalized on the way in.

        ``hf:///org/repo`` and the browser URL for the same repo must converge, or
        one artifact becomes two disconnected halves of a graph.
        """
        sink._write_job(
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
        assert stored.output == "hf://huggingface.co/models/org/repo"

    def test_carried_metadata_survives(self, sink, rows):
        sink._write_job(
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

    def test_the_namespace_reaches_the_stored_row(self, sink, rows):
        """End to end: the blob's job group carries it, so the graph can authorize."""
        from gbserver.lineage.attributes import job_detail

        sink._write_job(
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
    """Every recorded job is tagged, and a release count uses the tag index."""

    @pytest.fixture(name="tagged_sink")
    def tagged_sink_fixture(self, rows):
        suffix = uuid_module.uuid4().hex[:8]
        factory = SqliteStorageFactory()
        self.tags = factory.create_lineage_job_tag_storage(table_name=f"t_tag_{suffix}")
        return DBLineageStore(
            storage=rows,
            job_storage=factory.create_lineage_job_storage(
                table_name=f"t_job_{suffix}"
            ),
            tag_storage=self.tags,
        )

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

    def test_filter_by_any_tag(self, tagged_sink):
        self._write(tagged_sink, "J1", "t1", extra_tags=["team=nlp"])
        self._write(tagged_sink, "J2", "t2", extra_tags=["team=vision"])
        assert self.tags.get_job_ids_by_tags(["team=nlp"]) == {"J1"}
        assert self.tags.get_job_ids_by_tags(["build_id=BLD"]) == {"J1", "J2"}

    def test_release_count_matches_the_scan(self, tagged_sink, rows):
        self._write(tagged_sink, "J1", "t1")
        self._write(tagged_sink, "J2", "t2")
        untagged = DBLineageStore(storage=rows)  # no tag storage: scans
        for target in (None, "t1", "t2", "t3"):
            assert tagged_sink.count_release_ids(
                "BLD", target_id=target
            ) == untagged.count_release_ids("BLD", target_id=target)
        assert tagged_sink.count_release_ids("BLD") == 2
        assert tagged_sink.count_release_ids("BLD", target_id="t1") == 1

    def test_the_read_service_lists_jobs_by_tag(self, tagged_sink, rows):
        from gbserver.lineage.db_service import DBLineageService

        self._write(tagged_sink, "J1", "t1", extra_tags=["team=nlp"])
        self._write(tagged_sink, "J2", "t2", extra_tags=["team=vision"])
        service = DBLineageService(
            storage=rows,
            job_storage=tagged_sink.job_storage,
            tag_storage=self.tags,
        )
        result = service.list_jobs_by_tags(["build_id=BLD"], required_tags=["team=nlp"])
        assert result["total"] == 1
        assert result["jobs"][0]["job_id"] == "J1"
        assert "team=nlp" in result["jobs"][0]["tags"]

        paged = service.list_jobs_by_tags(["build_id=BLD"], limit=1, offset=1)
        assert paged["total"] == 2
        assert [job["job_id"] for job in paged["jobs"]] == ["J2"]
