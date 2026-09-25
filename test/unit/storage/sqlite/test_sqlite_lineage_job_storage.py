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

"""Storage-level tests for lineage job records against SQLite.

These exercise what only a real table can prove: that the declared indexes and the
unique on ``job_id`` actually exist (``__create_unique_indexes`` only *warns* on
failure, so a clean log is not evidence), that every promoted column is text, and
that the four large payloads this table exists to hold survive a round trip intact.
"""

import os
import sqlite3
import uuid as uuid_module

import pytest

from gbserver.lineage.attributes import PAYLOAD, build_job_attributes, payload_detail
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_lineage_job import StoredLineageJob

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)

# A step config big enough that storing it per lineage row would be the problem this
# table solves: a job with 3 inputs and 2 outputs would hold six copies.
BIG_STEP_CONFIG = {
    "steps": [{"name": f"step-{i}", "args": ["--flag"] * 20} for i in range(12)]
}


@pytest.fixture(name="storage")
def storage_fixture():
    """A lineage job storage on a table unique to this test."""
    table = f"t_ljob_{uuid_module.uuid4().hex[:8]}"
    return SqliteStorageFactory().create_lineage_job_storage(table_name=table)


def job(
    job_id: str = "J",
    job_namespace: str = "my-space/my-build",
    space_name: str = "my-space",
    owner: str = "alice",
    status: str = "SUCCEEDED",
    started_at: str = "2026-01-01 00:00:00",
    source_system: str = "granite.build",
    **kwargs,
) -> StoredLineageJob:
    """A job record, with its blob assembled the way the sink assembles it."""
    if "attributes" in kwargs:
        attributes = dict(kwargs.pop("attributes") or {})
    else:
        attributes = build_job_attributes(
            job_metadata=kwargs.pop("job_metadata", None),
            source_system=source_system,
            ids=kwargs.pop("ids", None),
        )
    assert not kwargs, f"unhandled job() keywords: {sorted(kwargs)}"
    return StoredLineageJob(
        job_id=job_id,
        job_namespace=job_namespace,
        space_name=space_name,
        owner=owner,
        source_system=source_system,
        status=status,
        started_at=started_at,
        attributes=attributes,
    )


class TestLookup:
    """Every query is a single indexed lookup keyed on job_id."""

    def test_get_job_returns_the_record(self, storage):
        storage.add(job(job_id="J1"))
        found = storage.get_job("J1")
        assert found is not None
        assert found.job_id == "J1"

    def test_get_job_returns_none_when_absent(self, storage):
        storage.add(job(job_id="J1"))
        assert storage.get_job("nope") is None

    def test_empty_job_id_queries_nothing(self, storage):
        storage.add(job(job_id="J1"))
        assert storage.get_job("") is None
        assert storage.get_jobs([]) == []
        assert storage.has_job("") is False

    def test_get_jobs_batches_a_whole_set(self, storage):
        for i in range(5):
            storage.add(job(job_id=f"J{i}"))
        found = storage.get_jobs(["J0", "J2", "J4"])
        assert sorted(j.job_id for j in found) == ["J0", "J2", "J4"]

    def test_get_jobs_drops_empties_and_duplicates(self, storage):
        storage.add(job(job_id="J1"))
        found = storage.get_jobs(["J1", "J1", "", "J1"])
        assert [j.job_id for j in found] == ["J1"]

    def test_get_jobs_omits_ids_with_no_record(self, storage):
        """A row can precede or outlive its job record; a caller enriching rows it
        already holds must not get an error for the gap."""
        storage.add(job(job_id="J1"))
        found = storage.get_jobs(["J1", "missing"])
        assert [j.job_id for j in found] == ["J1"]

    def test_get_jobs_by_id_keys_the_batch(self, storage):
        storage.add(job(job_id="J1"))
        storage.add(job(job_id="J2"))
        by_id = storage.get_jobs_by_id(["J1", "J2", "missing"])
        assert sorted(by_id) == ["J1", "J2"]
        assert by_id["J1"].job_id == "J1"

    def test_has_job_is_presence_not_count(self, storage):
        storage.add(job(job_id="J1"))
        assert storage.has_job("J1") is True
        assert storage.has_job("J2") is False


class TestIdempotence:
    """One record per execution; re-recording is a no-op, not a duplicate."""

    def test_the_same_job_id_cannot_be_stored_twice(self, storage):
        storage.add(job(job_id="J1"))
        with pytest.raises(Exception):
            storage.add(job(job_id="J1"))

    def test_differing_attributes_do_not_admit_a_duplicate(self, storage):
        """The unique is on job_id alone, so a second record for one execution is a
        duplicate however much its blob differs."""
        storage.add(job(job_id="J1", job_metadata={"job_name": "first"}))
        with pytest.raises(Exception):
            storage.add(job(job_id="J1", job_metadata={"job_name": "second"}))
        assert len(storage.get_jobs(["J1"])) == 1

    def test_different_jobs_coexist(self, storage):
        storage.add(job(job_id="J1"))
        storage.add(job(job_id="J2"))
        assert len(storage.get_jobs(["J1", "J2"])) == 2


class TestPayloads:
    """The four large payloads are why this table exists."""

    def test_all_four_payloads_survive_a_round_trip(self, storage):
        storage.add(
            job(
                job_id="J1",
                job_metadata={
                    "job_input_params": BIG_STEP_CONFIG,
                    "execution_stats": {"gpu_hours": 12.5},
                    "job_output_stats": {"rows": 1000},
                    "source_code_details": {"repo": "git://example/repo"},
                },
            )
        )
        payload = payload_detail(storage.get_job("J1").attributes)
        assert payload["job_input_params"] == BIG_STEP_CONFIG
        assert payload["execution_stats"] == {"gpu_hours": 12.5}
        assert payload["job_output_stats"] == {"rows": 1000}
        assert payload["source_code_details"] == {"repo": "git://example/repo"}

    def test_a_job_with_no_payloads_omits_the_group(self, storage):
        """An omitted group lets a reader tell "not recorded" from "recorded
        empty"."""
        storage.add(job(job_id="J1", job_metadata={"job_name": "n"}))
        assert PAYLOAD not in storage.get_job("J1").attributes
        assert payload_detail(storage.get_job("J1").attributes) == {}

    def test_light_job_detail_and_origin_ids_survive(self, storage):
        storage.add(
            job(
                job_id="J1",
                job_metadata={"job_name": "train", "job_type": "BUILD"},
                ids={"build_id": "b1", "target_run_uuid": "t1"},
            )
        )
        attributes = storage.get_job("J1").attributes
        assert attributes["job"]["name"] == "train"
        assert attributes["origin"]["ids"] == {"build_id": "b1", "target_run_uuid": "t1"}


class TestFields:
    """Every promoted column survives storage as written."""

    def test_all_columns_survive_storage(self, storage):
        storage.add(
            job(
                job_id="J1",
                job_namespace="sp/bd",
                space_name="sp",
                owner="bob",
                status="FAILED",
                started_at="2026-02-03 04:05:06",
                source_system="lakehouse",
            )
        )
        found = storage.get_job("J1")
        assert found.job_namespace == "sp/bd"
        assert found.space_name == "sp"
        assert found.owner == "bob"
        assert found.status == "FAILED"
        assert found.started_at == "2026-02-03 04:05:06"
        assert found.source_system == "lakehouse"

    def test_a_source_without_a_space_stores_empties(self, storage):
        """Lakehouse has no notion of a space or an owner; the columns hold "" rather
        than NULL so they behave under an IN like every other value."""
        storage.add(
            job(job_id="J1", space_name="", owner="", job_namespace="", started_at="")
        )
        found = storage.get_job("J1")
        assert found.space_name == ""
        assert found.owner == ""
        assert found.job_namespace == ""

    def test_the_model_default_for_attributes_is_an_empty_dict(self, storage):
        storage.add(StoredLineageJob(job_id="J1"))
        assert storage.get_job("J1").attributes == {}

    def test_started_at_is_stored_verbatim(self, storage):
        """The source's own string form, never rewritten to UTC."""
        storage.add(job(job_id="J1", started_at="2026-01-02 03:04:05.678901"))
        assert storage.get_job("J1").started_at == "2026-01-02 03:04:05.678901"

    def test_attributes_are_not_queryable(self, storage):
        """The blob is Text, so nothing inside it can be a predicate.

        Asserted on the column list rather than by issuing a doomed query: a bad
        predicate goes through the retrying query path, which sleeps between attempts
        instead of failing fast.
        """
        storage.add(job(job_id="J1", job_metadata={"job_name": "findme"}))
        columns = storage.get_column_names()
        assert "job_name" not in columns
        assert "attributes" not in columns


class TestScoping:
    """space_name and owner are indexed because they are how a listing is scoped."""

    def test_jobs_are_filtered_by_space(self, storage):
        storage.add(job(job_id="J1", space_name="a"))
        storage.add(job(job_id="J2", space_name="b"))
        found = storage.get_by_where({"space_name": "a"})
        assert [j.job_id for j in found] == ["J1"]

    def test_space_accepts_a_batch(self, storage):
        """An access filter resolves the caller's spaces once, then narrows in SQL --
        which needs the IN clause get_by_where builds for string columns."""
        for i, space in enumerate(["a", "b", "c"]):
            storage.add(job(job_id=f"J{i}", space_name=space))
        found = storage.get_by_where({"space_name": ["a", "c"]})
        assert sorted(j.space_name for j in found) == ["a", "c"]

    def test_jobs_are_filtered_by_owner(self, storage):
        storage.add(job(job_id="J1", owner="alice"))
        storage.add(job(job_id="J2", owner="bob"))
        found = storage.get_by_where({"owner": "bob"})
        assert [j.job_id for j in found] == ["J2"]


class TestSchema:
    """The declared indexes and unique must actually exist in the table.

    ``__create_unique_indexes`` only warns on failure and runs only at table-creation
    time, so these assert the index is really there rather than trusting a clean log.
    """

    @staticmethod
    def _query_schema(storage, sql: str, parameters: tuple = ()) -> list:
        connection = sqlite3.connect(str(storage._get_db_file_path()))
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    @classmethod
    def _index_statements(cls, storage) -> list:
        return cls._query_schema(
            storage,
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (storage.table_name,),
        )

    @pytest.mark.parametrize(
        "column",
        ["job_id", "space_name", "owner", "started_at"],
    )
    def test_column_is_indexed(self, storage, column):
        storage.add(job())
        statements = " ".join(sql or "" for _, sql in self._index_statements(storage))
        assert f"({column})" in statements, f"{column} is not indexed: {statements}"

    def test_unique_on_job_id_exists(self, storage):
        """It can only be created with the table, never added later."""
        storage.add(job())
        statements = [sql or "" for _, sql in self._index_statements(storage)]
        unique = [s for s in statements if "UNIQUE" in s and "job_id" in s]
        assert unique, statements
        assert "(job_id)" in unique[0]

    def test_every_promoted_column_is_text(self, storage):
        """get_by_where only builds an IN clause for string columns; anything else
        silently degrades to ``column == [list]`` -- a predicate that returns
        plausible but wrong rows with no error.

        So the invariant is that *every* promoted column is text, which makes that
        failure unreachable rather than merely avoided by convention.
        """
        storage.add(job())
        columns = {
            name: kind
            for _, name, kind, *_ in self._query_schema(
                storage, f"PRAGMA table_info('{storage.table_name}')"
            )
        }
        for column in (
            "job_id",
            "job_namespace",
            "space_name",
            "owner",
            "source_system",
            "status",
            "started_at",
        ):
            assert columns[column].startswith("VARCHAR"), (column, columns[column])

        non_text = {
            name: kind
            for name, kind in columns.items()
            # "index" is the autoincrement PK and "json" is the blob; neither is
            # ever a query predicate.
            if name not in ("index", "json") and not kind.startswith("VARCHAR")
        }
        assert not non_text, f"a non-text promoted column is a latent bug: {non_text}"

    def test_started_at_is_text_not_a_timestamp(self, storage):
        """Ordering is lexicographic over the source's own spelling. Converting the
        column to DateTime would also break the all-text IN invariant above.
        """
        storage.add(job())
        columns = {
            name: kind
            for _, name, kind, *_ in self._query_schema(
                storage, f"PRAGMA table_info('{storage.table_name}')"
            )
        }
        assert columns["started_at"].startswith("VARCHAR")

    def test_no_endpoint_columns_exist(self, storage):
        """A job has many endpoint pairs, so neither belongs here. Naming a column
        ``source``/``target`` would also pick up the SQL layer's global wide-column
        width, which is keyed on column name across every table.
        """
        storage.add(job())
        columns = {
            name
            for _, name, *_ in self._query_schema(
                storage, f"PRAGMA table_info('{storage.table_name}')"
            )
        }
        assert "source" not in columns
        assert "target" not in columns

    def test_no_tags_column_yet(self, storage):
        """Tags are deliberately out of scope for this table for now."""
        storage.add(job())
        columns = {
            name
            for _, name, *_ in self._query_schema(
                storage, f"PRAGMA table_info('{storage.table_name}')"
            )
        }
        assert "tags" not in columns
