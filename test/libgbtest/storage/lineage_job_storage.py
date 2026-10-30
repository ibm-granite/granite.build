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

"""Shared tests for the lineage job table, ``gb_lineage_job``.

The bodies here are backend-agnostic and run against a prefixed copy of the admin
tables, so they never touch the real one:

- test/unit/storage/sqlite/test_sqlite_lineage_job_storage.py (SQLite)
- test/unit/storage/postgres/test_postgres_lineage_job_storage.py (local PostgreSQL)
- test/integration/ibm/storage/sql/test_sql_lineage_job_storage.py (hosted PostgreSQL)
"""

import time
from typing import Any
from unittest.mock import patch

import pytest
from libgbtest.storage.storage import AbstractStorageTest, AbstractStorageTestSupport
from sqlalchemy import text

from gbserver.lineage.attributes import PAYLOAD, build_job_attributes, payload_detail
from gbserver.storage.lineage_job_storage import BaseLineageJobStorage
from gbserver.storage.storage import BaseItemStorage, BaseStoredItem
from gbserver.storage.stored_lineage_job import StoredLineageJob

# A step config big enough that storing it per lineage row would be the problem this
# table solves: a job with 3 inputs and 2 outputs would hold six copies.
BIG_STEP_CONFIG = {
    "steps": [{"name": f"step-{i}", "args": ["--flag"] * 20} for i in range(12)]
}


def job(
    job_id: str = "J",
    job_namespace: str = "my-space/my-build",
    space_name: str = "my-space",
    owner: str = "alice",
    status: str = "SUCCEEDED",
    started_at: str = "2026-01-01 00:00:00",
    source_system: str = "granite.build",
    tags: list | None = None,
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
        status=status,
        started_at=started_at,
        tags=list(tags or []),
        attributes=attributes,
    )


class LineageJobStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="job_id")

    def _get_test_item(self, index: int) -> BaseStoredItem:
        # Zero-padded so the job_id sort order matches the index order.
        return job(
            job_id=f"job{index:04d}",
            job_namespace=f"space{index}/build{index}",
            space_name=f"space{index}",
            owner=f"owner{index}",
            started_at=f"2026-01-01 00:00:{index:02d}",
            tags=[f"tag{index}"],
        )


class BaseLineageJobStorageTest(AbstractStorageTest):
    """``gb_lineage_job`` via ``*LineageJobStorage``."""

    _db_stamped_fields = ["recorded_at"]

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return LineageJobStorageTestSupport()

    def _get_tested_storage(self) -> BaseLineageJobStorage:
        return self.storage.lineage_job_storage

    def _get_where_search_columns(
        self, storage: BaseItemStorage, item: BaseStoredItem
    ) -> dict[str, Any]:
        columns = super()._get_where_search_columns(storage, item)
        # Unique per item (see _get_where_test_item), so a search on it can't match more than one.
        del columns["job_id"]
        # Stamped by the database on every write, so no two items share it.
        del columns["recorded_at"]
        return columns

    def _get_where_test_item(self, index: int) -> BaseStoredItem:
        """The where tests add the same index twice; job_id is unique."""
        item = super()._get_where_test_item(index)
        assert isinstance(item, StoredLineageJob)
        item.job_id = item.uuid
        return item

    def _verify_equal(self, result, expected):
        """recorded_at is the database's stamp, never the caller's, so ignore it."""
        if isinstance(result, StoredLineageJob) and isinstance(
            expected, StoredLineageJob
        ):
            unstamped = dict.fromkeys(self._db_stamped_fields, "")
            return super()._verify_equal(
                result.model_copy(update=unstamped),
                expected.model_copy(update=unstamped),
            )
        return super()._verify_equal(result, expected)

    # get_job / get_jobs / get_jobs_by_id / has_job: every query is a single indexed
    # lookup keyed on job_id.

    def test_get_job(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        found = storage.get_job("J1")
        assert found is not None
        assert found.job_id == "J1"

    def test_get_job_absent(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        assert storage.get_job("nope") is None

    def test_get_job_empty_job_id(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        assert storage.get_job("") is None
        assert storage.get_jobs([]) == []
        assert storage.has_job("") is False

    def test_get_jobs(self):
        storage = self._get_tested_storage()
        for i in range(5):
            storage.add(job(job_id=f"J{i}"))
        found = storage.get_jobs(["J0", "J2", "J4"])
        assert sorted(j.job_id for j in found) == ["J0", "J2", "J4"]

    def test_get_jobs_drops_empties_and_duplicates(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        found = storage.get_jobs(["J1", "J1", "", "J1"])
        assert [j.job_id for j in found] == ["J1"]

    def test_get_jobs_omits_missing(self):
        """A row can precede or outlive its job record; a caller enriching rows it
        already holds must not get an error for the gap."""
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        found = storage.get_jobs(["J1", "missing"])
        assert [j.job_id for j in found] == ["J1"]

    def test_get_jobs_by_id(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        storage.add(job(job_id="J2"))
        by_id = storage.get_jobs_by_id(["J1", "J2", "missing"])
        assert sorted(by_id) == ["J1", "J2"]
        assert by_id["J1"].job_id == "J1"

    def test_has_job(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        assert storage.has_job("J1") is True
        assert storage.has_job("J2") is False

    # Uniqueness on job_id: one record per execution; re-recording is a no-op, not a
    # duplicate.

    def test_uniqueness_enforcement(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        with pytest.raises(Exception):
            storage.add(job(job_id="J1"))

    def test_uniqueness_ignores_attributes(self):
        """The unique is on job_id alone, so a second record for one execution is a
        duplicate however much its blob differs."""
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1", job_metadata={"job_name": "first"}))
        with pytest.raises(Exception):
            storage.add(job(job_id="J1", job_metadata={"job_name": "second"}))
        assert len(storage.get_jobs(["J1"])) == 1

    def test_uniqueness_allows_different_jobs(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1"))
        storage.add(job(job_id="J2"))
        assert len(storage.get_jobs(["J1", "J2"])) == 2

    # Payloads: the four large payloads are why this table exists.

    def test_round_trip_payloads(self):
        storage = self._get_tested_storage()
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

    def test_round_trip_no_payloads(self):
        """An omitted group lets a reader tell "not recorded" from "recorded
        empty"."""
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1", job_metadata={"job_name": "n"}))
        assert PAYLOAD not in storage.get_job("J1").attributes
        assert payload_detail(storage.get_job("J1").attributes) == {}

    def test_round_trip_job_detail_and_origin_ids(self):
        storage = self._get_tested_storage()
        storage.add(
            job(
                job_id="J1",
                job_metadata={"job_name": "train", "job_type": "BUILD"},
                ids={"build_id": "b1", "target_run_uuid": "t1"},
            )
        )
        attributes = storage.get_job("J1").attributes
        assert attributes["job"]["name"] == "train"
        assert attributes["origin"]["ids"] == {
            "build_id": "b1",
            "target_run_uuid": "t1",
        }

    # Fields: every promoted column survives storage as written.

    def test_round_trip_all_columns(self):
        storage = self._get_tested_storage()
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
        # source_system is now in the row table, not the job table;
        # it's stored in the attributes blob for provenance tracking
        assert found.attributes.get("origin", {}).get("system") == "lakehouse"

    def test_round_trip_source_without_a_space(self):
        """Lakehouse has no notion of a space or an owner; the columns hold "" rather
        than NULL so they behave under an IN like every other value."""
        storage = self._get_tested_storage()
        storage.add(
            job(job_id="J1", space_name="", owner="", job_namespace="", started_at="")
        )
        found = storage.get_job("J1")
        assert found.space_name == ""
        assert found.owner == ""
        assert found.job_namespace == ""

    def test_round_trip_default_attributes_empty(self):
        storage = self._get_tested_storage()
        storage.add(StoredLineageJob(job_id="J1"))
        assert storage.get_job("J1").attributes == {}

    def test_round_trip_started_at_verbatim(self):
        """The source's own string form, never rewritten to UTC."""
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1", started_at="2026-01-02 03:04:05.678901"))
        assert storage.get_job("J1").started_at == "2026-01-02 03:04:05.678901"

    def test_schema_attributes_not_queryable(self):
        """The blob is Text, so nothing inside it can be a predicate.

        Asserted on the column list rather than by issuing a doomed query: a bad
        predicate goes through the retrying query path, which sleeps between attempts
        instead of failing fast.
        """
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1", job_metadata={"job_name": "findme"}))
        columns = storage.get_column_names()
        assert "job_name" not in columns
        assert "attributes" not in columns

    def test_schema_no_endpoint_columns(self):
        """A job has many endpoint pairs, so neither belongs here. Naming a column
        ``input``/``output`` would also pick up the SQL layer's global wide-column
        width, which is keyed on column name across every table.
        """
        storage = self._get_tested_storage()
        storage.add(job())
        columns = storage.get_column_names()
        assert "input" not in columns
        assert "output" not in columns

    def test_schema_tags_is_a_column(self):
        """Tags are promoted so a tag search is a SQL filter, not a blob scan."""
        storage = self._get_tested_storage()
        storage.add(job())
        assert "tags" in storage.get_column_names()

    # get_by_where on space_name and owner: indexed because they are how a listing is
    # scoped.

    def test_get_by_where_space(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1", space_name="a"))
        storage.add(job(job_id="J2", space_name="b"))
        found = storage.get_by_where({"space_name": "a"})
        assert [j.job_id for j in found] == ["J1"]

    def test_get_by_where_space_batch(self):
        """An access filter resolves the caller's spaces once, then narrows in SQL --
        which needs the IN clause get_by_where builds for string columns."""
        storage = self._get_tested_storage()
        for i, space in enumerate(["a", "b", "c"]):
            storage.add(job(job_id=f"J{i}", space_name=space))
        found = storage.get_by_where({"space_name": ["a", "c"]})
        assert sorted(j.space_name for j in found) == ["a", "c"]

    def test_get_by_where_owner(self):
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1", owner="alice"))
        storage.add(job(job_id="J2", owner="bob"))
        found = storage.get_by_where({"owner": "bob"})
        assert [j.job_id for j in found] == ["J2"]

    # search_by_tags answers what W&B's ``{"tags": {"$in": [...]}}`` answers.
    #
    # The distinction that matters: ``any_of`` is a DISJUNCTION. The storage layer's
    # generic list-column filter is an AND, so a search for two tags no single job
    # carries would return nothing -- which is the wrong answer to "jobs tagged X or
    # Y", and would make ``POST /lineage/search`` mean different things per provider.

    def _tagged(self) -> BaseLineageJobStorage:
        storage = self._get_tested_storage()
        storage.add(job(job_id="J1", tags=["team=nlp", "build_id=B1"]))
        storage.add(job(job_id="J2", tags=["team=vision", "build_id=B2"]))
        storage.add(job(job_id="J3", tags=["team=nlp", "build_id=B3", "phase=eval"]))
        return storage

    def test_search_by_tags_any_of(self):
        """The regression guard: an AND implementation returns 0 here."""
        total, page = self._tagged().search_by_tags(["team=nlp", "team=vision"])
        assert total == 3
        assert {j.job_id for j in page} == {"J1", "J2", "J3"}

    def test_search_by_tags_any_of_one_tag(self):
        total, page = self._tagged().search_by_tags(["team=vision"])
        assert (total, [j.job_id for j in page]) == (1, ["J2"])

    def test_search_by_tags_all_of(self):
        total, page = self._tagged().search_by_tags(
            [], all_of=["team=nlp", "phase=eval"]
        )
        assert (total, [j.job_id for j in page]) == (1, ["J3"])

    def test_search_by_tags_all_of_unsatisfied(self):
        tagged = self._tagged()
        assert tagged.search_by_tags([], all_of=["team=nlp", "team=vision"])[0] == 0

    def test_search_by_tags_any_of_and_all_of(self):
        total, page = self._tagged().search_by_tags(["team=nlp"], all_of=["phase=eval"])
        assert (total, [j.job_id for j in page]) == (1, ["J3"])

    def test_search_by_tags_no_tags(self):
        """An empty tag list is what the route's "match everything" means."""
        assert self._tagged().search_by_tags([])[0] == 3

    def test_search_by_tags_prefix_not_a_match(self):
        """LIKE over the joined column over-matches; the exact re-check catches it.

        ``%team=nlp%`` also hits ``team=nlp-eval`` in SQL, so a result that skipped
        the per-item check would report a job the caller did not ask for.
        """
        tagged = self._tagged()
        tagged.add(job(job_id="J4", tags=["team=nlp-eval"]))
        total, page = tagged.search_by_tags(["team=nlp"])
        assert "J4" not in {j.job_id for j in page}
        assert total == 2

    def test_search_by_tags_unknown_tag(self):
        assert self._tagged().search_by_tags(["team=nope"]) == (0, [])

    def test_search_by_tags_total_counts_matches(self):
        """So a caller can page: the page is capped, the total is not."""
        total, page = self._tagged().search_by_tags([], limit=2)
        assert total == 3
        assert len(page) == 2

    def test_search_by_tags_paging(self):
        tagged = self._tagged()
        seen = []
        for offset in (0, 2):
            seen += [
                j.job_id for j in tagged.search_by_tags([], limit=2, offset=offset)[1]
            ]
        assert sorted(seen) == ["J1", "J2", "J3"]

    def test_search_by_tags_empty_tag_string(self):
        """Otherwise a blank would LIKE-match every row and widen the search."""
        assert self._tagged().search_by_tags(["", "team=vision"])[0] == 1

    # recorded_at: stamped by the database inside the write, and the indexer's
    # checkpoint.

    def _qualified_table(self, storage) -> str:
        schema = getattr(storage, "_db_schema", None)
        return f"{schema}.{storage.table_name}" if schema else storage.table_name

    def test_get_recorded_after(self):
        storage = self._get_tested_storage()
        stamps = {
            "J3": "2026-01-02T00:00:00.000000+00:00",
            "J1": "2026-01-01T00:00:00.000000+00:00",
            "J2": "2026-01-01T00:00:00.000000+00:00",  # ties J1: job_id orders them
            "J0": "2025-12-31T00:00:00.000000+00:00",
        }
        for job_id in stamps:
            storage.add(job(job_id))
        # The database stamps recorded_at on write; set known values directly.
        with storage._engine.begin() as conn:  # pylint: disable=protected-access
            for job_id, recorded_at in stamps.items():
                conn.execute(
                    text(
                        f"UPDATE {self._qualified_table(storage)} SET recorded_at = :v "
                        "WHERE job_id = :j"
                    ),
                    {"v": recorded_at, "j": job_id},
                )
        cutoff = "2026-01-01T00:00:00.000000+00:00"
        first = storage.get_recorded_after(cutoff, None, 2)
        assert [j.job_id for j in first] == ["J1", "J2"]
        last = first[-1]
        rest = storage.get_recorded_after(cutoff, (last.recorded_at, last.job_id), 2)
        assert [j.job_id for j in rest] == ["J3"]
        assert [j.job_id for j in storage.get_recorded_after(None, None, 10)] == [
            "J0",
            "J1",
            "J2",
            "J3",
        ]

    def test_recorded_at_stamped_by_database(self):
        """The caller's value is ignored: the database stamps the write, in
        utc_now_iso's fixed-width UTC form, and the JSON blob does not keep a copy."""
        from datetime import datetime, timedelta, timezone

        storage = self._get_tested_storage()
        record = job("J1")
        record.recorded_at = "1999-01-01T00:00:00.000000+00:00"
        storage.add(record)
        stored = storage.get_job("J1")
        assert len(stored.recorded_at) == len("2026-01-01T00:00:00.000000+00:00")
        stamped = datetime.fromisoformat(stored.recorded_at)
        assert abs(datetime.now(timezone.utc) - stamped) < timedelta(minutes=1)
        with storage._engine.connect() as conn:  # pylint: disable=protected-access
            blob = conn.execute(
                text(
                    f"SELECT json FROM {self._qualified_table(storage)} "
                    "WHERE job_id = 'J1'"
                )
            ).scalar()
        assert "recorded_at" not in blob

    def test_recorded_at_is_a_database_expression(self):
        """recorded_at reaches the INSERT as SQL the database evaluates, so every retry
        of the statement is stamped at its own attempt rather than at a time fixed
        before the first one."""
        from sqlalchemy.sql.elements import ClauseElement

        from gbserver.storage.sql.sql_storage import BaseSQLItemStorage

        storage = self._get_tested_storage()
        seen = []
        real = BaseSQLItemStorage._add_item_dicts

        def capture(self, items):
            seen.extend(r["recorded_at"] for r in items)
            return real(self, items)

        with patch.object(BaseSQLItemStorage, "_add_item_dicts", capture):
            storage.add(job("J1"))
        assert seen and all(isinstance(v, ClauseElement) for v in seen)

    def test_update_fields_moves_recorded_at_forward(self):
        storage = self._get_tested_storage()
        storage.add(job("J1"))
        before = storage.get_job("J1")
        time.sleep(0.01)
        storage.update_fields(before.uuid, {"tags": ["k=v"]})
        after = storage.get_job("J1")
        assert after.tags == ["k=v"]
        assert after.recorded_at > before.recorded_at

    def test_update_fields_failure_leaves_recorded_at(self):
        storage = self._get_tested_storage()
        storage.add(job("J1"))
        before = storage.get_job("J1")
        with pytest.raises(ValueError):
            storage.update_fields(before.uuid, {"no_such_field": 1})
        after = storage.get_job("J1")
        assert after.recorded_at == before.recorded_at
        assert after.tags == before.tags
