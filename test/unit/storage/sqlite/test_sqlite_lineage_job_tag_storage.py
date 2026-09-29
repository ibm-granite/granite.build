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

"""Storage-level tests for lineage job tags against SQLite."""

import os
import uuid as uuid_module

import pytest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_lineage_job_tag import (
    MAX_TAG_LENGTH,
    StoredLineageJobTag,
    is_storable_tag,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="storage")
def storage_fixture():
    table = f"t_ljtag_{uuid_module.uuid4().hex[:8]}"
    storage = SqliteStorageFactory().create_lineage_job_tag_storage(table_name=table)
    for job_id, tags in {
        "J1": ["build_id=B1", "space_name=s", "team=nlp"],
        "J2": ["build_id=B1", "space_name=s"],
        "J3": ["build_id=B2", "team=nlp"],
    }.items():
        for tag in tags:
            storage.add(StoredLineageJobTag(job_id=job_id, tag=tag))
    return storage


def test_any_of_is_an_or(storage):
    assert storage.get_job_ids_by_tags(["build_id=B1", "build_id=B2"]) == {
        "J1",
        "J2",
        "J3",
    }


def test_all_of_is_an_and(storage):
    assert storage.get_job_ids_by_tags(["build_id=B1"], all_of=["team=nlp"]) == {"J1"}
    assert storage.get_job_ids_by_tags([], all_of=["team=nlp", "space_name=s"]) == {
        "J1"
    }


def test_match_is_exact_not_substring(storage):
    assert storage.get_job_ids_by_tags(["build_id=B"]) == set()
    assert storage.get_job_ids_by_tags(["team=nl"]) == set()


def test_empty_filter_matches_nothing(storage):
    assert storage.get_job_ids_by_tags([]) == set()
    assert storage.get_job_ids_by_tags(["", ""], all_of=[]) == set()


def test_duplicate_tag_is_rejected(storage):
    with pytest.raises(Exception):
        storage.add(StoredLineageJobTag(job_id="J1", tag="team=nlp"))
    assert storage.get_tags(["J1"])["J1"].count("team=nlp") == 1


def test_get_tags_groups_by_job(storage):
    assert storage.get_tags(["J2", "J3", "missing"]) == {
        "J2": ["build_id=B1", "space_name=s"],
        "J3": ["build_id=B2", "team=nlp"],
    }
    assert storage.get_tags([]) == {}


def test_a_tag_at_the_column_width_round_trips(storage):
    tag = "k=" + "x" * (MAX_TAG_LENGTH - 2)
    storage.add(StoredLineageJobTag(job_id="J9", tag=tag))
    assert storage.get_job_ids_by_tags([tag]) == {"J9"}


def test_is_storable_tag():
    assert is_storable_tag("k=v")
    assert not is_storable_tag("")
    assert not is_storable_tag(None)
    assert not is_storable_tag("x" * (MAX_TAG_LENGTH + 1))
