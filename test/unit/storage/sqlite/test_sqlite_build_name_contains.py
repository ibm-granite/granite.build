"""``Contains`` where-values on a real SQLite build table: case-insensitive
substring match on ``name``, with ``%``/``_`` taken literally, and ``count``
agreeing with ``get_by_where`` so paginated totals stay right."""

import os
import uuid as uuid_module

import pytest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.storage import Contains
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)

NAMES = ["granite-SFT-run", "sft_lora", "eval-100%", "evalX100", "other"]


@pytest.fixture(name="builds")
def builds_fixture():
    table = f"t_builds_{uuid_module.uuid4().hex[:8]}"
    storage = SqliteStorageFactory().create_build_storage(table_name=table)
    for name in NAMES:
        storage.add(
            StoredBuild.create(
                name=name,
                space_name="s",
                source_uri="",
                username="u",
                build_archive="dGVzdA==",
                status=Status.SUCCESS,
                targets=[],
                tags=[],
            )
        )
    yield storage
    storage.delete_table()


def _names(storage, where):
    return sorted(b.name for b in storage.get_by_where(where))


@pytest.mark.parametrize(
    "term, expected",
    [
        ("sft", ["granite-SFT-run", "sft_lora"]),
        ("SFT", ["granite-SFT-run", "sft_lora"]),
        ("100%", ["eval-100%"]),
        ("t_l", ["sft_lora"]),
        ("nomatch", []),
    ],
)
def test_contains_matches_substring_literally(builds, term, expected):
    where = {"name": Contains(term)}
    assert _names(builds, where) == expected
    assert builds.count(where) == len(expected)


def test_contains_combines_with_exact_filters(builds):
    assert _names(builds, {"name": Contains("sft"), "username": "u"}) == [
        "granite-SFT-run",
        "sft_lora",
    ]
    assert _names(builds, {"name": Contains("sft"), "username": "nobody"}) == []


def test_plain_string_is_still_exact(builds):
    assert _names(builds, {"name": "sft"}) == []
    assert _names(builds, {"name": "sft_lora"}) == ["sft_lora"]
