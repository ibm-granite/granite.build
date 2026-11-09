"""``Contains`` where-values on a real SQLite build table; see
libgbtest.storage.build_name_contains for what is checked."""

import os
import uuid as uuid_module

import pytest
from libgbtest.storage import build_name_contains as checks

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="builds")
def builds_fixture():
    table = f"t_builds_{uuid_module.uuid4().hex[:8]}"
    storage = SqliteStorageFactory().create_build_storage(table_name=table)
    checks.populate(storage)
    yield storage
    storage.delete_table()


@pytest.mark.parametrize("term, expected", checks.CONTAINS_CASES)
def test_contains_matches_substring_literally(builds, term, expected):
    checks.check_contains(builds, term, expected)


def test_contains_combines_with_exact_filters(builds):
    checks.check_combines_with_exact_filters(builds)


def test_plain_string_is_still_exact(builds):
    checks.check_plain_string_is_still_exact(builds)
