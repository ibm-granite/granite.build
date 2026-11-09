"""``Contains`` where-values on a real PostgreSQL build table; see
libgbtest.storage.build_name_contains for what is checked."""

import uuid as uuid_module

import pytest
from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage import build_name_contains as checks

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = LOCAL_POSTGRES_TEST_MARKS


@pytest.fixture(name="builds")
def builds_fixture():
    table = f"t_builds_{uuid_module.uuid4().hex[:8]}"
    storage = SQLStorageFactory().create_build_storage(table_name=table)
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
