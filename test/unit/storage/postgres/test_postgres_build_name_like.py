# SPDX-License-Identifier: Apache-2.0

"""``Like`` where-values on a real PostgreSQL build table; see
libgbtest.storage.build_name_like for what is checked."""

import uuid as uuid_module

import pytest
from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage import build_name_like as checks

from gbserver.storage.sql.storage_factory import SQLStorageFactory

pytestmark = LOCAL_POSTGRES_TEST_MARKS


@pytest.fixture(name="builds")
def builds_fixture():
    table = f"t_builds_{uuid_module.uuid4().hex[:8]}"
    storage = SQLStorageFactory().create_build_storage(table_name=table)
    checks.populate(storage)
    yield storage
    storage.delete_table()


@pytest.mark.parametrize("wildcard, expected", checks.WILDCARD_CASES)
def test_name_wildcard_matches(builds, wildcard, expected):
    checks.check_wildcard(builds, wildcard, expected)


def test_like_combines_with_exact_filters(builds):
    checks.check_combines_with_exact_filters(builds)


def test_plain_string_is_still_exact(builds):
    checks.check_plain_string_is_still_exact(builds)


def test_like_on_non_string_column_fails_fast(builds):
    checks.check_non_string_column_fails_fast(builds)
