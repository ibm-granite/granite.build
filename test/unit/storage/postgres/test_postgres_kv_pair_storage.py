# Copyright Granite.Build Authors
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

from libgbtest.local_postgres import LOCAL_POSTGRES_TEST_MARKS
from libgbtest.storage.kv_pair_storage import (
    BaseKeyValuePairStorageTest,
)
from libgbtest.storage.kv_pair_storage import (
    TestKeyValuePairValueMethods as _TestKeyValuePairValueMethods,
)
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin

# The test bodies are mixins (SQL*Tests) shared with the IBM copy in
# test/integration/ibm/storage/sql/test_sql_kv_pair_storage.py,
# which runs them against the hosted PostgreSQL; here they run against the
# local test PostgreSQL.
pytestmark = LOCAL_POSTGRES_TEST_MARKS


class SQLKeyValuePairStorageTests(
    PostgresStorageTestMixin, BaseKeyValuePairStorageTest
):
    """SQLKeyValuePairStorage tests, shared with the IBM copy (see TestSQLKeyValuePairStorage below)."""


class SQLKeyValuePairValueMethodsTests(
    PostgresStorageTestMixin, _TestKeyValuePairValueMethods
):
    """SQLKeyValuePairValueMethods tests, shared with the IBM copy (see TestSQLKeyValuePairValueMethods below)."""


class TestSQLKeyValuePairStorage(SQLKeyValuePairStorageTests):
    """SQLKeyValuePairStorage tests against the local test PostgreSQL."""


class TestSQLKeyValuePairValueMethods(SQLKeyValuePairValueMethodsTests):
    """SQLKeyValuePairValueMethods tests against the local test PostgreSQL."""
