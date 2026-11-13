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

"""SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The non-legacy test bodies live in
test/unit/storage/postgres/test_postgres_node_failure_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.node_failure_storage import BaseLegacyNodeFailureTest
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from unit.storage.postgres.test_postgres_node_failure_storage import (
    SQLNodeFailureStorageTests,
)

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLNodeFailureStorage(SQLNodeFailureStorageTests):
    """SQLNodeFailureStorage tests against the hosted PostgreSQL."""


class TestSQLLegacyNodeFailure(PostgresStorageTestMixin, BaseLegacyNodeFailureTest):
    """Reads the existing rows in the hosted database's real tables (no prefix)."""
