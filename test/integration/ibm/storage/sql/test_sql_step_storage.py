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
test/unit/storage/postgres/test_postgres_step_storage.py,
which runs them against the local test PostgreSQL. The legacy tests read existing rows in the hosted
database's real tables, so they exist only here.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from libgbtest.storage.postgres_storage import PostgresStorageTestMixin
from libgbtest.storage.step_storage import BaseLegacyStoredStepTest
from unit.storage.postgres.test_postgres_step_storage import SQLStepStorageTests

pytestmark = HOSTED_POSTGRES_TEST_MARKS


class TestSQLStepStorage(SQLStepStorageTests):
    """SQLStepStorage tests against the hosted PostgreSQL."""


class TestSQLLegacyStoredStep(PostgresStorageTestMixin, BaseLegacyStoredStepTest):
    """Reads the existing rows in the hosted database's real tables (no prefix)."""
