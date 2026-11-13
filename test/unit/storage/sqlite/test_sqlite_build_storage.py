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

import os

import integration.ibm.storage.sql.test_sql_build_storage as HIDE_FROM_PYTEST  # Doing it this way keeps pytest from also adding TestSQLBuildStorage tests as part of this file.
import pytest
from libgbtest.storage.build_storage import BuildStorageTestSupport

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory


class SqliteBuildStorageTestSupport(BuildStorageTestSupport):
    """Standalone-safe test support that doesn't require GitHub API."""

    def _get_build_user(self) -> str:
        return "test-user"


@pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)
class TestSqliteBuildStorage(HIDE_FROM_PYTEST.TestSQLBuildStorage):

    @classmethod
    def _get_storage_factory(cls):
        return SqliteStorageFactory()

    @classmethod
    def _get_test_config(cls) -> BuildStorageTestSupport:
        return SqliteBuildStorageTestSupport()


# @pytest.mark.skipif( os.environ.get("SKIP_SQL_ADMIN_TESTS","False").lower() == 'true', reason="Don't want to run this in CICD.")
# class TestSqliteLegacyStoredBuild(BaseLegacyStoredBuildTest):

#     @classmethod
#     def _get_storage_factory(cls):
#         # This makes it a SQL test.
#         return SqliteStorageFactory()
