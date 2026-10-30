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

"""delete_table on a storage that never touched its table.

A test that swaps the storage factory keeps storage objects that never built their
model, while another instance created the table. Their teardown delete_table must
drop it in one attempt: a failing attempt is retried with backoff while the SQLite
file lock is held, stalling every other storage call in the process for minutes.
"""

from unittest.mock import patch

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_space import StoredSpace

TABLE = "test_delete_table_unused"


def test_an_unused_instance_drops_the_table_without_retrying():
    SqliteStorageFactory().create_space_storage(table_name=TABLE).add(
        StoredSpace(name="s1", git_repo_uri="file:///s1")
    )
    unused = SqliteStorageFactory().create_space_storage(table_name=TABLE)
    assert unused._sql_alchemy_model is None  # pylint: disable=protected-access

    # Patch tenacity's time reference only (see test_sqlite_get_by_where_retry).
    with patch("tenacity.nap.time") as tenacity_time:
        unused.delete_table()

    tenacity_time.sleep.assert_not_called()
    fresh = SqliteStorageFactory().create_space_storage(table_name=TABLE)
    assert not fresh._does_table_exist()  # pylint: disable=protected-access
