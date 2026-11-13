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

"""Lineage SQL storage tests against the hosted PostgreSQL (GBSERVER_SQL_* from the environment).

The test bodies live in test/unit/storage/postgres/test_postgres_target_finished_after.py,
which runs them against the local test PostgreSQL.
"""

from libgbtest.local_postgres import HOSTED_POSTGRES_TEST_MARKS
from unit.storage.postgres.test_postgres_target_finished_after import *  # noqa: F401,F403

pytestmark = HOSTED_POSTGRES_TEST_MARKS
