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

"""The trial id and its foreign key are ``VARCHAR(16)`` wide."""

from __future__ import annotations

from sqlalchemy import String

from autotunex.db.tables.results import ResultTable
from autotunex.db.tables.trials import TrialTable


def test_trial_id_column_is_16_wide() -> None:
    column_type = TrialTable.__table__.c.id.type

    assert isinstance(column_type, String)
    assert column_type.length == 16


def test_result_trial_id_column_is_16_wide() -> None:
    column_type = ResultTable.__table__.c.trial_id.type

    assert isinstance(column_type, String)
    assert column_type.length == 16
