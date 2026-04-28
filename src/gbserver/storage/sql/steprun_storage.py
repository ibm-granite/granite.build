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

from typing import Self

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.steprun_storage import (
    BaseStoredStepRunStorage,
    IStoredStepRunStorage,
)
from gbserver.storage.stored_step_run import StoredStepRun


class SQLStepRunStorage(
    BaseSQLItemStorage[StoredStepRun], BaseStoredStepRunStorage, IStoredStepRunStorage
):

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)
