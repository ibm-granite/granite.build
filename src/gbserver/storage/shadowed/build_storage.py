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

from gbserver.storage.build_storage import IStoredBuildStorage
from gbserver.storage.lh.build_storage import LhBuildStorage
from gbserver.storage.shadowed.storage import BaseDualItemStorage
from gbserver.storage.sql.build_storage import SQLBuildStorage


class LhSQLBuildStorage(BaseDualItemStorage, IStoredBuildStorage):

    def __init__(self: Self, **kwargs) -> None:
        kwargs["primary_class"] = LhBuildStorage
        kwargs["secondary_class"] = SQLBuildStorage
        super().__init__(**kwargs)


class SQLLhBuildStorage(BaseDualItemStorage, IStoredBuildStorage):

    def __init__(self: Self, **kwargs) -> None:
        kwargs["primary_class"] = SQLBuildStorage
        kwargs["secondary_class"] = LhBuildStorage
        super().__init__(**kwargs)
