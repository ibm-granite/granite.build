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

import traceback
from typing import Optional, Self

from gbserver.storage.lh.space_storage import LhSpaceStorage
from gbserver.storage.shadowed.storage import BaseDualItemStorage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.sql.space_storage import SQLSpaceStorage
from gbserver.storage.stored_space import StoredSpace
from gbserver.utils.unwrap_errors import get_readable_error_message


class BaseDualSpaceStorage(BaseDualItemStorage, IStoredSpaceStorage):

    def __init__(self: Self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_by_name(self, name: str) -> Optional[StoredSpace]:
        r = self.primary.get_by_name(name)  # type: ignore[union-attr]
        try:
            self.secondary.get_by_name(name)  # type: ignore[union-attr]
        except Exception as e:
            err_stack = traceback.format_exc()
            body = get_readable_error_message(e=e, err_stack=err_stack)
            self.logger.error(
                f"Secondary storage got exception after success on the primary. {body}"
            )
        return r


class LhSQLSpaceStorage(BaseDualSpaceStorage, IStoredSpaceStorage):

    def __init__(self: Self, **kwargs) -> None:
        kwargs["primary_class"] = LhSpaceStorage
        kwargs["secondary_class"] = SQLSpaceStorage
        super().__init__(**kwargs)


class SQLLhSpaceStorage(BaseDualSpaceStorage, IStoredSpaceStorage):

    def __init__(self: Self, **kwargs) -> None:
        kwargs["primary_class"] = SQLSpaceStorage
        kwargs["secondary_class"] = LhSpaceStorage
        super().__init__(**kwargs)
