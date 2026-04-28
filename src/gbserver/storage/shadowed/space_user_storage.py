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
from typing import List, Optional

from gbserver.storage.lh.space_user_storage import LhSpaceUserStorage
from gbserver.storage.shadowed.storage import BaseDualItemStorage
from gbserver.storage.space_user_storage import ISpaceUserStorage
from gbserver.storage.sql.space_user_storage import SQLSpaceUserStorage
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.utils.unwrap_errors import get_readable_error_message


class BaseDualSpaceUserStorage(BaseDualItemStorage, ISpaceUserStorage):

    def get_by_space(self, space_name: str) -> List[StoredSpaceUser]:
        r = self.primary.get_by_space(space_name)
        try:
            self.secondary.get_by_space(space_name)
        except Exception as e:
            err_stack = traceback.format_exc()
            body = get_readable_error_message(e=e, err_stack=err_stack)
            self.logger.error(
                f"Secondary storage got exception after success on the primary. {body}"
            )
        return r

    def get_by_username(self, username: str) -> List[StoredSpaceUser]:
        r = self.primary.get_by_username(username)
        try:
            self.secondary.get_by_username(username)
        except Exception as e:
            err_stack = traceback.format_exc()
            body = get_readable_error_message(e=e, err_stack=err_stack)
            self.logger.error(
                f"Secondary storage got exception after success on the primary. {body}"
            )
        return r

    def get_by_space_and_username(
        self, space_name: str, username: str
    ) -> Optional[StoredSpaceUser]:
        r = self.primary.get_by_space_and_username(space_name, username)
        try:
            self.secondary.get_by_space_and_username(space_name, username)
        except Exception as e:
            err_stack = traceback.format_exc()
            body = get_readable_error_message(e=e, err_stack=err_stack)
            self.logger.error(
                f"Secondary storage got exception after success on the primary. {body}"
            )
        return r


class LhSQLSpaceUserStorage(BaseDualSpaceUserStorage, ISpaceUserStorage):

    def __init__(self, **kwargs) -> None:
        kwargs["primary_class"] = LhSpaceUserStorage
        kwargs["secondary_class"] = SQLSpaceUserStorage
        super().__init__(**kwargs)


class SQLLhSpaceUserStorage(BaseDualSpaceUserStorage, ISpaceUserStorage):

    def __init__(self, **kwargs) -> None:
        kwargs["primary_class"] = SQLSpaceUserStorage
        kwargs["secondary_class"] = LhSpaceUserStorage
        super().__init__(**kwargs)
