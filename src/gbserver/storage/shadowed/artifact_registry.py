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
from typing import List, Optional, Self, Union

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.storage.lh.artifact_registry import LhArtifactRegistry
from gbserver.storage.shadowed.storage import BaseDualItemStorage
from gbserver.storage.sql.artifact_registry import SQLArtifactRegistry
from gbserver.utils.unwrap_errors import get_readable_error_message


class BaseDualArtifactRegistry(BaseDualItemStorage, IArtifactRegistry):

    def __init__(self: Self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_by_uri(
        self: Self, uri: str, space_name: str = ""
    ) -> Union[List[ArtifactRegistration], Optional[ArtifactRegistration]]:
        assert self.primary is not None, "the primary artifact registry is missing"
        assert isinstance(self.primary, IArtifactRegistry), f"invalid primary: {self.primary}"
        r = self.primary.get_by_uri(uri=uri, space_name=space_name)
        if self.secondary is not None:
            try:
                assert isinstance(
                    self.secondary, IArtifactRegistry
                ), f"invalid secondary: {self.secondary}"
                self.secondary.get_by_uri(uri=uri, space_name=space_name)
            except Exception as e:
                err_stack = traceback.format_exc()
                body = get_readable_error_message(e=e, err_stack=err_stack)
                self.logger.error(
                    f"Secondary storage got exception after success on the primary. {body}"
                )
        return r


class LhSQLArtifactRegistry(BaseDualArtifactRegistry, IArtifactRegistry):

    def __init__(self: Self, **kwargs) -> None:
        kwargs["primary_class"] = LhArtifactRegistry
        kwargs["secondary_class"] = SQLArtifactRegistry
        super().__init__(**kwargs)


class SQLLhArtifactRegistry(BaseDualArtifactRegistry, IArtifactRegistry):

    def __init__(self: Self, **kwargs) -> None:
        kwargs["primary_class"] = SQLArtifactRegistry
        kwargs["secondary_class"] = LhArtifactRegistry
        super().__init__(**kwargs)
