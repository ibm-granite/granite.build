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

from datetime import datetime
from typing import Optional, Self

from gbserver.storage.storage import BaseStoredItem
from gbserver.types.status import Status


class StoredStepRun(BaseStoredItem):
    # Required initializations
    build_id: str
    target_id: str
    definition_uri: str

    # Defaulting initializations
    status: Status = Status.PENDING
    status_msg: str = ""
    config: dict = {}
    # Runtime key/values produced by the step itself (e.g. a resolved git commit
    # SHA), pushed via the GB_STEP_METADATA_KEY/VALUE stdout hook and merged by
    # the buildrunner. Kept separate from `config` (the rendered build.yaml input)
    # so step-generated data never mutates the declared configuration. Serialized
    # into the row's JSON blob like `config`, so it needs no column/migration and
    # old rows deserialize to {}.
    metadata: dict = {}
    config_dir: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)
