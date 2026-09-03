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

"""
Storing space metadata in persistent storage.
"""

from typing import Optional

from gbserver.storage.storage import BaseStoredItem


class StoredSpace(BaseStoredItem):
    """
    This class is used to store the full details of a space.
    Each instance of this class becomes a row in the gb_spaces table in persistent storage.
    """

    name: str
    git_repo_uri: str

    # TODO: this should not be here. instead from the space.yaml.
    # Optional so that records stored without it (and future records that omit
    # it) deserialize cleanly in both directions.
    lakehouse_namespace: Optional[str] = None

    # Cached HuggingFace Enterprise resource group id for this space's DEFAULT
    # resource group (the ``gbspace-<space>`` group derived from the space name).
    # Populated from create-spaces YAML or lazily written back after a successful
    # HF API lookup (see gbserver.spaces.hf_push_config). Only the default group's
    # id is cached here; an explicitly-requested non-default group is never stored.
    # Stored inside the existing ``json`` column, so no schema migration / new SQL
    # column is required.
    hf_default_resource_group_id: Optional[str] = None
