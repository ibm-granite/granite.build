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

import datetime
from base64 import b64decode, b64encode
from typing import Optional, Self

from pydantic import BaseModel, Field

from gbserver.storage.storage import BaseStoredItem


class RepoSnapshot(BaseModel):
    """A snapshot of one repo's latest content, stored in a gb_spaces row.

    Like StoredBuild.build_archive, ``archive`` is a zip encoded as base64. It
    holds only the tree at ``source_commit`` -- no git history.
    """

    # repr=False keeps the payload out of logged reprs.
    archive: Optional[str] = Field(default=None, repr=False)
    # Normalized https://host/owner/repo form (see gbserver.spaces.repo_identity);
    # for the space snapshot, the same repo as git_repo_uri.
    source_uri: Optional[str] = None
    source_ref: Optional[str] = None  # branch the snapshot was taken from
    source_commit: Optional[str] = None
    snapshot_time: Optional[datetime.datetime] = None

    def load_archive(self: Self) -> bytes:
        """Decode the archive to zip bytes (empty if unset)."""
        return b64decode(self.archive or "")

    def save_archive(self: Self, data: bytes) -> None:
        """Encode zip bytes into the archive."""
        self.archive = b64encode(data).decode("utf-8")


class StoredSpace(BaseStoredItem):
    """
    This class is used to store the full details of a space.
    Each instance of this class becomes a row in the gb_spaces table in persistent storage.
    """

    name: str
    # The space's git repo: its identity for secrets, build PRs and clients.
    # Historically also where the space is loaded from; see space_uri.
    git_repo_uri: str

    # Where the space definitions are loaded from, when not git_repo_uri:
    # ``db://spaces/<name>`` serves them from the snapshots below. Kept in the
    # ``json`` column (no schema migration); unset means git_repo_uri.
    space_uri: Optional[str] = None

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

    # Snapshots of the space repo and the assets repo its space.yaml base_uris
    # point at, served when space_uri is a db:// URI (gbserver.spaces.db_space).
    # Stored inside the existing ``json`` column; no schema migration.
    space: Optional[RepoSnapshot] = None
    assets: Optional[RepoSnapshot] = None

    def get_space_uri(self: Self) -> str:
        """Where to load the space definitions from (space_uri, else git_repo_uri)."""
        return self.space_uri or self.git_repo_uri

    def without_archives(self: Self) -> Self:
        """Copy with the archive payloads dropped; source metadata is kept."""
        update = {
            key: snap.model_copy(update={"archive": None})
            for key, snap in (("space", self.space), ("assets", self.assets))
            if snap is not None
        }
        return self.model_copy(update=update)
