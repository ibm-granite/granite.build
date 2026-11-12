#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
    # Normalized https://host/owner/repo form (see gbserver.spaces.repo_identity).
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

    # Snapshots of the space repo and the assets repo its space.yaml base_uris
    # point at. Used when git_repo_uri is a db:// URI (gbserver.spaces.db_space).
    # Stored inside the existing ``json`` column; no schema migration.
    space: Optional[RepoSnapshot] = None
    assets: Optional[RepoSnapshot] = None

    def without_archives(self: Self) -> Self:
        """Copy with the archive payloads dropped; source metadata is kept."""
        update = {
            key: snap.model_copy(update={"archive": None})
            for key, snap in (("space", self.space), ("assets", self.assets))
            if snap is not None
        }
        return self.model_copy(update=update)
